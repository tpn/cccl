# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower straight-line asynchronous shared copies with single-use handles."""

from __future__ import annotations

import math
import operator
from typing import Any

import numba_cuda_mlir.numba_cuda.types as numba_types
from numba_cuda_mlir.cuda.shared import array as shared_array
from numba_cuda_mlir.numba_cuda.core.analysis import compute_cfg_from_blocks
from numba_cuda_mlir.numba_cuda.np.numpy_support import as_dtype

from .._temp_storage import TempStorage
from .._thread_data import ThreadData
from ._group_planner_support import (
    GroupRewriteError,
    _group_operation_name,
    ir,
)
from ._parameters import _validate_common_numeric_dtype, normalize_dtype_param


def _error(message):
    return GroupRewriteError(f"load_to_shared_async: {message}")


def _array_origin(planner, value, seen=None):
    """Resolve array aliases/slices to their allocation, without guessing."""

    seen = set() if seen is None else seen
    if not isinstance(value, ir.Var) or value.name in seen:
        return None
    seen.add(value.name)
    definition = planner._definition(value)
    if isinstance(definition, ir.Var):
        return _array_origin(planner, definition, seen)
    if isinstance(definition, ir.Expr):
        if definition.op in {"getitem", "static_getitem"}:
            index = definition.index
            resolved, selected = planner._try_constant(index)
            if resolved and isinstance(selected, slice):
                step = selected.step
            else:
                selected = planner._definition(index)
                if not (
                    isinstance(selected, ir.Expr)
                    and selected.op == "call"
                    and planner._callable(selected.func) is slice
                ):
                    raise _error(
                        "memory operands must be one-dimensional "
                        "contiguous arrays"
                    )
                step = (
                    planner.context.constant(selected.args[2])
                    if len(selected.args) == 3
                    else None
                )
            if step not in (None, 1):
                raise _error(
                    "memory operands must be one-dimensional contiguous arrays"
                )
            return _array_origin(planner, definition.value, seen)
        if definition.op == "cast":
            return _array_origin(planner, definition.value, seen)
    return definition


def _declared_dtype(planner, call):
    reference = (
        call.args[1] if len(call.args) > 1 else dict(call.kws).get("dtype")
    )
    resolved, dtype = planner._try_constant(reference)
    if resolved and dtype is not None:
        return normalize_dtype_param(dtype)
    definition = planner._definition(reference)
    if (
        isinstance(definition, ir.Expr)
        and definition.op == "getattr"
        and definition.attr == "dtype"
    ):
        return planner.context.dtype(definition.value)
    return None


def _shared_dtype(planner, value):
    origin = _array_origin(planner, value)
    if not isinstance(origin, ir.Expr) or origin.op != "call":
        raise _error("shared_buffer must come from shared-memory storage")
    if planner._callable(origin.func) is shared_array:
        keywords = dict(origin.kws)
        alignment = planner.context.constant(
            origin.args[2]
            if len(origin.args) >= 3
            else keywords.get("alignment", keywords.get("alignas", 8))
        )
        if not isinstance(alignment, int) or alignment < 16:
            raise _error(
                "native shared_buffer requires "
                "cuda.shared.array(..., alignment=16)"
            )
        shape = planner.context.constant(
            origin.args[0] if origin.args else dict(origin.kws)["shape"]
        )
        if isinstance(shape, tuple):
            if len(shape) != 1:
                raise _error("shared_buffer must be one-dimensional")
            shape = shape[0]
        if not isinstance(shape, int) or shape <= 0:
            raise _error(
                "native shared_buffer requires a positive static extent"
            )
        return _declared_dtype(planner, origin)
    method = planner._definition(origin.func)
    if (
        isinstance(method, ir.Expr)
        and method.op == "getattr"
        and method.attr == "reserve"
        and planner.context.temp_storage(method.value) is not None
    ):
        return _declared_dtype(planner, origin)
    raise _error("shared_buffer must come from shared-memory storage")


class _Emitter:
    """Build ordinary compiler IR for a validated issue/wait pair."""

    def __init__(self, planner, inst):
        self.planner = planner
        self.scope = inst.target.scope
        self.loc = inst.loc
        self.statements: list[Any] = []

    def value(self, value, stem="copy_arg"):
        return self.planner._value_var(
            self.statements,
            scope=self.scope,
            loc=self.loc,
            stem=stem,
            value=value,
        )

    def assign(self, expression, stem, target=None):
        result = target or self.planner._new_var(self.scope, self.loc, stem)
        self.statements.append(ir.Assign(expression, result, self.loc))
        return result

    def attr(self, value, name):
        return self.assign(ir.Expr.getattr(value, name, self.loc), name)

    def call(self, function, args=(), kwargs=None, target=None):
        function = self.value(function, "copy_function")
        args = [self.value(arg) for arg in args]
        keywords = tuple(
            (name, self.value(value)) for name, value in (kwargs or {}).items()
        )
        return self.assign(
            ir.Expr.call(function, args, keywords, self.loc),
            "copy_result",
            target,
        )


def _lower_pair(planner, issue, wait):
    from .._lowering._load_to_shared import make_shared_copy_invocables

    call = issue.value
    function = planner._callable(call.func)
    bound = planner._bind(function, call)
    args = bound.arguments
    group = planner._group(args["group"])
    if group is None or group.kind != "block" or group.mapping is not None:
        raise _error("requires a complete block")
    group = planner._resolve_group(group, feature="load_to_shared_async")
    block_dim = group.hierarchy.block_dim
    if block_dim is None or math.prod(block_dim) % 32:
        raise _error("block size must be a multiple of 32")
    count = planner.context.constant(args["num_items"])
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise _error("num_items must be a nonnegative compile-time integer")
    source, buffer = args["source"], args["shared_buffer"]
    source_origin = _array_origin(planner, source)
    if not isinstance(source_origin, ir.Arg):
        raise _error("source must come from a global-memory array argument")
    source_type = planner.state.args[source_origin.index]
    if not (
        isinstance(source_type, numba_types.Array)
        and source_type.ndim == 1
        and source_type.layout == "C"
    ):
        raise _error("source must be a one-dimensional contiguous array")
    dtype = planner.context.dtype(source)
    if dtype is None:
        raise _error("could not infer the source dtype")
    dtype = _validate_common_numeric_dtype(
        dtype, operation="load_to_shared_async"
    )
    if _shared_dtype(planner, buffer) != dtype:
        raise _error("source and shared_buffer must have the same dtype")
    if count * as_dtype(dtype).itemsize > 2**31 - 1 - 31:
        raise _error(
            "num_items padded byte capacity must fit a signed 32-bit integer"
        )
    offset = args["offset"]
    resolved, static_offset = planner._try_constant(offset)
    if resolved and (
        isinstance(static_offset, bool)
        or not isinstance(static_offset, int)
        or static_offset < 0
    ):
        raise _error("offset must be a nonnegative integer")
    if not resolved:
        offset_dtype = planner.context.dtype(offset)
        if offset_dtype is not None and not isinstance(
            offset_dtype, numba_types.Integer
        ):
            raise _error("offset must be an integer")
    issue_fn, wait_fn, context_layout, barrier_layout = (
        make_shared_copy_invocables(
            dtype, block_dim, count, planner._provider_compile_context()
        )
    )

    emit = _Emitter(planner, issue)
    context = emit.call(
        ThreadData,
        (context_layout[0], numba_types.uint8),
        {"alignment": context_layout[1]},
    )
    owner = emit.call(TempStorage)
    reserve = emit.attr(owner, "reserve")
    barrier = emit.call(
        reserve,
        (barrier_layout[0], numba_types.uint8),
        {"alignment": barrier_layout[1]},
    )
    source_size = emit.attr(source, "size")
    buffer_size = emit.attr(buffer, "size")
    emit.call(
        issue_fn,
        (context, barrier, source, source_size, buffer, buffer_size, offset),
        target=issue.target,
    )
    planner.replacements[issue] = emit.statements
    planner.dead_func_names.add(call.func.name)

    emit = _Emitter(planner, wait)
    start = emit.call(wait_fn, (context,))
    stop = emit.assign(
        ir.Expr.binop(operator.add, start, emit.value(count), emit.loc),
        "copy_stop",
    )
    selected = emit.call(slice, (start, stop))
    emit.assign(
        ir.Expr.getitem(buffer, selected, emit.loc), "copy_view", wait.target
    )
    planner.replacements[wait] = emit.statements


def plan_shared_copies(planner):
    """Consume each handle exactly once in the outer control-flow region.

    Handles only support local aliases and a direct wait. Reject control-flow
    placement, escapes and missing/duplicate waits before allocating storage
    or compiling providers. Dedicated barrier descriptors keep outstanding
    copies independent of normal scratch reuse.
    """

    statements = [
        (label, inst)
        for label, block in planner.func_ir.blocks.items()
        for inst in block.body
    ]
    issues = {
        inst: label
        for label, inst in statements
        if isinstance(inst, ir.Assign)
        and isinstance(inst.value, ir.Expr)
        and inst.value.op == "call"
        and _group_operation_name(planner._callable(inst.value.func))
        == "load_to_shared_async"
    }
    if not issues:
        return
    cfg = compute_cfg_from_blocks(planner.func_ir.blocks)
    outer = {label for label in cfg.backbone() if not cfg.in_loops(label)}
    if any(label not in outer for label in issues.values()):
        raise _error("issue and wait must be in the kernel outer region")
    handles = {inst.target.name: inst for inst in issues}
    aliases = set()
    changed = True
    while changed:
        changed = False
        for label, inst in statements:
            if not isinstance(inst, ir.Assign) or inst.target.name in handles:
                continue
            value = inst.value
            if isinstance(value, ir.Expr) and value.op == "cast":
                value = value.value
            if isinstance(value, ir.Var) and value.name in handles:
                if label not in outer:
                    raise _error(
                        "pending handles cannot enter runtime control flow"
                    )
                handles[inst.target.name] = handles[value.name]
                aliases.add(inst)
                changed = True
    methods = {}
    method_assigns = set()
    for label, inst in statements:
        if not isinstance(inst, ir.Assign):
            continue
        value = inst.value
        if (
            isinstance(value, ir.Expr)
            and value.op == "getattr"
            and value.value.name in handles
            and value.attr == "wait"
        ):
            if label not in outer:
                raise _error(
                    "issue and wait must be in the kernel outer region"
                )
            methods[inst.target.name] = handles[value.value.name]
            method_assigns.add(inst)
    waits = {issue: [] for issue in issues}
    wait_assigns = set()
    for label, inst in statements:
        if not isinstance(inst, ir.Assign):
            continue
        call = inst.value
        if (
            isinstance(call, ir.Expr)
            and call.op == "call"
            and call.func.name in methods
        ):
            if label not in outer:
                raise _error(
                    "issue and wait must be in the kernel outer region"
                )
            if call.args or call.kws or call.vararg is not None:
                raise _error("pending.wait() accepts no arguments")
            waits[methods[call.func.name]].append(inst)
            wait_assigns.add(inst)
    for _, inst in statements:
        used = {value.name for value in inst.list_vars()}
        if isinstance(inst, ir.Assign):
            used.discard(inst.target.name)
        if used & handles.keys() and inst not in aliases | method_assigns:
            raise _error("pending handles may only be aliased or waited on")
        if used & methods.keys() and inst not in wait_assigns:
            raise _error("pending.wait methods cannot escape")
    for issue, matches in waits.items():
        if len(matches) != 1:
            raise _error("every pending copy requires exactly one wait")
        _lower_pair(planner, issue, matches[0])
    for inst in aliases | method_assigns:
        planner.replacements[inst] = [
            ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
        ]
