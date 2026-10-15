# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Put the resolved storage into the IR and tell lowering what to call.

Storage materialization is the last stage of ``CoopCallResolution``, and the
only one that changes the IR. It:

* selects each site's provider method and pairs its operands with where
  lowering finds their values, as a ``CallEntry``;
* prepends the scratch backing, a ``cuda.shared.array``, and the region's
  slice of it to the kernel's entry block, with their types;
* passes the region to every call that uses scratch, as the
  ``SCRATCH_KEYWORD`` keyword argument.

It takes no decisions and cannot fail on a user's kernel: every call entry
is built before the IR is touched.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

from . import _bridges as nb
from ._link import Bundle
from ._lower import SCRATCH_KEYWORD
from ._plan import KernelPlan
from ._sites import Site
from ._specialize import Specialization
from ._storage import StorageLayout


@dataclass(frozen=True)
class CallEntry:
    """What lowering emits for one call site.

    ``operands`` follows ``abi_arguments``, one entry per provider operand:

    ``("scratch",)``
        the scratch region's pointer;
    ``("pointer", index)``
        the data pointer of the call's argument ``index``;
    ``("box", index, dtype, value)``
        a one-item array of ``dtype`` holding the scalar argument ``index``,
        or the static ``value`` when it is not ``None``;
    ``("value", keyword)``
        the call's keyword argument, converted to the ABI type.
    """

    symbol: str
    abi_arguments: tuple
    operands: tuple
    barrier_after: bool
    link_items: tuple


@dataclass
class CoopPlan:
    """``state.metadata["coop_plan"]``: the plan and its lowering data.

    ``calls`` is keyed by the name of each call's target variable.
    """

    kernel: KernelPlan
    calls: dict[str, CallEntry] = field(default_factory=dict)


# Methods and operands ---------------------------------------------------------


def _ignored_parameter(parameter: Any) -> bool:
    from .. import _types as coop_types

    if isinstance(
        parameter,
        (
            coop_types.CxxFunction,
            coop_types.StatelessOperator,
            coop_types.StatefulOperator,
        ),
    ):
        return True
    return (
        isinstance(parameter, coop_types.PointerOffset)
        and parameter.static_value is not None
    )


def _inputs(method: Any) -> list:
    return [p for p in method if not _ignored_parameter(p) and not p.is_output]


def _select_method(algorithm: Any, operands: int) -> Any:
    for method in algorithm.parameters:
        if len(_inputs(method)) == operands:
            return method
    raise AssertionError(f"{algorithm.c_name} has no {operands}-operand method")


def _operands(method: Any, site: Site, algorithm: Any, scratch: bool) -> tuple:
    """Pair each provider operand with where lowering finds its value.

    Coop's providers take the scratch pointer, the memory pointer, the
    payload, then the runtime controls in ``CONTROLS`` order.
    """

    from .. import _types as coop_types

    pointer_kinds = (
        coop_types.Pointer,
        coop_types.Array,
        coop_types.PointerReference,
    )
    payload = (
        ("pointer", 2)
        if site.scalar is None
        else ("box", 2, site.request.dtype, site.scalar.value)
    )
    pointers = [("scratch",)] * scratch + [("pointer", 1), payload]
    runtime = [
        ("value", name)
        for name, binding in site.request.controls
        if binding.kind.value == "runtime"
    ]
    abi_arguments, operands = [], []
    for parameter in _inputs(method):
        if isinstance(parameter, pointer_kinds):
            abi_arguments.append(nb.types.CPointer(nb.types.none))
            operands.append(pointers.pop(0))
        else:
            abi_arguments.append(parameter.dtype())
            operands.append(runtime.pop(0))
    if pointers or runtime:
        raise AssertionError(f"{algorithm.c_name}: operands left over")
    return tuple(abi_arguments), tuple(operands)


def _call_entry(
    site: Site, algorithm: Any, scratch: bool, barrier: bool, bundle: Bundle
) -> CallEntry:
    runtime = sum(b.kind.value == "runtime" for _, b in site.request.controls)
    method = _select_method(algorithm, 2 + scratch + runtime)
    abi_arguments, operands = _operands(method, site, algorithm, scratch)
    return CallEntry(
        symbol=algorithm.mangled_name(method) + "__abi",
        abi_arguments=abi_arguments,
        operands=operands,
        barrier_after=barrier,
        link_items=(bundle.link_item,),
    )


# Rewriting --------------------------------------------------------------------


def _new_var(typemap, scope, name, ty, loc):
    var = scope.redefine(name, loc)
    typemap[var.name] = ty
    return var


def _emit_backing(func_ir, typemap, calltypes, typingctx, storage):
    """Prepend the backing and the region's slice to the entry block."""

    size, alignment = storage.backing.size, storage.backing.alignment
    entry = func_ir.blocks[min(func_ir.blocks)]
    scope, loc = entry.scope, entry.loc
    types = nb.types

    def new_var(name, ty):
        return _new_var(typemap, scope, name, ty, loc)

    array_fn_ty = typingctx.resolve_getattr(
        types.Module(nb.cuda.shared), "array"
    )
    dtype_ty = typingctx.resolve_value_type(types.uint8)
    size_ty = types.IntegerLiteral(size)
    align_ty = types.IntegerLiteral(alignment)
    array_fn = new_var("$coop_shared_array", array_fn_ty)
    size_var = new_var("$coop_backing_size", size_ty)
    dtype_var = new_var("$coop_backing_dtype", dtype_ty)
    align_var = new_var("$coop_backing_align", align_ty)
    call = nb.ir.Expr.call(
        array_fn, [size_var, dtype_var], (("alignment", align_var),), loc
    )
    array_sig = typingctx.resolve_function_type(
        array_fn_ty, (size_ty, dtype_ty), {"alignment": align_ty}
    )
    calltypes[call] = array_sig
    backing = new_var("$coop_backing", array_sig.return_type)

    slice_fn_ty = typingctx.resolve_value_type(slice)
    slice_sig = typingctx.resolve_function_type(
        slice_fn_ty, (types.intp, types.intp), {}
    )
    slice_fn = new_var("$coop_slice", slice_fn_ty)
    lo = new_var("$coop_region_lo", types.intp)
    hi = new_var("$coop_region_hi", types.intp)
    bounds = new_var("$coop_region_bounds", slice_sig.return_type)
    slice_call = nb.ir.Expr.call(slice_fn, [lo, hi], (), loc)
    calltypes[slice_call] = slice_sig
    getitem = nb.ir.Expr.getitem(backing, bounds, loc)
    getitem_sig = typingctx.resolve_function_type(
        operator.getitem, (array_sig.return_type, slice_sig.return_type), {}
    )
    calltypes[getitem] = getitem_sig
    region = new_var("$coop_region_implicit", getitem_sig.return_type)

    entry.body[:0] = [
        nb.ir.Assign(
            nb.ir.Global("shared_array", nb.cuda.shared.array, loc),
            array_fn,
            loc,
        ),
        nb.ir.Assign(nb.ir.Const(size, loc), size_var, loc),
        nb.ir.Assign(nb.ir.Global("uint8", types.uint8, loc), dtype_var, loc),
        nb.ir.Assign(nb.ir.Const(alignment, loc), align_var, loc),
        nb.ir.Assign(call, backing, loc),
        nb.ir.Assign(nb.ir.Global("slice", slice, loc), slice_fn, loc),
        nb.ir.Assign(nb.ir.Const(0, loc), lo, loc),
        nb.ir.Assign(nb.ir.Const(storage.region.size, loc), hi, loc),
        nb.ir.Assign(slice_call, bounds, loc),
        nb.ir.Assign(getitem, region, loc),
    ]
    func_ir._definitions = nb.build_definitions(func_ir.blocks)
    return region


def materialize(
    func_ir: Any,
    typemap: dict,
    calltypes: Any,
    typingctx: Any,
    sites: tuple[Site, ...],
    specialization: Specialization,
    storage: StorageLayout,
    bundle: Bundle,
) -> dict[str, CallEntry]:
    """Rewrite the IR for the resolved storage; return each call's entry."""

    calls = {
        site.stmt.target.name: _call_entry(
            site, algorithm, scratch, barrier, bundle
        )
        for site, algorithm, scratch, barrier in zip(
            sites,
            specialization.site_algorithms,
            storage.scratch,
            specialization.barrier_after,
            strict=True,
        )
    }
    if storage.region is None:
        return calls
    region = _emit_backing(func_ir, typemap, calltypes, typingctx, storage)
    for site, scratch in zip(sites, storage.scratch, strict=True):
        if scratch:
            site.stmt.value.kws = (
                *site.stmt.value.kws,
                (SCRATCH_KEYWORD, region),
            )
    return calls


__all__ = ["CallEntry", "CoopPlan", "materialize"]
