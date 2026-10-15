# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Find a kernel's cooperative call sites and check them, from typed IR.

Site analysis is the first stage of ``CoopCallResolution``. It reads the
typed function and nothing else: not the launch, not NVRTC. It does not
change the IR. For each kernel it:

* collects the cooperative call sites in program order;
* refuses cooperative calls inside functions compiled as separate device
  functions (``targetoptions["device"]``), which have no launch;
* checks that write-only payloads agree on their dtype, and that every read
  of a payload typed to its final dtype;
* builds each site's dims-free request;
* measures the kernel's own static shared memory.

Every error a user can fix in the kernel's source is raised here, before
anything is specialized or compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from ._errors import TypedPipelineError
from ._templates import (
    OPERATIONS,
    LoadStoreRequest,
    is_wrapper_location,
    request_for,
)
from ._types import WRITE, ThreadDataType


@dataclass(frozen=True, eq=False)
class Site:
    """A call site; ``line`` is the kernel source line of the public call.

    ``stmt`` is the call's assignment; its target names the site.
    """

    stmt: Any
    kind: str
    line: int | None
    request: LoadStoreRequest


@dataclass(frozen=True)
class Sites:
    """A kernel's checked call sites, in program order.

    ``user_shared_bytes`` is the static shared memory the kernel allocates
    itself with ``cuda.shared.array``.
    """

    sites: tuple[Site, ...]
    user_shared_bytes: int


# Checks -----------------------------------------------------------------------


def _check_write_agreement(func_ir: Any, typemap: dict) -> None:
    """Payloads typed only by indexed writes must agree on the dtype."""

    written: dict[str, set] = {}
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, (nb.ir.SetItem, nb.ir.StaticSetItem)):
                ty = typemap.get(stmt.target.name)
                if isinstance(ty, ThreadDataType) and ty.strength == WRITE:
                    written.setdefault(stmt.target.name, set()).add(
                        nb.unliteral(typemap[stmt.value.name])
                    )
    for name, dtypes in written.items():
        if dtypes != {typemap[name].dtype}:
            found = ", ".join(sorted(map(str, dtypes)))
            raise TypedPipelineError(
                f"coop.ThreadData {name!r}: its writes disagree on the dtype "
                f"({found}). Pass dtype= or make the writes agree."
            )


def _check_reads(func_ir: Any, typemap: dict) -> None:
    """Every read of a payload must type to the payload's final dtype.

    A read typed while weaker evidence held can keep a wider type after a
    stronger refinement; that is an error rather than a silent widening.
    """

    for block in func_ir.blocks.values():
        for stmt in block.body:
            if not (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, nb.ir.Expr)
                and stmt.value.op in ("getitem", "static_getitem")
            ):
                continue
            payload = typemap.get(stmt.value.value.name)
            if not isinstance(payload, ThreadDataType):
                continue
            result = typemap.get(stmt.target.name)
            if isinstance(result, nb.types.Array):
                result = result.dtype
            result = nb.unliteral(result)
            if result != payload.dtype:
                raise TypedPipelineError(
                    f"coop.ThreadData {stmt.value.value.name!r}: a read at "
                    f"line {stmt.loc.line} typed as {result}, but the "
                    f"payload's dtype is {payload.dtype}. Pass "
                    f"dtype={payload.dtype} to coop.ThreadData."
                )


def _callee(typemap: dict, expr: Any) -> Any:
    ty = typemap.get(expr.func.name)
    if isinstance(ty, nb.types.Function):
        return ty.typing_key
    return None


def _shape(expr: Any) -> Any:
    return expr.args[0] if expr.args else dict(expr.kws).get("shape")


def _user_static_shared_bytes(func_ir: Any, typemap: dict, calltypes) -> int:
    total = 0
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if not (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, nb.ir.Expr)
                and stmt.value.op == "call"
                and _callee(typemap, stmt.value) is nb.cuda.shared.array
            ):
                continue
            shape = _shape(stmt.value)
            shape = None if shape is None else typemap[shape.name]
            result = calltypes[stmt.value].return_type
            if isinstance(shape, nb.types.IntegerLiteral):
                # Numba types bool without a bitwidth and stores it in a byte.
                bitwidth = getattr(result.dtype, "bitwidth", 8)
                total += shape.literal_value * bitwidth // 8
    return total


# The analysis -----------------------------------------------------------------


class _Analysis:
    """The typed function, and the per-site reads that need its IR."""

    def __init__(self, func_ir, typemap):
        self.func_ir, self.typemap = func_ir, typemap

    def collect(self) -> list[tuple[Any, str, int | None]]:
        found = []
        line = None
        for label in sorted(self.func_ir.blocks):
            for stmt in self.func_ir.blocks[label].body:
                if not is_wrapper_location(stmt.loc):
                    line = stmt.loc.line
                if (
                    isinstance(stmt, nb.ir.Assign)
                    and isinstance(stmt.value, nb.ir.Expr)
                    and stmt.value.op == "call"
                ):
                    kind = OPERATIONS.get(_callee(self.typemap, stmt.value))
                    if kind is not None:
                        found.append((stmt, kind, line))
        return found

    def site(self, stmt, kind, line) -> Site:
        expr = stmt.value
        request = request_for(
            kind,
            [self.typemap[a.name] for a in expr.args],
            {k: self.typemap[v.name] for k, v in expr.kws},
        )
        return Site(stmt, kind, line, request)


def analyze(
    func_ir: Any,
    typemap: dict,
    calltypes: Any,
    *,
    targetoptions: dict,
    func_id: Any,
) -> Sites | None:
    """Return the kernel's checked sites, or ``None`` for a device function.

    A device function without cooperative calls has nothing to resolve; one
    with them is an error.
    """

    analysis = _Analysis(func_ir, typemap)
    found = analysis.collect()
    if targetoptions.get("device"):
        if found:
            _, kind, line = found[0]
            raise TypedPipelineError(
                f"coop.{kind} at line {line} "
                f"is inside {func_id.func_qualname!r}, a separately "
                "compiled device function, which has no launch to plan "
                "for. Let the function inline (the cuda.jit default) "
                "instead of inline='never'."
            )
        return None
    _check_write_agreement(func_ir, typemap)
    _check_reads(func_ir, typemap)
    sites = tuple(analysis.site(*f) for f in found)
    return Sites(sites, _user_static_shared_bytes(func_ir, typemap, calltypes))


__all__ = ["Site", "Sites", "analyze"]
