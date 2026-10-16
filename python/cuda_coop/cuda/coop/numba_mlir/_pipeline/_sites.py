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
* reads the extent of a ``cuda.coop.numba_mlir`` call's ``cuda.local.array``
  payload from its constructor's shape;
* binds each site's ``valid_items``, ``oob_default`` and ``offset`` as
  static or runtime by where the value comes from,
  and checks them and any scalar ``store`` value by coop's rules;
* builds each site's dims-free request with its controls;
* measures the kernel's own static shared memory.

Every error a user can fix in the kernel's source is raised here, before
anything is specialized or compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from . import _bridges as nb
from ._errors import TypedPipelineError
from ._templates import (
    CONTROLS,
    OPERATIONS,
    QUALIFIED_OPERATIONS,
    LoadStoreRequest,
    is_scalar_value,
    is_wrapper_location,
    request_for,
    with_controls,
)
from ._types import WRITE, ThreadDataType


@dataclass(frozen=True)
class Scalar:
    """A scalar ``store`` value; ``value`` is ``None`` when it is runtime."""

    value: Any


@dataclass(frozen=True, eq=False)
class Site:
    """A call site; ``line`` is the kernel source line of the public call.

    ``stmt`` is the call's assignment; its target names the site.
    """

    stmt: Any
    kind: str
    line: int | None
    request: LoadStoreRequest
    scalar: Scalar | None = None


@dataclass(frozen=True)
class Sites:
    """A kernel's checked call sites, in program order.

    ``user_shared_bytes`` is the static shared memory the kernel allocates
    itself with ``cuda.shared.array``.
    """

    sites: tuple[Site, ...]
    user_shared_bytes: int


_RUNTIME = object()


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

    def __init__(self, func_ir, typemap, args):
        self.func_ir, self.typemap, self.args = func_ir, typemap, tuple(args)

    def collect(self) -> list[tuple[Any, str, int | None, bool]]:
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
                    callee = _callee(self.typemap, stmt.value)
                    kind = OPERATIONS.get(callee)
                    if kind is not None:
                        found.append(
                            (stmt, kind, line, callee in QUALIFIED_OPERATIONS)
                        )
        return found

    def site(self, stmt, kind, line, qualified) -> Site:
        expr = stmt.value
        request = request_for(
            kind,
            [self.typemap[a.name] for a in expr.args],
            {k: self.typemap[v.name] for k, v in expr.kws},
            qualified=qualified,
            extent=self._array_extent(stmt, kind) if qualified else None,
        )
        scalar = self._store_scalar(stmt, kind, request.dtype)
        request = with_controls(
            request, self._controls(stmt, kind, request.dtype)
        )
        return Site(stmt, kind, line, request, scalar)

    def _provenance(self, var: Any) -> Any:
        """Return ``var``'s static provenance, or ``_RUNTIME``.

        A static value comes, through aliases, casts and phis,
        from a constant, a global, a
        free variable or a literal argument.
        """

        from .._compiler._scalar_provenance import (
            try_resolve_static_scalar_provenance,
        )

        definitions = self.func_ir._definitions
        arguments = self.args
        resolved, scalar = try_resolve_static_scalar_provenance(
            var,
            definitions=lambda v: definitions.get(v.name, ()),
            argument_type=lambda index: (
                arguments[index] if 0 <= index < len(arguments) else None
            ),
        )
        return scalar if resolved else _RUNTIME

    def _controls(self, stmt, kind: str, dtype: Any) -> tuple:
        """Bind and validate the site's scalar controls.

        Check static ``valid_items`` and ``offset`` first, then the integer
        types of runtime ones, then ``oob_default`` against the payload dtype.
        """

        from ..._core import ArgumentBinding
        from .._compiler._parameters import (
            _validate_common_numeric_dtype,
            _validate_runtime_integer_dtype,
            coerce_static_scalar,
        )

        keywords = dict(stmt.value.kws)
        found = {}
        for name in CONTROLS[kind]:
            var = keywords.get(name)
            if var is None:
                continue
            scalar = self._provenance(var)
            if scalar is _RUNTIME:
                found[name] = (_RUNTIME, self.typemap[var.name])
            elif scalar.value is not None:
                found[name] = (scalar, None)

        def check(condition: bool, message: str) -> None:
            if condition:
                raise TypedPipelineError(message)

        for name in ("valid_items", "offset"):
            scalar, _ = found.get(name, (None, None))
            if scalar is None or scalar is _RUNTIME:
                continue
            value = scalar.value
            check(
                isinstance(value, bool) or not isinstance(value, Integral),
                f"coop {kind} {name} must be an integer",
            )
            if name == "offset":
                check(
                    int(value) < 0,
                    "coop load/store static offset must be nonnegative",
                )
                check(
                    int(value) > (1 << 63) - 1,
                    "coop load/store static offset must fit a signed 64-bit "
                    "integer",
                )
        for name in ("valid_items", "offset"):
            scalar, ty = found.get(name, (None, None))
            if scalar is _RUNTIME:
                try:
                    _validate_runtime_integer_dtype(
                        ty, operation=kind, parameter=name
                    )
                except TypeError as exc:
                    raise TypedPipelineError(str(exc)) from None

        controls = []
        for name in CONTROLS[kind]:
            if name not in found:
                continue
            scalar, ty = found[name]
            if scalar is _RUNTIME:
                if name == "oob_default":
                    try:
                        ty = _validate_common_numeric_dtype(
                            ty, operation="load", parameter="oob_default"
                        )
                    except (TypeError, ValueError) as exc:
                        raise TypedPipelineError(str(exc)) from None
                    check(
                        ty != dtype,
                        "cuda.coop.numba_mlir.load runtime oob_default dtype "
                        f"{ty} does not match payload dtype {dtype}",
                    )
                controls.append((name, ArgumentBinding.runtime()))
                continue
            value = scalar.value
            if name == "oob_default":
                try:
                    value = coerce_static_scalar(
                        value,
                        dtype,
                        operation="load",
                        parameter="oob_default",
                        source_dtype=scalar.dtype,
                    )
                except (TypeError, ValueError) as exc:
                    raise TypedPipelineError(str(exc)) from None
            controls.append((name, ArgumentBinding.static(value)))
        return tuple(controls)

    def _store_scalar(self, stmt, kind: str, dtype: Any) -> Scalar | None:
        """Check a scalar ``store`` value against its destination.

        A static value converts to the destination's dtype under coop's
        scalar rules; a runtime value must already have that dtype exactly.
        """

        from .._compiler._parameters import coerce_static_scalar

        if kind != "store":
            return None
        var = stmt.value.args[2]
        ty = self.typemap[var.name]
        if not is_scalar_value(ty):
            return None
        scalar = self._provenance(var)
        if scalar is _RUNTIME or scalar.value is None:
            actual = getattr(ty, "literal_type", ty)
            if actual != dtype:
                raise TypedPipelineError(
                    f"cuda.coop.numba_mlir.store value dtype {actual} does "
                    f"not match destination dtype {dtype}"
                )
            return Scalar(None)
        try:
            value = coerce_static_scalar(
                scalar.value,
                dtype,
                operation="store",
                parameter="value",
                source_dtype=scalar.dtype,
            )
        except (TypeError, ValueError) as exc:
            raise TypedPipelineError(str(exc)) from None
        return Scalar(value)

    def _array_extent(self, stmt, kind: str) -> int | None:
        """Return the extent of a plain-array payload, read from the IR.

        The payload must come, through aliases, from
        ``cuda.local.array`` calls with a constant integer shape, and they
        must agree. Return ``None`` for a payload that is not a plain array.
        """

        var = stmt.value.args[2]
        ty = self.typemap[var.name]
        if not isinstance(ty, nb.types.Array) or isinstance(ty, ThreadDataType):
            return None
        extents, pending, seen = set(), [var], set()
        while pending:
            var = pending.pop()
            if var.name in seen:
                continue
            seen.add(var.name)
            for definition in self.func_ir._definitions.get(var.name, ()):
                if isinstance(definition, nb.ir.Var):
                    pending.append(definition)
                elif isinstance(definition, nb.ir.Expr) and (
                    definition.op == "cast"
                ):
                    pending.append(definition.value)
                elif isinstance(definition, nb.ir.Expr) and (
                    definition.op == "phi"
                ):
                    pending.extend(definition.incoming_values)
                else:
                    extents.add(self._local_array_extent(definition))
        if None in extents or not extents:
            raise TypedPipelineError(
                f"cuda.coop.numba_mlir.{kind} requires a static "
                "items_per_thread extent before provider selection"
            )
        if len(extents) > 1:
            raise TypedPipelineError(
                "cuda.coop.numba_mlir loop-carried payloads have "
                "inconsistent items_per_thread extents"
            )
        return extents.pop()

    def _local_array_extent(self, definition: Any) -> int | None:
        if not (
            isinstance(definition, nb.ir.Expr)
            and definition.op == "call"
            and _callee(self.typemap, definition) is nb.cuda.local.array
        ):
            return None
        shape = _shape(definition)
        ty = None if shape is None else self.typemap.get(shape.name)
        if isinstance(ty, nb.types.IntegerLiteral):
            return ty.literal_value
        return None


def analyze(
    func_ir: Any,
    typemap: dict,
    calltypes: Any,
    *,
    args: tuple,
    targetoptions: dict,
    func_id: Any,
) -> Sites | None:
    """Return the kernel's checked sites, or ``None`` for a device function.

    A device function without cooperative calls has nothing to resolve; one
    with them is an error.
    """

    analysis = _Analysis(func_ir, typemap, args)
    found = analysis.collect()
    if targetoptions.get("device"):
        if found:
            _, kind, line, _ = found[0]
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


__all__ = ["Scalar", "Site", "Sites", "analyze"]
