# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type cooperative calls with Numba typing templates.

Each kernel-callable object of both public APIs, ``cuda.coop`` and
``cuda.coop.numba_mlir``, gets one typing template. Objects the typed
pipeline supports are typed for real; every other one gets a template that
rejects the call with ``unsupported_primitive``, so a kernel never falls
back to the planner and never fails with Numba's generic "untyped global"
message.

``load`` and ``store`` are reached through inline wrappers. Numba typing
cannot refine a call argument from a template, and a payload's
dtype often comes from the ``load`` that fills it. Each public ``load`` and
``store`` object therefore gets a wrapper, generated at activation from the
object's own signature and attached to it, which Numba inlines before
typing::

    def _coop_load_wrapper(group, source, output, /, algorithm='direct', ...):
        if _never():
            output._coop_refine_load(source)
        _coop_load_op(group, source, output, algorithm=algorithm, ...)

The guarded method call is a witness: its template returns a refined
receiver, which Numba propagates to the caller's payload with the evidence
strength of a producer (or, for ``store``, of a store destination). The
private operation then types with a precise payload and builds coop's own
request, a dims-free ``GroupPrimitiveCall`` that the storage pass completes
with the launch's block.
"""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from ._errors import TypedPipelineError, unsupported_primitive
from ._types import (
    EXPLICIT,
    PRODUCER,
    STORE_DST,
    CoopGroupType,
    ThreadDataType,
)

# Public names that are never called inside a kernel: host-side descriptors
# and registration, typing protocols, and Numba-CUDA-MLIR's own namespaces
# re-exported for convenience.
_NOT_KERNEL_CALLABLE = frozenset(
    {
        "Hierarchy",
        "StatefulFunction",
        "TempStorageLike",
        "ThreadDataLike",
        "ThreadGroup",
        "ThreadHierarchy",
        "local",
        "register",
        "shared",
    }
)

# Arguments the typed pipeline does not take yet, by operation.
_REFUSED_CONTROLS = {
    "load": ("valid_items", "oob_default", "offset"),
    "store": ("valid_items", "offset"),
}

_PAYLOAD = {"load": "output", "store": "value"}
_MEMORY = {"load": "source", "store": "destination"}


def kernel_primitives() -> dict[object, str]:
    """Map each kernel-callable public object to its qualified name."""

    from ... import numba_mlir as qualified
    from ..._core import api as common

    primitives: dict[object, str] = {}
    for prefix, module in (
        ("cuda.coop", common),
        ("cuda.coop.numba_mlir", qualified),
    ):
        for name in module.__all__:
            if name in _NOT_KERNEL_CALLABLE:
                continue
            obj = getattr(module, name)
            if callable(obj):
                primitives.setdefault(obj, f"{prefix}.{name}")
    return primitives


def group_operation(obj: object) -> str | None:
    """Return coop's registered group operation for a public object."""

    from ..._core.api._dispatch import _common_group_operation_name
    from .._compiler._operations import group_operation_name

    return _common_group_operation_name(obj) or group_operation_name(obj)


def register_template(
    key: object,
    generic: Callable,
    name: str,
    *,
    prefer_literal: bool = False,
) -> type:
    """Type calls to ``key`` with ``generic(template, args, kws)``."""

    template = type(
        f"CoopTyped_{name}",
        (nb.AbstractTemplate,),
        {"key": key, "generic": generic, "prefer_literal": prefer_literal},
    )
    nb.typing_registry.register(template)
    nb.typing_registry.register_global(key, nb.types.Function(template))
    return template


def _rejecting(qualified_name: str) -> Callable:
    def generic(self, args, kws):
        raise unsupported_primitive(qualified_name)

    return generic


def _not_inlined(qualified_name: str) -> Callable:
    def generic(self, args, kws):
        raise TypedPipelineError(
            f"{qualified_name} reached typing without its inline wrapper. "
            "Call it directly by name inside the kernel."
        )

    return generic


def _omitted(ty: Any) -> bool:
    return ty is None or isinstance(ty, (nb.types.NoneType, nb.types.Omitted))


def _payload_dtype(ty: Any) -> Any:
    if isinstance(ty, nb.types.NumberClass):
        return ty.instance_type
    if isinstance(ty, nb.types.DType):
        return ty.dtype
    return None


# this_block ----------------------------------------------------------------


def _this_block(function: Callable, qualified_name: str) -> Callable:
    def generic(self, args, kws):
        if args or kws:
            raise TypedPipelineError(f"{qualified_name}() takes no arguments")
        return nb.signature(CoopGroupType(function()))

    return generic


# ThreadData ----------------------------------------------------------------

_CONSTRUCTOR_PARAMETERS = ("items_per_thread", "dtype", "alignment")


def _bind_constructor(qualified_name: str, args, kws) -> dict:
    if len(args) > 2:
        raise TypedPipelineError(
            f"{qualified_name}: alignment is a keyword-only argument"
        )
    bound = dict(zip(_CONSTRUCTOR_PARAMETERS, args))
    for name, ty in kws.items():
        if name not in _CONSTRUCTOR_PARAMETERS or name in bound:
            raise TypedPipelineError(
                f"{qualified_name}: unexpected argument {name!r}"
            )
        bound[name] = ty
    if "items_per_thread" not in bound:
        raise TypedPipelineError(
            f"{qualified_name}: items_per_thread is required"
        )
    return bound


def _fold_constructor_arguments(args, kws) -> tuple:
    bound = dict(zip(_CONSTRUCTOR_PARAMETERS, args))
    bound.update(dict(kws))
    return tuple(bound[n] for n in _CONSTRUCTOR_PARAMETERS if n in bound)


def constructor_alignment(qualified_name: str, ty: Any) -> int | None:
    """Return the normalized ``alignment=`` of a constructor call."""

    from ..._core.api._payload import _normalize_alignment

    if _omitted(ty):
        return None
    if not isinstance(ty, nb.types.IntegerLiteral):
        raise TypedPipelineError(
            f"{qualified_name}: alignment must be a compile-time integer"
        )
    try:
        return _normalize_alignment(ty.literal_value)
    except (TypeError, ValueError) as exc:
        raise TypedPipelineError(f"{qualified_name}: {exc}") from None


def _thread_data(qualified_name: str) -> Callable:
    def generic(self, args, kws):
        bound = _bind_constructor(qualified_name, args, kws)
        items = bound["items_per_thread"]
        if isinstance(items, nb.types.Integer) and not isinstance(
            items, nb.types.IntegerLiteral
        ):
            raise nb.force_literal_arg({0}, _fold_constructor_arguments)
        if (
            not isinstance(items, nb.types.IntegerLiteral)
            or items.literal_value <= 0
        ):
            raise TypedPipelineError(
                f"{qualified_name}: items_per_thread must be a positive "
                f"compile-time integer, got {items}"
            )
        alignment = constructor_alignment(
            qualified_name, bound.get("alignment")
        )
        dtype_ty = bound.get("dtype")
        if _omitted(dtype_ty):
            payload = ThreadDataType(
                nb.undefined, items.literal_value, alignment=alignment
            )
        else:
            dtype = _payload_dtype(dtype_ty)
            if dtype is None:
                raise TypedPipelineError(
                    f"{qualified_name}: dtype must be a Numba or NumPy "
                    f"scalar type, got {dtype_ty}"
                )
            payload = ThreadDataType(
                dtype, items.literal_value, EXPLICIT, alignment
            )
        return nb.signature(payload, *args, *kws.values())

    return generic


# The method witness ----------------------------------------------------------


def _never() -> bool:
    """A condition that is false at run time and unknown to typing."""

    raise NotImplementedError("cuda.coop: kernel only")


def _witness(td, args, strength):
    if (
        len(args) == 1
        and isinstance(td, ThreadDataType)
        and isinstance(args[0], nb.types.Array)
        and args[0].dtype.is_precise()
    ):
        return nb.signature(
            nb.types.none, *args, recvr=td.refined(args[0].dtype, strength)
        )
    # Anything else leaves the receiver alone; the operation's template
    # then reports what is wrong with the call.
    return nb.signature(nb.types.none, *args, recvr=td)


def _witness_template(key: type) -> type:
    class Witness(nb.AttributeTemplate):
        @nb.bound_function("coop.thread_data.refine_load")
        def resolve__coop_refine_load(self, td, args, kws):
            return _witness(td, args, PRODUCER)

        @nb.bound_function("coop.thread_data.refine_store")
        def resolve__coop_refine_store(self, td, args, kws):
            return _witness(td, args, STORE_DST)

    Witness.key = key
    Witness.__name__ = f"CoopWitness_{key.__name__}"
    return Witness


# load and store --------------------------------------------------------------


def _coop_load_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _coop_store_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


OPERATIONS: dict[Callable, str] = {
    _coop_load_op: "load",
    _coop_store_op: "store",
}


@dataclass(frozen=True, eq=False)
class LoadStoreRequest:
    """A typed ``load`` or ``store``, before the launch is known.

    Requests are interned by ``key``, so identity is equality.
    ``call`` is coop's ``GroupPrimitiveCall``; the storage pass completes
    it with ``plan_group_primitive`` once it knows the block.
    """

    kind: str
    dtype: Any
    items_per_thread: int
    algorithm: str
    call: Any

    @property
    def key(self) -> tuple:
        return (self.kind, self.dtype, self.items_per_thread, self.algorithm)


_REQUESTS: dict[tuple, LoadStoreRequest] = {}


def _make_request(kind, group, dtype, items, algorithm) -> LoadStoreRequest:
    from ..._core import ArgumentBinding
    from ..._core.group._model import GroupPrimitiveCall
    from ..._core.group.load_store import (
        GroupLoadStoreAlgorithm,
        GroupLoadStoreKind,
        GroupLoadStoreSemantics,
    )
    from .._lowering._core import NumbaMlirCoreAdapter

    semantics = GroupLoadStoreSemantics(
        kind=GroupLoadStoreKind(kind),
        dtype=NumbaMlirCoreAdapter().core_dtype(dtype),
        items_per_thread=items,
        algorithm=GroupLoadStoreAlgorithm(algorithm),
        valid_items=ArgumentBinding.omitted(),
        oob_default=ArgumentBinding.omitted(),
        offset=ArgumentBinding.omitted(),
    )
    call = GroupPrimitiveCall(group=group.group, operation=semantics)
    return LoadStoreRequest(kind, dtype, items, algorithm, call)


def _algorithm(kind: str, ty: Any) -> str | None:
    from ..._core.group.load_store import GroupLoadStoreAlgorithm

    if _omitted(ty):
        return "direct"
    if not isinstance(ty, nb.types.StringLiteral):
        if isinstance(ty, nb.types.UnicodeType):
            return None
        raise TypedPipelineError(
            f"coop.{kind}: algorithm must be a string literal, got {ty}"
        )
    try:
        return GroupLoadStoreAlgorithm(ty.literal_value).value
    except ValueError:
        names = ", ".join(repr(a.value) for a in GroupLoadStoreAlgorithm)
        raise TypedPipelineError(
            f"coop.{kind}: unknown algorithm {ty.literal_value!r}; "
            f"use one of {names}"
        ) from None


def _check_payload(kind: str, payload: Any) -> None:
    name = _PAYLOAD[kind]
    if isinstance(payload, ThreadDataType):
        return
    if isinstance(payload, nb.types.Array):
        raise TypedPipelineError(
            f"coop.{kind}: the {name} payload {payload} has no typed "
            "extent. Use coop.ThreadData: a plain array's shape is not "
            "part of its type."
        )
    if kind == "store" and isinstance(payload, nb.types.Number):
        raise TypedPipelineError(
            f"coop.store: a scalar value ({payload}) is not supported on "
            "the typed pipeline yet; store a coop.ThreadData."
        )
    raise TypedPipelineError(
        f"coop.{kind}: {name} must be a coop.ThreadData, got {payload}"
    )


def _bind_operation(kind: str, args, kws) -> dict:
    bound = dict(zip(("group", _MEMORY[kind], _PAYLOAD[kind]), args))
    bound.update(kws)
    return bound


def _request(kind: str, args, kws) -> LoadStoreRequest | None:
    """Validate a typed call and return its interned request.

    Return ``None`` while the payload's dtype is still undefined, so that a
    later typing round retries with the refined payload.
    """

    from ..._core.dtype_policy import validate_common_numeric_dtype_name

    bound = _bind_operation(kind, args, kws)
    if not _omitted(bound.get("temp_storage")):
        raise TypedPipelineError(
            f"coop.{kind}: explicit temp_storage= is not supported on the "
            "typed pipeline; omit it and the pipeline provides the scratch."
        )
    for name in _REFUSED_CONTROLS[kind]:
        if not _omitted(bound.get(name)):
            raise TypedPipelineError(
                f"coop.{kind}: {name}= is not supported on the typed "
                "pipeline yet; omit it."
            )
    group = bound.get("group")
    if not isinstance(group, CoopGroupType) or not group.is_block:
        raise TypedPipelineError(
            f"coop.{kind}: the group must be coop.this_block(); {group} is "
            "not supported on the typed pipeline."
        )
    memory = bound.get(_MEMORY[kind])
    if not isinstance(memory, nb.types.Array):
        raise TypedPipelineError(
            f"coop.{kind}: {_MEMORY[kind]} must be an array, got {memory}"
        )
    payload = bound.get(_PAYLOAD[kind])
    _check_payload(kind, payload)
    if not payload.dtype.is_precise():
        return None
    if payload.dtype != memory.dtype:
        raise TypedPipelineError(
            f"coop.{kind}: payload dtype {payload.dtype} conflicts with "
            f"{_MEMORY[kind]} dtype {memory.dtype}"
        )
    algorithm = _algorithm(kind, bound.get("algorithm"))
    if algorithm is None:
        return None
    try:
        validate_common_numeric_dtype_name(str(payload.dtype), operation=kind)
    except TypeError as exc:
        raise TypedPipelineError(str(exc)) from None
    key = (kind, payload.dtype, payload.items, algorithm)
    request = _REQUESTS.get(key)
    if request is None:
        request = _REQUESTS[key] = _make_request(
            kind, group, payload.dtype, payload.items, algorithm
        )
    return request


def request_for(kind: str, args, kws) -> LoadStoreRequest:
    """Return the request a typed call built, from its argument types."""

    request = _request(kind, args, kws)
    if request is None:
        raise AssertionError(f"coop.{kind} reached the storage pass untyped")
    return request


def _operation(kind: str) -> Callable:
    def generic(self, args, kws):
        if _request(kind, args, kws) is None:
            return None
        return nb.signature(nb.types.none, *args, *kws.values())

    return generic


# Inline wrappers --------------------------------------------------------------

WRAPPER_SOURCE: dict[str, str] = {}
_WRAPPER_FILENAME = "<cuda.coop typed-pipeline wrapper for {}>"


def is_wrapper_location(loc: Any) -> bool:
    """Whether ``loc`` is inside an inlined wrapper, not the kernel."""

    prefix = _WRAPPER_FILENAME.split("{", 1)[0]
    return str(getattr(loc, "filename", "")).startswith(prefix)


def _parameter_text(p: inspect.Parameter) -> str:
    if p.default is p.empty:
        return p.name
    if eval(repr(p.default), {}) != p.default:
        raise TypeError(f"default of {p.name!r} has no literal spelling")
    return f"{p.name}={p.default!r}"


def make_wrapper(
    public: Callable, qualified_name: str, operation: str
) -> Callable:
    """Generate the inline wrapper of a public ``load`` or ``store`` object.

    The wrapper has the public object's parameters, with keyword-only ones
    relaxed to ordinary parameters: Numba's inliner cannot bind keyword-only
    parameters. The private operation still receives them by
    keyword.
    """

    params = list(inspect.signature(public).parameters.values())
    positional_only = [p for p in params if p.kind is p.POSITIONAL_ONLY]
    ordinary = [
        p for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    header = [_parameter_text(p) for p in positional_only]
    if positional_only:
        header.append("/")
    header += [_parameter_text(p) for p in ordinary]
    call = [p.name for p in positional_only]
    call += [
        p.name if p.kind is p.POSITIONAL_OR_KEYWORD else f"{p.name}={p.name}"
        for p in ordinary
    ]
    op = _coop_load_op if operation == "load" else _coop_store_op
    name = f"_coop_{operation}_wrapper"
    source = (
        f"def {name}({', '.join(header)}):\n"
        "    if _never():\n"
        f"        {_PAYLOAD[operation]}._coop_refine_{operation}"
        f"({_MEMORY[operation]})\n"
        f"    {op.__name__}({', '.join(call)})\n"
    )
    filename = _WRAPPER_FILENAME.format(qualified_name)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {"_never": _never, op.__name__: op}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    WRAPPER_SOURCE[qualified_name] = source
    return namespace[name]


# Registration ----------------------------------------------------------------


def _lower_never(builder, target, args, kwargs):
    builder.store_var(
        target, nb.arith.constant(nb.mlir_ir.IntegerType.get_signless(1), 0)
    )


def _lower_witness(builder, target, args, kwargs):
    pass


_registered = False


def register() -> None:
    """Register a template for every kernel-callable public object.

    ``this_block``, ``ThreadData``, ``load`` and ``store`` of both APIs are
    typed; every other primitive is rejected.
    """

    global _registered
    if _registered:
        return
    for obj, qualified_name in kernel_primitives().items():
        short = qualified_name.rsplit(".", 1)[1]
        name = qualified_name.replace(".", "_")
        operation = group_operation(obj)
        if short == "this_block":
            register_template(obj, _this_block(obj, qualified_name), name)
        elif short == "ThreadData":
            register_template(
                obj, _thread_data(qualified_name), name, prefer_literal=True
            )
        elif operation in OPERATIONS.values():
            nb.attach_inline_wrapper(
                obj, make_wrapper(obj, qualified_name, operation)
            )
            register_template(obj, _not_inlined(qualified_name), name)
        else:
            register_template(obj, _rejecting(qualified_name), name)

    for op, kind in OPERATIONS.items():
        register_template(
            op, _operation(kind), f"private_{kind}", prefer_literal=True
        )
    for key in (ThreadDataType, nb.types.Array, nb.types.Number):
        nb.typing_registry.register_attr(_witness_template(key))
    for method in ("refine_load", "refine_store"):
        nb.lowering_registry.lower(
            f"coop.thread_data.{method}", nb.types.VarArg(nb.types.Any)
        )(_lower_witness)

    nb.extending.type_callable(_never)(lambda context: lambda: nb.types.boolean)
    nb.lowering_registry.lower(_never)(_lower_never)
    _registered = True


__all__ = [
    "OPERATIONS",
    "WRAPPER_SOURCE",
    "LoadStoreRequest",
    "constructor_alignment",
    "group_operation",
    "is_wrapper_location",
    "kernel_primitives",
    "make_wrapper",
    "register",
    "register_template",
    "request_for",
]
