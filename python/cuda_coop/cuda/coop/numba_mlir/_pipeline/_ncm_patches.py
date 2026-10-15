# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Patch Numba-CUDA-MLIR where the typed pipeline needs a compiler change.

Each patch replaces attributes of Numba-CUDA-MLIR classes with versions that
behave as an upstream ask would make them, and is listed with that ask in the
``_bridges`` table. The typed pipeline's ``_activate`` applies them once,
before it registers anything; the default pipeline never imports this
module, so it leaves the compiler untouched.

A patch is pinned to what it was written against: Numba-CUDA-MLIR
``NCM_VERSION`` and a digest of the source of every attribute it replaces or
relies on. ``apply`` checks every pin before it changes anything, and raises
``TypedPipelineUnavailableError``, naming the patch and its ask, on the first
mismatch.

A patch leaves kernels without cooperative types as they were in one of two
ways, which its docstring states:

* error to success: it acts only where the compiler would have refused the
  kernel;
* cooperative types only: it acts only on values whose type is, or is a
  tuple holding, a ``ThreadDataType``.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numba_cuda_mlir

from . import _bridges as nb

NCM_VERSION = "0.5.4"

try:
    from numba_cuda_mlir.typing import cuda as cuda_typing
except ImportError as exc:
    raise nb.TypedPipelineUnavailableError(
        "The cuda.coop typed pipeline patches Numba-CUDA-MLIR "
        f"{NCM_VERSION}; the installed compiler is missing {exc.name or exc}."
    ) from exc


@dataclass(frozen=True)
class Pin:
    """An attribute a patch replaces or relies on, as it was written against.

    ``digest`` is the start of the SHA-256 of the attribute's source, or
    ``None`` when the class must not define the attribute itself.
    """

    owner: type
    name: str
    digest: str | None


@dataclass(frozen=True)
class Patch:
    """A compiler change, the ask that retires it, and its pins."""

    name: str
    ask: str
    pins: tuple[Pin, ...]
    build: Callable[[], tuple[tuple[type, str, Any], ...]]


def source_digest(obj: Any) -> str | None:
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _pinned(pin: Pin) -> bool:
    if pin.digest is None:
        return pin.name not in vars(pin.owner)
    return source_digest(vars(pin.owner).get(pin.name)) == pin.digest


def check(patch: Patch) -> None:
    """Raise unless ``patch`` matches the installed compiler."""

    installed = numba_cuda_mlir.__version__
    if installed != NCM_VERSION:
        changed = f"the installed compiler is {installed}"
    else:
        stale = [p for p in patch.pins if not _pinned(p)]
        if not stale:
            return
        changed = f"{stale[0].owner.__qualname__}.{stale[0].name} has changed"
    raise nb.TypedPipelineUnavailableError(
        f"The cuda.coop typed pipeline's {patch.name!r} patch was written "
        f"against Numba-CUDA-MLIR {NCM_VERSION}, but {changed}. The patch is "
        f"retired by {patch.ask}; until then, unset "
        "CUDA_COOP_NUMBA_MLIR_PIPELINE to use the default pipeline."
    )


# Argument-sized cuda.local.array (U-P-1) ------------------------------------

_LOCAL_ARRAY_PARAMETERS = ("shape", "dtype", "alignment", "alignas")


def _fold_local_array_arguments(args, kws) -> tuple:
    bound = dict(zip(_LOCAL_ARRAY_PARAMETERS, args))
    bound.update(dict(kws))
    return tuple(bound[n] for n in _LOCAL_ARRAY_PARAMETERS if n in bound)


def _local_array_generic(self) -> Callable:
    """Type ``cuda.local.array`` and request a non-literal shape as a literal.

    Numba-CUDA-MLIR finds no match for a ``cuda.local.array`` whose shape is
    a non-literal integer. When the call would type with a literal shape,
    this raises ``ForceLiteralArg`` for the shape instead, so a shape that
    is a kernel argument is retried as a literal: the kernel compiles once
    per distinct value of that argument. A shape that is not a kernel
    argument is still refused, with Numba's "Cannot request literal type".
    Tuple shapes are unchanged, since the launcher retries only scalar
    arguments as literals.

    Scope: error to success. A call that typed before types the same.
    """

    typer = cuda_typing.GenericArrayTemplate.generic(self)

    def literal_shape_typer(shape, dtype, alignment=None, alignas=None):
        sig = typer(shape, dtype, alignment, alignas)
        if (
            sig is None
            and isinstance(shape, nb.types.Integer)
            and not isinstance(shape, nb.types.IntegerLiteral)
            and typer(nb.types.IntegerLiteral(1), dtype, alignment, alignas)
            is not None
        ):
            raise nb.force_literal_arg({0}, _fold_local_array_arguments)
        return sig

    return literal_shape_typer


LOCAL_ARRAY_SHAPE = Patch(
    name="argument-sized cuda.local.array",
    ask="U-P-1 (cuda.local.array requests a literal shape)",
    pins=(
        Pin(cuda_typing.GenericArrayTemplate, "generic", "280ceb177adabb9b"),
        Pin(cuda_typing.LocalArrayTemplate, "generic", None),
    ),
    build=lambda: (
        (cuda_typing.LocalArrayTemplate, "generic", _local_array_generic),
    ),
)

PATCHES = (LOCAL_ARRAY_SHAPE,)

_ABSENT = object()
_replaced: list[tuple[type, str, Any]] = []


def apply() -> None:
    """Check every patch, then apply them all; later calls do nothing."""

    if _replaced:
        return
    for patch in PATCHES:
        check(patch)
    for patch in PATCHES:
        for owner, name, value in patch.build():
            _replaced.append((owner, name, vars(owner).get(name, _ABSENT)))
            setattr(owner, name, value)


def replaced() -> tuple[tuple[type, str], ...]:
    """The ``(class, attribute)`` pairs ``apply`` replaced."""

    return tuple((owner, name) for owner, name, _ in _replaced)


__all__ = [
    "LOCAL_ARRAY_SHAPE",
    "NCM_VERSION",
    "PATCHES",
    "Patch",
    "Pin",
    "apply",
    "check",
    "replaced",
    "source_digest",
]
