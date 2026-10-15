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
"""

from __future__ import annotations

from collections.abc import Callable

from . import _bridges as nb
from ._errors import unsupported_primitive

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


def register() -> None:
    """Register a template for every kernel-callable public object."""

    for obj, qualified_name in kernel_primitives().items():
        register_template(
            obj,
            _rejecting(qualified_name),
            qualified_name.replace(".", "_"),
        )


__all__ = [
    "kernel_primitives",
    "register",
    "register_template",
]
