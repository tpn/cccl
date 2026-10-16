# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose per-thread payload construction to the Numba-CUDA-MLIR compiler.

``ThreadData`` is a kernel-language marker: the typed pipeline lowers a
supported call to a fixed-size local array, inferring its dtype from the
surrounding operations when possible. Calling the Python function directly
raises a compiler-context error.

``local`` and ``shared`` expose the active runtime's array namespaces. They
are loaded on first attribute access so importing this module does not itself
require those runtime namespaces to be initialized.
"""

from __future__ import annotations

from .._core.thread_group import CoopCompilerContextRequiredError
from ._compiler._activation import _require_runtime

# Annotations keep the runtime namespaces lazy while documenting module
# ownership for introspection and static analysis.
local: object
shared: object


def ThreadData(
    items_per_thread,
    dtype=None,
    *,
    alignment=None,
):
    """Create a fixed-size Numba local array for cooperative operations.

    The parameters, uninitialized contents, and example follow
    :func:`cuda.coop.ThreadData`. This qualified constructor returns the
    active Numba-CUDA-MLIR runtime's local-array representation. The compiler
    infers the element type from supported context, including Load's
    source, Store's destination, and typed indexed assignments. Leave the
    constructor type unspecified for those uses. See
    :ref:`element-type inference <coop-faq-thread-data-dtype>` for cases that
    need additional type information.

    Prefer a kernel parameter named ``items_per_thread`` and construct the
    payload with ``ThreadData(items_per_thread)``. The compiler specializes
    the kernel for each supplied count, which remains fixed during execution.

    ``alignment`` requests a minimum alignment in bytes. This backend also
    enforces the compiler's pointer-alignment requirement. See
    :ref:`per-thread payloads <coop-thread-data>` for indexing and lifetime.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.numba_mlir.ThreadData must be called from a supported "
        "GPU kernel."
    )


def __getattr__(name: str):
    """Resolve and cache an array namespace from the active runtime."""
    if name in {"local", "shared"}:
        value = getattr(_require_runtime(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "ThreadData",
    "local",
    "shared",
]
