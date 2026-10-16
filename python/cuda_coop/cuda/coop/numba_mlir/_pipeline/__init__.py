# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile cooperative kernels through Numba typing and direct MLIR lowering.

Backend activation registers this pipeline. It has four stages, all on
Numba-CUDA-MLIR's own extension points:

1. typing templates on cooperative Numba types (``_types``, ``_templates``),
   which give each cooperative call a request that knows nothing of the
   launch;
2. one after-inference call resolution per kernel (``_resolve``), which
   turns every request into a concrete provider call and whose decisions
   are plain data (``_plan``);
3. lowering of each resolved call to a provider call (``_lower``);
4. link items backed by the resolution's one compile (``_link``).

Nothing cooperative runs before typing. Every Numba-CUDA-MLIR interface that
is not public is reached through ``_bridges``, and every change to
Numba-CUDA-MLIR's own behaviour is a pinned patch in ``_ncm_patches``,
applied only by this pipeline. The supported surface is
``this_block()``, ``ThreadData``, and ``load`` and ``store`` on a block
group; any other cooperative call is a typing error, never a fallback.
"""

from __future__ import annotations

_activated = False


def _activate() -> None:
    """Register the pipeline's typing, rewrite and lowering hooks once.

    The backend's activation calls this under its activation lock.
    """

    global _activated
    if _activated:
        return
    from . import _bridges, _lower, _ncm_patches, _resolve, _templates, _types

    _ncm_patches.apply()
    _types.register()
    _templates.register()
    _lower.register()
    _resolve.register()
    _bridges.refresh_registries()
    _activated = True
