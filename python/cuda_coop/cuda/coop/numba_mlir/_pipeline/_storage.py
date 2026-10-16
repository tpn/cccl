# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Size each site's scratch and lay it out in shared memory.

Storage analysis is the third stage of ``CoopCallResolution``, and the only
place the typed pipeline calls NVRTC. It compiles the kernel's providers in
one bundle (``_link``), which also reports each provider's scratch size and
alignment, and lays scratch out in one implicit region
sized to the largest need, backed by one static ``cuda.shared.array``
that, with the kernel's own shared memory, fits the 48 KB static limit.

Which sites take scratch was decided before the compile, from each
provider's storage ABI; the compiled layouts must agree. ``layout`` is pure
and takes plain sizes. This stage does not change the IR.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import _link
from ._errors import TypedPipelineError
from ._plan import Backing, Region
from ._specialize import Specialization

STATIC_SHARED_LIMIT = 48 * 1024
IMPLICIT_REGION = "implicit.0"
_BACKING_ALIGNMENT = 16


@dataclass(frozen=True)
class StorageLayout:
    """The kernel's scratch backing and region, and which sites use them.

    ``backing`` and ``region`` are ``None`` when no site takes scratch.
    """

    backing: Backing | None
    region: Region | None
    scratch: tuple[bool, ...]


def layout(
    needs: Sequence[tuple[int, int] | None], user_shared_bytes: int
) -> StorageLayout:
    """Lay out the sites' scratch ``needs``, one ``(size, alignment)`` each.

    A site without scratch needs ``None``. Every site shares one region, so
    the region is as large as the largest need, not their sum.
    """

    scratch = tuple(need is not None for need in needs)
    used = [need for need in needs if need is not None]
    if not used:
        return StorageLayout(None, None, scratch)
    size = max(s for s, _ in used)
    alignment = max(a for _, a in used)
    backing_alignment = max(_BACKING_ALIGNMENT, alignment)
    backing_size = -(-size // backing_alignment) * backing_alignment
    if backing_size + user_shared_bytes > STATIC_SHARED_LIMIT:
        raise TypedPipelineError(
            f"cooperative scratch needs {backing_size} bytes of shared "
            f"memory, which with the kernel's own {user_shared_bytes} bytes "
            "exceeds "
            f"the {STATIC_SHARED_LIMIT}-byte static limit. Scratch above "
            "48 KB is not supported on the typed pipeline; use a smaller "
            "block, fewer items per thread, or a storage-free algorithm "
            "(direct, striped, vectorize)."
        )
    return StorageLayout(
        Backing("static", backing_size, backing_alignment),
        Region(IMPLICIT_REGION, 0, size, alignment),
        scratch,
    )


def analyze(
    specialization: Specialization,
    user_shared_bytes: int,
    identity: Any,
    context: Any,
) -> tuple[StorageLayout, _link.Bundle]:
    """Compile the specialization's bundle and lay out its scratch."""

    bundle = _link.compile_bundle(specialization.algorithms, identity, context)
    needs = []
    for algorithm, scratch in zip(
        specialization.site_algorithms, specialization.scratch, strict=True
    ):
        need = bundle.layouts[algorithm._private_symbol_digest]
        if (need is not None) != scratch:
            raise AssertionError(
                f"{algorithm.c_name}: its {algorithm.storage_abi} storage "
                f"ABI disagrees with its compiled scratch layout {need}"
            )
        needs.append(need)
    return layout(needs, user_shared_bytes), bundle


__all__ = [
    "IMPLICIT_REGION",
    "STATIC_SHARED_LIMIT",
    "StorageLayout",
    "analyze",
    "layout",
]
