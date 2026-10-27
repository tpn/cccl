# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Issue a block-wide copy into shared memory without waiting for it."""

from __future__ import annotations

from typing import Any

from .._core.api.load_to_shared import PendingSharedCopyLike
from .._core.api.thread_group import BlockGroup


def load_to_shared_async(
    group: BlockGroup,
    source: object,
    shared_buffer: object,
    /,
    *,
    num_items: int,
    offset: Any = 0,
) -> PendingSharedCopyLike:
    """Issue a global-to-shared copy with the common API contract.

    See :func:`cuda.coop.load_to_shared_async` for buffer padding,
    synchronization, and single-use wait requirements. Sources must expose
    a contiguous global-memory pointer. The destination must be a compact
    one-dimensional shared tensor with a known capacity and at least 16-byte
    alignment, such as a suitably aligned ``TempStorage.reserve`` result.

    The pending object's ``wait()`` returns a compact native CuTe shared
    tensor with the source dtype and ``num_items`` elements. Its pointer can
    differ from the supplied buffer's pointer because of CUB head padding.
    This draft requires issue and wait in the kernel's entry block. Runtime
    branches or loops may contain independent work between them, but cannot
    issue, wait on, or carry a pending copy.
    """

    from ._lowering._load_to_shared import provider_load_to_shared_async

    return provider_load_to_shared_async(
        group, source, shared_buffer, num_items=num_items, offset=offset
    )


__all__ = ["load_to_shared_async"]
