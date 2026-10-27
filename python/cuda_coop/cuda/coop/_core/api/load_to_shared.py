# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Start one block-wide global-to-shared transfer and wait for its result."""

from __future__ import annotations

from typing import Any, Protocol

from cuda.coop._typing import IntegerValue

from ._dispatch import _common_group_operation, _group_primitive_marker
from .thread_group import BlockGroup


class PendingSharedCopyLike(Protocol):
    """An issued shared-memory copy that must be waited on exactly once.

    Obtain this object from :func:`load_to_shared_async`. It keeps the copy's
    completion state alive until ``wait()`` returns the native shared view.
    """

    def wait(self) -> Any:
        """Complete the copy and return the shared view containing its values.

        Every thread in the issuing block must participate. The returned
        array or tensor has the source dtype and ``num_items`` elements.
        Its first element can lie after padding in the supplied buffer.
        Waiting consumes the pending copy, including aliases of that handle.
        The destination buffer remains owned by its original allocation.
        """
        ...


@_common_group_operation("load_to_shared_async", group_kinds=("block",))
def load_to_shared_async(
    group: BlockGroup,
    source: object,
    shared_buffer: object,
    /,
    *,
    num_items: int,
    offset: IntegerValue = 0,
) -> PendingSharedCopyLike:
    """Issue and commit one global-to-shared copy for a complete block.

    Call ``pending.wait()`` before accessing the copied values. Independent
    work can execute between issue and wait. This operation copies one
    contiguous range for the whole block; it has no per-thread item layout.

    Parameters
    ----------
    group : ThreadGroup
        A block whose size is a multiple of 32. All threads participate in
        issue and wait with the same source, destination and range.
    source : array
        Contiguous one-dimensional global-memory array or a supported native
        global pointer. Its element dtype
        determines the result dtype. Keep the source unchanged until wait
        completes. The selected range must be within the source allocation.
    shared_buffer : array
        Contiguous one-dimensional shared-memory buffer with the source
        dtype, aligned to at least 16 bytes. Its byte capacity must be at
        least ``round_up(num_items * itemsize, 16) + 16`` for a nonempty copy.
        CUB can place the data after head padding in this buffer. Consume
        the view returned by ``wait()`` to obtain the correct data origin.
        Keep this allocation disjoint from other pending destinations and
        live primitive scratch, and do not access it before wait completes.
        Allocate with ``TempStorage.reserve(..., alignment=16)``, Numba's
        ``cuda.shared.array(..., alignment=16)``, or CuTe's
        ``SmemAllocator.allocate_tensor(..., byte_alignment=16)``.
    num_items : int
        Nonnegative compile-time number of elements in the whole transfer.
        The byte count must fit a signed 32-bit integer. Zero copies no data
        and still requires a matching wait.
    offset : int, optional
        Nonnegative source offset in elements, default zero. The source end
        is ``offset + num_items``. Offset sources require only their normal
        element alignment; the buffer padding accommodates weaker alignment
        than the bulk-copy instruction requires.

    Returns
    -------
    PendingSharedCopyLike
        A single-use pending copy. Its ``wait()`` method returns a native
        shared array or tensor, not a per-thread ``ThreadData`` payload.

    Notes
    -----
    This draft requires issue and wait in the kernel's outer region.
    Issuing or waiting inside runtime branches or loops is unsupported.
    Every issued copy must be waited on exactly once before the kernel ends.
    Aliasing a pending handle does not create another transaction.

    The compiler keeps the copy's barrier storage separate from ordinary
    cooperative scratch. Wait completes and retires that barrier state; it
    does not synchronize future readers before reuse of the data buffer.
    Synchronize the block after its last reads before overwriting that buffer.

    The implementation uses CUB's ``detail::BlockLoadToShared``. Hopper and
    newer devices use bulk asynchronous copies with an mbarrier, Ampere uses
    ``cp.async``, and older devices use a synchronous fallback. Completion
    semantics are the same on all paths; overlap is hardware-dependent.
    """

    return _group_primitive_marker(
        "load_to_shared_async",
        group,
        source,
        shared_buffer,
        num_items=num_items,
        offset=offset,
    )


__all__ = ["PendingSharedCopyLike", "load_to_shared_async"]
