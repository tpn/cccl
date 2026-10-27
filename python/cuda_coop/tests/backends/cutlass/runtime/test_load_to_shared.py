# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise split global-to-shared copies and the returned padded view."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.memory import SmemAllocator

from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_load_to_shared_example(items_per_thread):
    from cuda import coop

    # load-to-shared-cutlass-example-begin
    threads = 64
    blocks = 3

    @cute.kernel
    def stage(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        rank = block.rank()
        block_index = cute.arch.block_idx()[0]
        tile_items = threads * items_per_thread
        memory = cute.make_tensor(
            source, cute.make_layout(blocks * tile_items + 1)
        )
        output = cute.make_tensor(
            destination, cute.make_layout(blocks * tile_items)
        )
        workspace = coop.TempStorage()
        # Four extra int32 elements cover CUB's 16-byte padding requirement.
        shared_buffer = workspace.reserve(
            tile_items + 4, cutlass.Int32, alignment=16
        )
        source_offset = block_index * tile_items + 1
        pending = coop.load_to_shared_async(
            block,
            memory,
            shared_buffer,
            num_items=tile_items,
            offset=source_offset,
        )

        # This work is independent of the shared destination and used below.
        independent = memory[source_offset + rank] * 2 + cutlass.Int32(rank)
        for iteration in range(block_index + 1):
            independent += iteration
        shared_values = pending.wait()

        for item in cutlass.range_constexpr(items_per_thread):
            neighbor = ((rank + 1) % threads) * items_per_thread + item
            index = block_index * tile_items + rank * items_per_thread + item
            output[index] = shared_values[neighbor] + independent

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        stage(source, destination, items_per_thread).launch(
            grid=blocks, block=(8, 8, 1)
        )

    tile_items = threads * items_per_thread
    source = np.arange(blocks * tile_items + 1, dtype=np.int32) * 3 - 5
    output = np.zeros(blocks * tile_items, dtype=np.int32)
    with device_array(source) as src, device_array(output) as dst:
        launch(src, dst, items_per_thread)
    expected = []
    for block_index in range(blocks):
        tile = source[
            block_index * tile_items + 1 : (block_index + 1) * tile_items + 1
        ]
        neighbor = np.roll(tile.reshape(threads, items_per_thread), -1, axis=0)
        independent = tile[:threads] * 2 + np.arange(threads, dtype=np.int32)
        independent += block_index * (block_index + 1) // 2
        expected.extend((neighbor + independent[:, None]).ravel())
    np.testing.assert_array_equal(output, np.asarray(expected, dtype=np.int32))
    # load-to-shared-cutlass-example-end


@pytest.mark.parametrize("api", ["common", "qualified"])
@pytest.mark.parametrize(
    "count,offset,dtype,scalar_type",
    [
        pytest.param(0, 0, np.int32, cutlass.Int32, id="empty-int32"),
        pytest.param(3, 1, np.int32, cutlass.Int32, id="partial-int32"),
        pytest.param(37, 2, np.int32, cutlass.Int32, id="tail-int32"),
        pytest.param(37, 1, np.uint8, cutlass.Uint8, id="partial-uint8"),
        pytest.param(3, 1, np.float64, cutlass.Float64, id="partial-float64"),
    ],
)
def test_native_buffer_partial_copy(api, count, offset, dtype, scalar_type):
    if api == "common":
        from cuda import coop
    else:
        import cuda.coop.cutlass as coop

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        block = coop.this_block()
        itemsize = scalar_type.width // 8
        capacity = max(
            16 // itemsize,
            ((count * itemsize + 15) // 16 * 16 + 16) // itemsize,
        )
        shared = SmemAllocator().allocate_tensor(
            scalar_type, cute.make_layout(capacity), byte_alignment=16
        )
        # Preserve unsigned metadata through CuTe's signless pointer ABI.
        memory = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(max(1, count + offset))),
            scalar_type,
        )
        pending = coop.load_to_shared_async(
            block, memory, shared, num_items=count, offset=offset
        )
        values = pending.wait()
        assert cutlass.const_expr(values.element_type is scalar_type)
        output = cute.recast_tensor(
            cute.make_tensor(destination, cute.make_layout(max(1, count))),
            scalar_type,
        )
        rank = block.rank()
        if cutlass.const_expr(count == 0):
            if rank == 0:
                output[0] = 17
        else:
            if rank < count:
                output[rank] = values[(rank + 1) % count]

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=64)

    source = np.arange(max(1, count + offset), dtype=dtype) * 7
    output = np.zeros(max(1, count), dtype=dtype)
    with device_array(source) as src, device_array(output) as dst:
        launch(src, dst)
    expected = (
        np.roll(source[offset : offset + count], -1)
        if count
        else np.array([17], dtype=np.int32)
    )
    np.testing.assert_array_equal(output, expected)


def test_two_outstanding_copies():
    from cuda import coop

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        block = coop.this_block()
        workspace = coop.TempStorage(sharing="exclusive")
        first_buffer = workspace.reserve(20, cutlass.Int32, alignment=16)
        second_buffer = workspace.reserve(20, cutlass.Int32, alignment=16)
        first = coop.load_to_shared_async(
            block, source, first_buffer, num_items=16, offset=1
        )
        second = coop.load_to_shared_async(
            block, source, second_buffer, num_items=16, offset=18
        )
        first_values = first.wait()
        second_values = second.wait()
        output = cute.make_tensor(destination, cute.make_layout(16))
        rank = block.rank()
        if rank < 16:
            output[rank] = (
                first_values[(rank + 1) % 16] + 2 * second_values[rank]
            )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=64)

    source = np.arange(34, dtype=np.int32) * 7
    output = np.zeros(16, dtype=np.int32)
    with device_array(source) as src, device_array(output) as dst:
        launch(src, dst)
    np.testing.assert_array_equal(
        output, np.roll(source[1:17], -1) + 2 * source[18:34]
    )
