# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise actual asynchronous global-to-shared transfers and native views."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
]


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_load_to_shared_example(items_per_thread):
    # load-to-shared-example-begin
    import numpy as np
    from numba_cuda_mlir import cuda, types

    from cuda import coop

    threads = 32
    tile_items = threads * items_per_thread
    buffer_items = tile_items + 4
    blocks = 3

    @cuda.jit
    def stage(source, destination, items_per_thread):
        block = coop.this_block()
        workspace = coop.TempStorage()
        # CUB needs up to 16 additional bytes for this int32 range.
        buffer = workspace.reserve(buffer_items, types.int32, alignment=16)
        base = cuda.blockIdx.x * tile_items
        thread = cuda.threadIdx.x
        pending = coop.load_to_shared_async(
            block, source, buffer, num_items=tile_items, offset=base + 1
        )

        # This computation is independent of the in-flight shared writes.
        independent = source[base + thread] * 7
        values = pending.wait()

        # Read neighboring threads' staged values through the returned view.
        for item in range(items_per_thread):
            index = thread * items_per_thread + item
            destination[base + index] = (
                values[(index + 1) % tile_items] + independent
            )

    source = np.arange(blocks * tile_items + 1, dtype=np.int32)
    result = cuda.device_array(blocks * tile_items, dtype=np.int32)
    stage[blocks, threads](cuda.to_device(source), result, items_per_thread)
    expected = np.concatenate(
        [
            np.roll(source[base + 1 : base + tile_items + 1], -1)
            + np.repeat(source[base : base + threads] * 7, items_per_thread)
            for base in range(0, blocks * tile_items, tile_items)
        ]
    )
    np.testing.assert_array_equal(result.copy_to_host(), expected)
    # load-to-shared-example-end


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("num_items", [0, 7, 32])
@pytest.mark.parametrize("reserved", [False, True], ids=["native", "reserved"])
def test_shared_copy_tail_and_zero(api, num_items, reserved):
    capacity = ((num_items * 4 + 15) // 16 * 16 + 16) // 4

    @cuda.jit
    def kernel(source, output):
        if reserved:
            storage = api.TempStorage()
            buffer = storage.reserve(capacity, types.int32, alignment=16)
        else:
            buffer = cuda.shared.array(capacity, types.int32, alignment=16)
        pending = api.load_to_shared_async(
            api.this_block(), source, buffer, num_items=num_items, offset=1
        )
        values = pending.wait()
        thread = cuda.threadIdx.x
        if thread < num_items:
            output[thread] = values[(thread + 1) % num_items]
        if thread == 0:
            output[num_items] = values.size

    source = np.arange(num_items + 1, dtype=np.int32) * 11 - 9
    output = cuda.device_array(num_items + 1, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output)
    expected = np.concatenate((np.roll(source[1:], -1), [num_items]))
    np.testing.assert_array_equal(output.copy_to_host(), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float64])
def test_shifted_partial_copy_element_sizes(dtype):
    num_items = 7
    element_type = types.uint8 if dtype is np.uint8 else types.float64
    capacity = (num_items * np.dtype(dtype).itemsize + 31) // np.dtype(
        dtype
    ).itemsize

    @cuda.jit
    def kernel(source, output):
        buffer = cuda.shared.array(capacity, element_type, alignment=16)
        # Exercise a contiguous alias as well as a misaligned source address.
        source_view = source[1:]
        pending = coop.load_to_shared_async(
            coop.this_block(), source_view, buffer, num_items=num_items
        )
        values = pending.wait()
        thread = cuda.threadIdx.x
        if thread < num_items:
            output[thread] = values[(thread + 1) % num_items]

    source = np.arange(num_items + 1, dtype=dtype) * 3
    output = cuda.device_array(num_items, dtype=dtype)
    kernel[1, 32](cuda.to_device(source), output)
    np.testing.assert_array_equal(
        output.copy_to_host(), np.roll(source[1:], -1)
    )


def test_two_pending_copies_survive_independent_control_flow_and_scratch():
    @cuda.jit
    def kernel(source, output):
        block = coop.this_block()
        workspace = coop.TempStorage(sharing="exclusive")
        first_buffer = workspace.reserve(36, types.int32, alignment=16)
        second_buffer = workspace.reserve(36, types.int32, alignment=16)
        first = coop.load_to_shared_async(
            block, source, first_buffer, num_items=32, offset=1
        )
        second = coop.load_to_shared_async(
            block, source, second_buffer, num_items=32, offset=33
        )
        thread = cuda.threadIdx.x
        independent = types.int64(source[thread])
        if thread % 2:
            independent += 3
        else:
            independent += 5
        for _ in range(source[0]):
            independent += 2
        total = coop.sum(block, independent)
        left = first.wait()
        right = second.wait()
        output[thread] = (
            left[(thread + 1) % 32] + right[(thread + 2) % 32] + independent
        )
        if thread == 0:
            output[32] = total

    source = np.arange(65, dtype=np.int32) + 2
    output = cuda.device_array(33, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output)
    independent = (
        source[:32] + np.where(np.arange(32) % 2, 3, 5) + 2 * source[0]
    )
    expected = np.concatenate(
        (
            np.roll(source[1:33], -1)
            + np.roll(source[33:65], -2)
            + independent,
            [independent.sum()],
        )
    )
    np.testing.assert_array_equal(output.copy_to_host(), expected)
