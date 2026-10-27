# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check direct CUB writes into shared rows and cross-thread consumption."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.memory import SmemAllocator

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize(
    "api", [coop, cutlass_coop], ids=["common", "qualified"]
)
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "group_kind,algorithm",
    [
        ("block", "direct"),
        ("block", "transpose"),
        ("block", "striped"),
        ("warp", "transpose"),
        ("logical", "direct"),
        ("logical", "striped"),
    ],
)
def test_shared_rows(api, items_per_thread, group_kind, algorithm):
    threads, blocks = 64, 2
    group_width = {"block": 64, "warp": 32, "logical": 8}[group_kind]

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = api.this_block()
        group = block
        if cutlass.const_expr(group_kind == "warp"):
            group = api.this_warp()
        elif cutlass.const_expr(group_kind == "logical"):
            group = api.this_warp().group_by(8)
        rank = block.rank()
        block_index = cute.arch.block_idx()[0]
        tile_items = threads * items_per_thread
        memory = cute.make_tensor(
            source, cute.make_layout(blocks * tile_items + 1)
        )
        storage = SmemAllocator().allocate_tensor(
            cutlass.Int32, cute.make_layout(tile_items + 2), byte_alignment=4
        )
        # A view selects the destination origin, independently of source offset.
        shared = cute.make_tensor(
            storage.iterator + 1, cute.make_layout(tile_items)
        )
        if rank < 2:
            storage[rank * (tile_items + 1)] = -31
        api.load(
            group,
            memory,
            shared,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
            offset=block_index * tile_items + 1,
            valid_items=group_width * items_per_thread - 1,
            oob_default=-7,
        )
        block.sync()
        output = cute.make_tensor(
            destination, cute.make_layout(blocks * (tile_items + 2))
        )
        for item in cutlass.range_constexpr(items_per_thread):
            output[
                block_index * (tile_items + 2) + rank * items_per_thread + item
            ] = shared[((rank + 1) % threads) * items_per_thread + item]
        if rank < 2:
            output[block_index * (tile_items + 2) + tile_items + rank] = (
                storage[rank * (tile_items + 1)]
            )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, items_per_thread).launch(
            grid=blocks, block=(8, 4, 2)
        )

    tile_items = threads * items_per_thread
    source = np.arange(blocks * tile_items + 1, dtype=np.int32) * 3 - 11
    output = np.zeros(blocks * (tile_items + 2), dtype=np.int32)
    with device_array(source) as src, device_array(output) as dst:
        launch(src, dst, items_per_thread)
    expected = []
    for block_index in range(blocks):
        values = source[
            block_index * tile_items + 1 : (block_index + 1) * tile_items + 1
        ].copy()
        groups = values.reshape(-1, group_width * items_per_thread)
        groups[:, -1] = -7
        if algorithm == "striped":
            rows = groups.reshape(-1, items_per_thread, group_width).transpose(
                0, 2, 1
            )
        else:
            rows = groups.reshape(-1, group_width, items_per_thread)
        expected.extend(
            np.roll(rows.reshape(threads, items_per_thread), -1, axis=0).ravel()
        )
        expected.extend([-31, -31])
    np.testing.assert_array_equal(output, np.asarray(expected, dtype=np.int32))


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_reserved_shared_example(items_per_thread):
    # shared-destination-cutlass-example-begin
    threads = 64

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block = coop.this_block()
        rank = block.rank()
        tile_items = threads * items_per_thread
        workspace = coop.TempStorage(sharing="exclusive")
        shared = workspace.reserve(tile_items, cutlass.Int32)
        coop.load(
            block,
            source,
            shared,
            items_per_thread=items_per_thread,
            algorithm="transpose",
            temp_storage=workspace,
        )
        block.sync()
        output = cute.make_tensor(destination, cute.make_layout(tile_items))
        for item in cutlass.range_constexpr(items_per_thread):
            output[rank * items_per_thread + item] = shared[
                ((rank + 1) % threads) * items_per_thread + item
            ]

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, items_per_thread).launch(
            grid=1, block=threads
        )

    source = np.arange(threads * items_per_thread, dtype=np.int32)
    output = np.zeros_like(source)
    with device_array(source) as src, device_array(output) as dst:
        launch(src, dst, items_per_thread)
    expected = np.roll(source.reshape(threads, items_per_thread), -1, axis=0)
    np.testing.assert_array_equal(output, expected.ravel())
    # shared-destination-cutlass-example-end


@pytest.mark.parametrize(
    "dtype,scalar_type",
    [(np.uint8, cutlass.Uint8), (np.float64, cutlass.Float64)],
)
def test_native_shared_widths(dtype, scalar_type):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        block = coop.this_block()
        rank = block.rank()
        memory = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(256)), scalar_type
        )
        shared = SmemAllocator().allocate_tensor(
            scalar_type,
            cute.make_layout(256),
            byte_alignment=scalar_type.width // 8,
        )
        cutlass_coop.load(block, memory, shared, items_per_thread=4)
        block.sync()
        output = cute.recast_tensor(
            cute.make_tensor(destination, cute.make_layout(256)), scalar_type
        )
        for item in cutlass.range_constexpr(4):
            output[rank * 4 + item] = shared[((rank + 1) % 64) * 4 + item]

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=64)

    source = np.arange(256, dtype=dtype)
    output = np.zeros_like(source)
    with device_array(source) as src, device_array(output) as dst:
        launch(src, dst)
    np.testing.assert_array_equal(
        output, np.roll(source.reshape(64, 4), -1, axis=0).ravel()
    )
