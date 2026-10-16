# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

# Import the kernel DSL before cuda.coop so automatic registration sees it.
# isort: off
import numpy as np
from numba_cuda_mlir import cuda, types

from cuda import coop  # First breakpoint: observe backend registration.
# isort: on


@cuda.jit
def copy_tile(source, destination, items_per_thread):
    block = coop.this_block()
    items = coop.ThreadData(items_per_thread)
    coop.load(block, source, items, algorithm="direct")
    coop.store(block, destination, items, algorithm="direct")


@cuda.jit
def group_queries(output):
    thread = coop.this_thread()
    warp = coop.this_warp()
    block = coop.this_block()
    grid = coop.this_grid()
    lanes = warp.group_by(8)
    pair = block.group_by(2, exhaustive=False)
    index = cuda.grid(1)

    thread.sync()
    warp.sync_aligned()
    lanes.sync_aligned()
    block.sync()

    output[0, index] = thread.rank("block")
    output[1, index] = warp.rank()
    output[2, index] = block.rank_as(types.int64, "grid")
    output[3, index] = block.count_as(types.int64)
    output[4, index] = grid.rank()
    output[5, index] = grid.count()
    output[6, index] = lanes.rank()
    output[7, index] = lanes.count()
    member = pair.is_member()
    output[8, index] = member
    if member:
        output[9, index] = pair.rank()
    else:
        output[9, index] = -1


@cuda.jit
def block_sum(source, output, items_per_thread):
    block = coop.this_block()
    items = coop.ThreadData(items_per_thread)
    for item in range(items_per_thread):
        items[item] = source[cuda.threadIdx.x * items_per_thread + item]
    total = coop.sum(block, items)
    if block.rank() == 0:
        output[0] = total


@cuda.jit
def scratch_reuse(source, sums, maxima, preserved, items_per_thread):
    block = coop.this_block()
    scratch = coop.TempStorage(auto_sync=True)
    items = coop.ThreadData(items_per_thread)
    for tile in range(2):
        offset = tile * cuda.blockDim.x * items_per_thread
        coop.load(
            block,
            source,
            items,
            offset=offset,
            algorithm="transpose",
            temp_storage=scratch,
        )
        total = coop.sum(block, items, temp_storage=scratch)
        maximum = coop.reduce(block, items, binary_op="max", temp_storage=scratch)
        coop.store(
            block,
            preserved,
            items,
            offset=offset,
            algorithm="transpose",
            temp_storage=scratch,
        )
        if block.rank() == 0:
            sums[tile] = total
            maxima[tile] = maximum


def run_copy(items_per_thread):
    """Copy a tile twice to inspect compilation and dispatcher reuse."""

    source = np.arange(128 * items_per_thread, dtype=np.int32)
    destination = np.zeros_like(source)

    copy_tile[1, 128](
        source, destination, items_per_thread
    )  # Compile, link, and launch.
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    print("First launch: copy verified")

    destination.fill(-1)
    copy_tile[1, 128](
        source, destination, items_per_thread
    )  # Reuse the compiled kernel.
    cuda.synchronize()
    np.testing.assert_array_equal(destination, source)
    print("Second launch: copy verified")


def run_groups():
    """Compare group coordinates and membership with independent host rows."""

    output = np.full((10, 192), -99, dtype=np.int64)
    indices = np.arange(192, dtype=np.int64)
    block_threads = indices % 96
    expected = np.stack(
        (
            block_threads,
            indices % 32,
            indices // 96,
            np.full(192, 96),
            indices,
            np.full(192, 192),
            indices % 8,
            np.full(192, 8),
            block_threads < 64,
            np.where(block_threads < 64, block_threads, -1),
        )
    )

    group_queries[2, 96](output)  # Compile, link, and launch.
    cuda.synchronize()
    np.testing.assert_array_equal(output, expected)
    print("First launch: groups verified")

    output.fill(-99)
    group_queries[2, 96](output)  # Reuse the compiled kernel.
    cuda.synchronize()
    np.testing.assert_array_equal(output, expected)
    print("Second launch: groups verified")


def run_reduce(items_per_thread):
    """Reduce a full tile and check the scalar written by the block root."""

    source = np.arange(64 * items_per_thread, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)
    expected = np.asarray([source.sum(dtype=np.int32)], dtype=np.int32)

    block_sum[1, 64](source, output, items_per_thread)  # Compile, link, and launch.
    cuda.synchronize()
    np.testing.assert_array_equal(output, expected)
    print("First launch: reduce verified")

    output.fill(-1)
    block_sum[1, 64](source, output, items_per_thread)  # Reuse the compiled kernel.
    cuda.synchronize()
    np.testing.assert_array_equal(output, expected)
    print("Second launch: reduce verified")


def run_scratch(items_per_thread):
    """Reuse automatically synchronized scratch across two full tiles."""

    source = np.arange(2 * 64 * items_per_thread, dtype=np.int32) % 17
    sums = np.zeros(2, dtype=np.int32)
    maxima = np.zeros_like(sums)
    preserved = np.full_like(source, -1)
    tiles = source.reshape(2, -1)
    expected_sums = tiles.sum(axis=1, dtype=np.int32)
    expected_maxima = tiles.max(axis=1)

    scratch_reuse[1, 64](
        source, sums, maxima, preserved, items_per_thread
    )  # Compile, link, and launch.
    cuda.synchronize()
    np.testing.assert_array_equal(sums, expected_sums)
    np.testing.assert_array_equal(maxima, expected_maxima)
    np.testing.assert_array_equal(preserved, source)
    print("First launch: scratch verified")

    sums.fill(-1)
    maxima.fill(-1)
    preserved.fill(-1)
    scratch_reuse[1, 64](
        source, sums, maxima, preserved, items_per_thread
    )  # Reuse the compiled kernel.
    cuda.synchronize()
    np.testing.assert_array_equal(sums, expected_sums)
    np.testing.assert_array_equal(maxima, expected_maxima)
    np.testing.assert_array_equal(preserved, source)
    print("Second launch: scratch verified")


def main(items_per_thread=2, stage=None):
    """Run the core tour, or select one stage including optional scratch."""

    if items_per_thread <= 0:
        raise ValueError("items_per_thread must be positive")
    if stage not in (None, "copy", "groups", "reduce", "scratch"):
        raise ValueError(f"unknown walkthrough stage: {stage}")

    if stage is None or stage == "copy":
        run_copy(items_per_thread)
    if stage is None or stage == "groups":
        run_groups()
    if stage is None or stage == "reduce":
        run_reduce(items_per_thread)
    if stage == "scratch":
        run_scratch(items_per_thread)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug cuda.coop compilation.")
    parser.add_argument(
        "--stage",
        choices=("copy", "groups", "reduce", "scratch"),
        help="run one stage; by default run copy, groups, then reduce",
    )
    parser.add_argument(
        "--items-per-thread",
        type=int,
        default=2,
        help="positive payload size for copy, reduce, and scratch (default: 2)",
    )
    args = parser.parse_args()
    if args.items_per_thread <= 0:
        parser.error("--items-per-thread must be positive")
    main(items_per_thread=args.items_per_thread, stage=args.stage)
