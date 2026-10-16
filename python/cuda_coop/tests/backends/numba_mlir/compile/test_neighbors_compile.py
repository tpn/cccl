# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile adjacent operations with inferred payload types."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

from cuda import coop

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture(autouse=True)
def _fixed_device(monkeypatch):
    """Select SM90 for compilation and require the GPU to remain hidden."""

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )


def _compile(kernel, signature):
    """Compile a known block launch without executing the kernel."""

    return kernel._compile_launch_config_signature(
        signature,
        (
            ("grid", (1, 1, 1)),
            ("block", (64, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_production_partial_with_inferred_payload(items_per_thread):
    """Infer float64 inputs, chain int32 flags, and retain the count trap."""

    @cuda.jit(chip="sm_90")
    def kernel(source, output, count, items_per_thread):
        values = coop.ThreadData(items_per_thread)
        for i in range(items_per_thread):
            values[i] = source[cuda.threadIdx.x * items_per_thread + i]
        differences = coop.adjacent_difference(
            coop.this_block(), values, valid_items=count
        )
        heads = coop.discontinuity(coop.this_block(), differences)
        for i in range(items_per_thread):
            output[cuda.threadIdx.x * items_per_thread + i] = heads[i]

    result = _compile(
        kernel,
        types.void(
            types.float64[::1],
            types.int32[::1],
            types.int64,
            types.IntegerLiteral(items_per_thread),
        ),
    )
    assert result.metadata["cubin"]
    assert "trap;" in next(iter(kernel.inspect_lto_ptx().values()))


@pytest.mark.parametrize(
    "options,match",
    [
        (
            {"direction": "right", "valid_items": 1, "tile_successor_item": 1},
            "partial",
        ),
        ({"valid_items": 193}, "valid_items"),
        ({"valid_items": True}, "valid_items"),
        ({"tile_predecessor_item": 1.5}, "tile_predecessor_item"),
    ],
)
def test_invalid_adjacent_options_rejected(options, match):
    # Bind options into a concrete function: variadic keyword forwarding is
    # intentionally outside the device frontend's contract.
    if "direction" in options:

        @cuda.jit(chip="sm_90")
        def kernel(source):
            values = coop.ThreadData(items_per_thread=3, dtype=types.int32)
            coop.load(coop.this_block(), source, values)
            return coop.adjacent_difference(
                coop.this_block(),
                values,
                direction="right",
                valid_items=1,
                tile_successor_item=1,
            )
    elif "valid_items" in options:
        count = options["valid_items"]

        @cuda.jit(chip="sm_90")
        def kernel(source):
            values = coop.ThreadData(items_per_thread=3, dtype=types.int32)
            coop.load(coop.this_block(), source, values)
            return coop.adjacent_difference(
                coop.this_block(), values, valid_items=count
            )
    else:

        @cuda.jit(chip="sm_90")
        def kernel(source):
            values = coop.ThreadData(items_per_thread=3, dtype=types.int32)
            coop.load(coop.this_block(), source, values)
            return coop.adjacent_difference(
                coop.this_block(), values, tile_predecessor_item=1.5
            )

    with pytest.raises(Exception, match=match):
        _compile(kernel, types.void(types.int32[::1]))
