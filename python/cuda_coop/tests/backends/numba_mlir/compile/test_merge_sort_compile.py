# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile Merge Sort kernels with inferred payload types."""

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
    """Select SM90 for compilation while requiring every GPU to be hidden."""

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


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_production_partial_pairs_with_dtype_inference(items_per_thread):
    """Compile inferred pair types with count guards and warp barriers."""

    @cuda.jit(chip="sm_90")
    def kernel(source, values, output, associations, count, items_per_thread):
        keys = coop.ThreadData(items_per_thread)
        payload = coop.ThreadData(items_per_thread)
        thread = cuda.threadIdx.x
        for item in range(items_per_thread):
            keys[item] = source[thread * items_per_thread + item]
            payload[item] = values[thread * items_per_thread + item]
        result, result_values = coop.merge_sort_pairs(
            coop.this_warp().group_by(8),
            keys,
            payload,
            valid_items=count,
            oob_default=1000,
        )
        for item in range(items_per_thread):
            output[thread * items_per_thread + item] = result[item]
            associations[thread * items_per_thread + item] = result_values[item]

    signature = types.void(
        types.int32[::1],
        types.float64[::1],
        types.int32[::1],
        types.float64[::1],
        types.int64,
        types.IntegerLiteral(items_per_thread),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        (
            ("grid", (1, 1, 1)),
            ("block", (64, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )
    assert result.metadata["cubin"]
    ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert "trap;" in ptx
    assert "bar.warp.sync" in ptx
