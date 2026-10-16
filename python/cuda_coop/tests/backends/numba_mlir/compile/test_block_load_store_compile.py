# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile storage-free and storage-bearing Load/Store kernels."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cuda.coop.numba_mlir import _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


_FIXED_COMPUTE_CAPABILITY = (9, 0)


@pytest.fixture(autouse=True)
def _fixed_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int]]:
    """Fix device discovery while retaining real NVRTC and nvJitLink calls."""

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "the Numba-CUDA-MLIR compile stage must hide all CUDA devices"
    )
    queries: list[tuple[int, int]] = []

    def current_device() -> SimpleNamespace:
        queries.append(_FIXED_COMPUTE_CAPABILITY)
        return SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    monkeypatch.setattr(_types.cuda, "get_current_device", current_device)
    return queries


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_production_routes_compile_storage_free_and_storage_bearing_kernels(
    monkeypatch: pytest.MonkeyPatch, items_per_thread
) -> None:
    import numba_cuda_mlir.tools as numba_mlir_tools
    from numba_cuda_mlir import cuda as compiler_cuda

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop

    fixed_device = SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    def fixed_compute_capability(as_type=str):
        assert as_type in (str, tuple)
        return _FIXED_COMPUTE_CAPABILITY if as_type is tuple else "sm_90"

    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        fixed_compute_capability,
    )
    monkeypatch.setattr(
        compiler_cuda, "get_current_device", lambda: fixed_device
    )

    @compiler_cuda.jit(chip="sm_90")
    def storage_free(source, destination, items_per_thread):
        storage = coop.TempStorage(
            128 * 1024, alignment=16, sharing="exclusive", auto_sync=True
        )
        payload = coop.ThreadData(items_per_thread, dtype=types.int32)
        coop.load(
            coop.this_block(),
            source,
            payload,
            algorithm="vectorize",
            temp_storage=storage,
        )
        coop.store(
            coop.this_block(),
            destination,
            payload,
            algorithm="vectorize",
            temp_storage=storage,
        )

    @compiler_cuda.jit(chip="sm_90")
    def storage_bearing(source, destination, items_per_thread):
        payload = qualified_coop.ThreadData(items_per_thread, dtype=types.int32)
        qualified_coop.load(
            qualified_coop.this_block(),
            source,
            payload,
            algorithm="warp_transpose_timesliced",
        )
        qualified_coop.store(
            qualified_coop.this_block(),
            destination,
            payload,
            algorithm="warp_transpose_timesliced",
        )

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.IntegerLiteral(items_per_thread),
    )
    launch_config_key = (
        ("grid", (1, 1, 1)),
        ("block", (64, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    results = [
        dispatcher._compile_launch_config_signature(
            signature, launch_config_key
        )
        for dispatcher in (storage_free, storage_bearing)
    ]
    for result in results:
        assert isinstance(result.metadata["ltoir"], bytes)
        assert result.metadata["ltoir"]
        assert isinstance(result.metadata["cubin"], bytes)
        assert result.metadata["cubin"]
        assert result.metadata["linked_external_link_items"]

    storage_free_ptx = next(iter(storage_free.inspect_lto_ptx().values()))
    storage_bearing_ptx = next(iter(storage_bearing.inspect_lto_ptx().values()))
    assert ".visible .entry" in storage_free_ptx
    assert ".shared" not in storage_free_ptx
    assert "bar.sync" not in storage_free_ptx
    assert ".visible .entry" in storage_bearing_ptx
    assert ".shared" in storage_bearing_ptx
    assert "bar.sync" in storage_bearing_ptx
