# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile Scan kernels and their callback descriptors without a GPU."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

from cuda.coop.numba_mlir import StatefulFunction, _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


_FIXED_COMPUTE_CAPABILITY = (9, 0)


_BLOCK_THREADS = 64


@pytest.fixture(autouse=True)
def _fixed_compiler_target(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int]]:
    """Fix architecture queries while keeping compilation and linking real.

    Record device queries so tests can check that the target was consulted.
    Clear compiled callback IR before each test so that cache cannot hide
    compilation for the selected target. CUDA devices must remain hidden.
    """

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "the Numba-CUDA-MLIR compile stage must hide all CUDA devices"
    )
    queries: list[tuple[int, int]] = []

    def current_device() -> SimpleNamespace:
        queries.append(_FIXED_COMPUTE_CAPABILITY)
        return SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    def compute_capability(as_type=str):
        assert as_type in (str, tuple)
        return _FIXED_COMPUTE_CAPABILITY if as_type is tuple else "sm_90"

    monkeypatch.setattr(_types.cuda, "get_current_device", current_device)
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        compute_capability,
    )
    _types._DEVICE_LTOIR_CACHE.clear()
    return queries


def test_production_kernel_compile_consumes_prefix_descriptors() -> None:
    """Compile public Scan calls through planning, lowering, and device link.

    Use ThreadData and local-array state with different dtypes, followed by
    a stateless array scan. One kernel must resolve each descriptor and link
    its callback. Stateful calls retain the state array as a runtime operand.
    The configured launch supplies block geometry without launching a GPU.
    """

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def stateless_prefix(block_aggregate):
        return block_aggregate + 7

    @cuda.jit(device=True)
    def running_prefix(state, block_aggregate):
        previous = state[0]
        state[0] = previous + block_aggregate
        return previous

    running = StatefulFunction(
        running_prefix,
        types.int64,
        name="compile_only_running_prefix",
    )
    running_int32 = StatefulFunction(
        running_prefix,
        types.int32,
        name="compile_only_running_prefix",
    )

    @cuda.jit(chip="sm_90")
    def kernel(source, destination, final_state):
        thread = cuda.threadIdx.x
        state = coop.ThreadData(items_per_thread=1, dtype=types.int64)
        state[0] = 11
        destination[thread] = coop.exclusive_sum(
            coop.this_block(),
            source[thread],
            state,
            prefix_op=running,
            algorithm="raking",
        )
        state_int32 = cuda.local.array(1, dtype=types.int32)
        state_int32[0] = 5
        destination[_BLOCK_THREADS + thread] = coop.exclusive_sum(
            coop.this_block(),
            source[thread],
            state_int32,
            prefix_op=running_int32,
            algorithm="raking_memoize",
        )
        values = cuda.local.array(2, dtype=types.int32)
        values[0] = source[thread * 2]
        values[1] = source[thread * 2 + 1]
        scanned = coop.inclusive_sum(
            coop.this_block(),
            values,
            prefix_op=stateless_prefix,
            algorithm="warp_scans",
        )
        destination[2 * _BLOCK_THREADS + thread * 2] = scanned[0]
        destination[2 * _BLOCK_THREADS + thread * 2 + 1] = scanned[1]
        if thread == 0:
            final_state[0] = state[0]

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.int64[::1],
    )
    launch_config_key = (
        ("grid", (1, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        launch_config_key,
    )

    assert isinstance(result.metadata["ltoir"], bytes)
    assert result.metadata["ltoir"]
    assert isinstance(result.metadata["cubin"], bytes)
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]
    ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert ".visible .entry" in ptx
    assert ".shared" in ptx
    assert "bar.sync" in ptx


@pytest.mark.parametrize("auto_sync", [None, False, True])
def test_production_kernel_compile_accepts_descriptors_through_inlined_helpers(
    auto_sync,
) -> None:
    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def scan_with(storage, value):
        return coop.inclusive_sum(
            coop.this_block(), value, temp_storage=storage
        )

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        # No group marker in the kernel body: every cooperative call lives in
        # the helper and becomes visible only after inlining. Acceptance must
        # not depend on an unrelated marker being present.
        thread = cuda.threadIdx.x
        storage = coop.TempStorage(auto_sync=auto_sync)
        first = scan_with(storage, source[thread])
        destination[thread] = scan_with(storage, first)

    signature = types.void(types.int32[::1], types.int32[::1])
    launch_config_key = (
        ("grid", (1, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        launch_config_key,
    )

    assert result.metadata["cubin"]
    # Only explicit opt-in adds a trailing barrier to each inlined call.
    assert result.metadata["mlir_module_str"].count("gpu.barrier") == (
        2 if auto_sync is True else 0
    )


def test_primitive_inside_standalone_scan_callback_has_clear_diagnostic():
    """Preserve the planning error from a callback that calls a primitive.

    Prefix callbacks compile as standalone device functions. Their nested
    primitive lacks the kernel launch context required by group planning.
    The diagnostic must identify that callback and explain the inlining
    requirement, even when provider construction triggers the compilation.
    """

    from numba_cuda_mlir.numba_cuda.core.errors import TypingError

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def primitive_prefix(aggregate):
        return coop.inclusive_sum(coop.this_block(), aggregate)

    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        thread = cuda.threadIdx.x
        destination[thread] = coop.exclusive_sum(
            coop.this_block(), source[thread], prefix_op=primitive_prefix
        )

    key = (
        ("grid", (1, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    with pytest.raises(
        TypingError,
        match="primitive_prefix.*must be inlined.*standalone callbacks",
    ):
        kernel._compile_launch_config_signature(
            types.void(types.int32[::1], types.int32[::1]), key
        )
