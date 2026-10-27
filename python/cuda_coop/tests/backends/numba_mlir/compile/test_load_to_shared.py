# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate shared-copy handle ownership before generating device calls."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop
from cuda.coop.numba_mlir._compiler._group_planner_support import (
    GroupRewriteError,
)

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _compile(kernel, threads=32, source_type=None):
    return kernel._compile_launch_config_signature(
        types.void(source_type or types.int32[::1], types.int32[::1]),
        (
            ("grid", (1, 1, 1)),
            ("block", (threads, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_pending_shared_copy_alias(api):
    @cuda.jit
    def kernel(source, output):
        storage = api.TempStorage()
        shared = storage.reserve(36, types.int32, alignment=16)
        pending = api.load_to_shared_async(
            api.this_block(), source, shared, num_items=32
        )
        alias = pending
        values = alias.wait()
        output[cuda.threadIdx.x] = values[cuda.threadIdx.x]

    _compile(kernel)


@pytest.mark.parametrize("waits", [0, 2])
def test_pending_copy_requires_exactly_one_wait(waits):
    @cuda.jit
    def kernel(source, output):
        shared = cuda.shared.array(36, types.int32)
        pending = coop.load_to_shared_async(
            coop.this_block(), source, shared, num_items=32
        )
        if waits == 2:
            pending.wait()
            pending.wait()

    with pytest.raises(GroupRewriteError, match="exactly one wait"):
        _compile(kernel)


def test_pending_copy_rejects_escape():
    @cuda.jit
    def kernel(source, output):
        shared = cuda.shared.array(36, types.int32)
        pending = coop.load_to_shared_async(
            coop.this_block(), source, shared, num_items=32
        )
        output[0] = pending
        pending.wait()

    with pytest.raises(GroupRewriteError, match="only be aliased or waited"):
        _compile(kernel)


@pytest.mark.parametrize("control", ["branch", "loop"])
def test_pending_copy_rejects_runtime_control_flow(control):
    @cuda.jit
    def kernel(source, output):
        shared = cuda.shared.array(36, types.int32)
        if control == "branch":
            if source[0] != 0:
                pending = coop.load_to_shared_async(
                    coop.this_block(), source, shared, num_items=32
                )
                pending.wait()
        else:
            for _ in range(source[0]):
                pending = coop.load_to_shared_async(
                    coop.this_block(), source, shared, num_items=32
                )
                pending.wait()

    with pytest.raises(GroupRewriteError, match="kernel outer region"):
        _compile(kernel)


def test_shared_copy_rejects_global_destination():
    @cuda.jit
    def kernel(source, output):
        pending = coop.load_to_shared_async(
            coop.this_block(), source, output, num_items=32
        )
        pending.wait()

    with pytest.raises(GroupRewriteError, match="shared-memory storage"):
        _compile(kernel)


def test_incomplete_warp_rejected():
    @cuda.jit
    def kernel(source, output):
        shared = cuda.shared.array(36, types.int32)
        pending = coop.load_to_shared_async(
            coop.this_block(), source, shared, num_items=32
        )
        pending.wait()

    with pytest.raises(GroupRewriteError, match="multiple of 32"):
        _compile(kernel, threads=16)


@pytest.mark.parametrize(
    "operand", ["source_stride", "buffer_stride", "scalar", "rank2"]
)
def test_shared_copy_rejects_noncontiguous_or_nonarray_operands(operand):
    @cuda.jit
    def kernel(source, output):
        shared = cuda.shared.array(36, types.int32, alignment=16)
        if operand == "source_stride":
            memory = source[::2]
            buffer = shared
        elif operand == "buffer_stride":
            memory = source
            buffer = shared[::2]
        elif operand == "scalar":
            memory = source[0]
            buffer = shared
        else:
            memory = source
            buffer = shared
        pending = coop.load_to_shared_async(
            coop.this_block(), memory, buffer, num_items=7
        )
        pending.wait()

    source_type = types.int32[:, ::1] if operand == "rank2" else None
    with pytest.raises(GroupRewriteError, match="one-dimensional contiguous"):
        _compile(kernel, source_type=source_type)


def test_native_shared_copy_requires_alignment():
    @cuda.jit
    def kernel(source, output):
        shared = cuda.shared.array(36, types.int32)
        pending = coop.load_to_shared_async(
            coop.this_block(), source, shared, num_items=32
        )
        pending.wait()

    with pytest.raises(GroupRewriteError, match="alignment=16"):
        _compile(kernel)


@pytest.mark.parametrize("num_items", [-1, True, 2**29])
def test_shared_copy_rejects_invalid_count(num_items):
    @cuda.jit
    def kernel(source, output):
        shared = cuda.shared.array(36, types.int32, alignment=16)
        pending = coop.load_to_shared_async(
            coop.this_block(), source, shared, num_items=num_items
        )
        pending.wait()

    with pytest.raises(GroupRewriteError, match="num_items"):
        _compile(kernel)


def test_shared_copy_rejects_dtype_mismatch():
    @cuda.jit
    def kernel(source, output):
        shared = cuda.shared.array(36, types.float32, alignment=16)
        pending = coop.load_to_shared_async(
            coop.this_block(), source, shared, num_items=32
        )
        pending.wait()

    with pytest.raises(GroupRewriteError, match="same dtype"):
        _compile(kernel)
