# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What the typed pipeline refuses for load and store, and how it says so.

A refusal names the argument or the construct and is raised at compile
time; none falls back to the planner.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def _compile(kernel, signature=None, block=BLOCK):
    return testing.compile_kernel(
        kernel, signature or f32_in_out(), block=block, chip=CHIP
    )


def test_load_refuses_valid_items():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=100)
        coop.store(block, output, items)

    with pytest.raises(Exception, match=r"coop\.load: valid_items= is not"):
        _compile(kernel)


def test_load_refuses_oob_default():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, oob_default=0.0)
        coop.store(block, output, items)

    with pytest.raises(Exception, match=r"coop\.load: oob_default= is not"):
        _compile(kernel)


def test_load_refuses_offset():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, offset=128)
        coop.store(block, output, items)

    with pytest.raises(Exception, match=r"coop\.load: offset= is not"):
        _compile(kernel)


def test_store_refuses_valid_items():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items, valid_items=100)

    with pytest.raises(Exception, match=r"coop\.store: valid_items= is not"):
        _compile(kernel)


def test_store_refuses_offset():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items, offset=128)

    with pytest.raises(Exception, match=r"coop\.store: offset= is not"):
        _compile(kernel)


def test_explicit_temp_storage_is_refused():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        scratch = cuda.shared.array(4096, types.uint8)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=scratch
        )
        coop.store(block, output, items)

    with pytest.raises(
        Exception, match=r"coop\.load: explicit temp_storage= is not"
    ):
        _compile(kernel)


def test_a_group_other_than_the_block_is_refused():
    def kernel(source, output):
        items = coop.ThreadData(4)
        coop.load(1, source, items)
        coop.store(coop.this_block(), output, items)

    with pytest.raises(
        Exception, match=r"coop\.load: the group must be coop\.this_block\(\)"
    ):
        _compile(kernel)


def test_a_scalar_store_value_is_refused():
    def kernel(source, output):
        coop.store(coop.this_block(), output, source[0])

    with pytest.raises(
        Exception, match=r"coop\.store: a scalar value \(float32\) is not"
    ):
        _compile(kernel)


def test_a_plain_array_payload_has_no_typed_extent():
    def kernel(source, output):
        items = cuda.local.array(4, types.float32)
        coop.load(coop.this_block(), source, items)
        coop.store(coop.this_block(), output, items)

    with pytest.raises(Exception, match=r"has no typed extent"):
        _compile(kernel)


def test_an_unknown_algorithm_lists_the_known_ones():
    def kernel(source, output):
        items = coop.ThreadData(4)
        coop.load(coop.this_block(), source, items, algorithm="diagonal")
        coop.store(coop.this_block(), output, items)

    with pytest.raises(
        Exception, match=r"unknown algorithm 'diagonal'; use one of .*direct"
    ):
        _compile(kernel)


def test_writes_that_disagree_without_coop_evidence():
    def kernel(source, output):
        items = coop.ThreadData(4)
        base = cuda.threadIdx.x * 4
        items[0] = 2.5
        for i in range(1, 4):
            items[i] = 1
        for i in range(4):
            output[base + i] = items[i]

    with pytest.raises(
        TypedPipelineError,
        match=r"'items': its writes disagree on the dtype",
    ):
        _compile(kernel)


def test_a_read_typed_before_the_payload_dtype_was_known():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        items[0] = 1
        first = items[0]
        coop.load(block, source, items)
        output[cuda.threadIdx.x] = first

    with pytest.raises(
        TypedPipelineError,
        match=(
            r"'items': a read at line \d+ typed as float64, but the payload's "
            r"dtype is float32\. Pass dtype=float32"
        ),
    ):
        _compile(kernel)


def test_a_coop_call_in_a_separate_device_function():
    @cuda.jit(device=True, inline="never")
    def load_into(source, items):
        coop.load(coop.this_block(), source, items)

    def kernel(source, output):
        items = coop.ThreadData(4, dtype=source.dtype)
        load_into(source, items)
        coop.store(coop.this_block(), output, items)

    with pytest.raises(
        Exception,
        match=r"coop\.load at line \d+ is inside .*load_into.*, a separately "
        r"compiled device function",
    ):
        _compile(kernel)


def test_scratch_above_the_static_limit():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(8)
        coop.load(block, source, items, algorithm="transpose")
        coop.store(block, output, items)

    signature = types.void(types.float64[::1], types.float64[::1])
    with pytest.raises(
        TypedPipelineError,
        match=r"cooperative scratch needs \d+ bytes .* the 49152-byte static "
        r"limit",
    ):
        _compile(kernel, signature, block=(1024, 1, 1))


def test_scratch_counts_the_kernels_own_shared_memory():
    def kernel(source, output):
        block = coop.this_block()
        own = cuda.shared.array(48000, types.uint8)
        own[cuda.threadIdx.x] = 0
        items = coop.ThreadData(4)
        coop.load(block, source, items, algorithm="transpose")
        coop.store(block, output, items)

    with pytest.raises(
        TypedPipelineError, match=r"with the kernel's own 48000 bytes"
    ):
        _compile(kernel)


def test_launch_bounds_below_the_launch_block():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items)

    with pytest.raises(TypedPipelineError, match=r"more than launch_bounds=64"):
        testing.compile_kernel(
            kernel, f32_in_out(), block=BLOCK, chip=CHIP, launch_bounds=64
        )
