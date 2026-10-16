# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile Exchange and Shuffle kernels with CUDA devices hidden."""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

import numba_cuda_mlir.tools as numba_mlir_tools
from numba_cuda_mlir import cuda, types

from cuda.coop.numba_mlir import _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


_FIXED_COMPUTE_CAPABILITY = (9, 0)


_BLOCK_THREADS = 64


_LOGICAL_WARP_WIDTHS = (1, 2, 4, 8, 16, 32)


@pytest.fixture(autouse=True)
def _fixed_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int]]:
    """Hide runtime discovery while leaving NVRTC and nvJitLink real."""

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "the Numba-CUDA-MLIR compile stage must hide all CUDA devices"
    )
    queries: list[tuple[int, int]] = []

    def current_device() -> SimpleNamespace:
        queries.append(_FIXED_COMPUTE_CAPABILITY)
        return SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    monkeypatch.setattr(_types.cuda, "get_current_device", current_device)
    return queries


def _production_compile_environment(monkeypatch: pytest.MonkeyPatch):
    """Fix architecture discovery for complete kernel compilation.

    Only the device queries are replaced. Kernel rewriting, provider
    compilation, and linking still use their production implementations.
    """

    import numba_cuda_mlir.tools as numba_mlir_tools
    from numba_cuda_mlir import cuda as compiler_cuda

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
    return compiler_cuda


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_production_kernel_compile_links_shared_storage_and_barriers(
    monkeypatch: pytest.MonkeyPatch, items_per_thread
) -> None:
    compiler_cuda = _production_compile_environment(monkeypatch)

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as common_coop

    @compiler_cuda.jit(chip="sm_90")
    def kernel(source, destination, distance, items_per_thread):
        thread = compiler_cuda.threadIdx.x
        payload = qualified_coop.ThreadData(items_per_thread, dtype=types.int32)
        for item in range(items_per_thread):
            payload[item] = source[thread * items_per_thread + item]
        exchanged = common_coop.exchange(
            common_coop.this_block(),
            payload,
            mode="blocked_to_striped",
        )
        shifted = common_coop.shuffle(
            common_coop.this_block(),
            exchanged,
            mode="up",
        )
        rotated = qualified_coop.shuffle(
            qualified_coop.this_block(),
            source[thread],
            mode="rotate",
            distance=distance,
        )
        for item in range(items_per_thread):
            destination[thread * items_per_thread + item] = (
                shifted[item] + rotated if item == 0 else shifted[item]
            )

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.int32,
        types.IntegerLiteral(items_per_thread),
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


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_untyped_load_composes_directly_into_exchange(
    monkeypatch: pytest.MonkeyPatch, items_per_thread
) -> None:
    compiler_cuda = _production_compile_environment(monkeypatch)

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as common_coop

    @compiler_cuda.jit(chip="sm_90")
    def kernel(source, destination, items_per_thread):
        thread = compiler_cuda.threadIdx.x
        payload = qualified_coop.ThreadData(items_per_thread)
        common_coop.load(
            common_coop.this_block(),
            source,
            payload,
            algorithm="direct",
        )
        exchanged = common_coop.exchange(
            common_coop.this_block(),
            payload,
            mode="blocked_to_striped",
        )
        for item in range(items_per_thread):
            destination[thread * items_per_thread + item] = exchanged[item]

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.IntegerLiteral(items_per_thread),
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
    assert result.metadata["ltoir"]
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]


def _evaluate_warp_mask(definitions, operand, rank):
    """Interpret the emitted barrier-mask arithmetic for one thread rank.

    The test compares this value with an independent set of expected lanes.
    Only operations used by the mask are supported; reject unfamiliar ones
    so a changed expression cannot silently evade the check. Integer casts
    must retain their signed or unsigned bit behavior.
    """
    expression = definitions[operand]
    operation = expression.split()[0]
    inputs = re.findall(r"%[\w-]+", expression)
    if operation == "arith.constant":
        return int(expression.split()[1])
    if operation == "gpu.thread_id":
        return rank if expression.split()[1] == "x" else 0
    if operation == "gpu.block_dim":
        return _BLOCK_THREADS if expression.split()[1] == "x" else 1
    values = [_evaluate_warp_mask(definitions, value, rank) for value in inputs]
    if operation in {"arith.index_cast", "arith.extui"}:
        return values[0]
    if operation == "arith.trunci":
        assert expression.endswith("to i32")
        return values[0] & 0xFFFFFFFF
    if operation == "arith.extsi":
        assert len(values) == 1 and expression.endswith(": i32 to i64"), (
            expression
        )
        value = values[0] & 0xFFFFFFFF
        return value - (1 << 32) if value & (1 << 31) else value
    assert (
        operation
        in {
            "arith.addi",
            "arith.muli",
            "arith.andi",
            "arith.floordivsi",
            "arith.shli",
        }
        and len(values) == 2
    ), f"unexpected warp-mask operation: {expression}"
    left, right = values
    if operation == "arith.addi":
        return left + right
    if operation == "arith.muli":
        return left * right
    if operation == "arith.andi":
        return left & right
    if operation == "arith.floordivsi":
        return left // right
    if operation == "arith.shli":
        return left << right
    raise AssertionError(f"unexpected warp-mask operation: {expression}")


@pytest.mark.parametrize("width", _LOGICAL_WARP_WIDTHS)
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_production_warp_exchange_emits_ordered_reuse_barriers(
    width, monkeypatch, items_per_thread
):
    import cuda.coop.numba_mlir as coop

    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )

    @cuda.jit(chip="sm_90")
    def kernel(source, destination, items_per_thread):
        thread = cuda.threadIdx.x
        payload = coop.ThreadData(items_per_thread, dtype=types.int32)
        for item in range(items_per_thread):
            payload[item] = source[thread * items_per_thread + item]
        first = coop.exchange(
            coop.this_warp().group_by(width),
            payload,
            mode="blocked_to_striped",
        )
        second = coop.exchange(
            coop.this_warp().group_by(width), first, mode="blocked_to_striped"
        )
        for item in range(items_per_thread):
            destination[thread * items_per_thread + item] = second[item]

    launch_key = (
        ("grid", (1, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        types.void(
            types.int32[::1],
            types.int32[::1],
            types.IntegerLiteral(items_per_thread),
        ),
        launch_key,
    )
    assert result.metadata["cubin"]
    mlir = result.metadata["mlir_module_str"]
    definitions = dict(
        re.findall(r"^\s*(%[\w-]+) = (.*)$", mlir, flags=re.MULTILINE)
    )
    operands = re.findall(r"nvvm.bar.warp.sync\s+(%[\w-]+)", mlir)
    assert len(operands) == 2
    assert "gpu.barrier" not in mlir
    events = [
        "call" if "func.call" in line else "barrier"
        for line in mlir.splitlines()
        if ("func.call" in line and "BlockedToStriped" in line)
        or "nvvm.bar.warp.sync" in line
    ]
    assert events == ["call", "barrier", "call", "barrier"]
    for rank in range(_BLOCK_THREADS):
        group_start = (rank % 32 // width) * width
        expected = sum(
            1 << lane for lane in range(group_start, group_start + width)
        )
        for operand in operands:
            assert (
                _evaluate_warp_mask(definitions, operand, rank) & 0xFFFFFFFF
                == expected
            )
