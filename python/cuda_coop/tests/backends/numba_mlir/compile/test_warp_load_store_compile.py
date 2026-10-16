# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile common and qualified Warp Load/Store kernels."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cuda.coop.numba_mlir import _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


_FIXED_COMPUTE_CAPABILITY = (9, 0)


_BLOCK_THREADS = 64


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
    """Fix device queries while retaining real compilation and linking."""
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


def _production_launch_config_key() -> tuple[tuple[str, object], ...]:
    """Supply exact block metadata for two physical warps."""
    return (
        ("grid", (1, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_production_routes_compile_common_and_qualified_warp_kernels(
    monkeypatch: pytest.MonkeyPatch, items_per_thread
) -> None:
    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    compiler_cuda = _production_compile_environment(monkeypatch)

    def qualified_kernel(algorithm: str):
        @compiler_cuda.jit(chip="sm_90")
        def kernel(
            load_source,
            store_source,
            observed,
            destination,
            valid_items,
            offset,
            items_per_thread,
        ):
            thread = compiler_cuda.threadIdx.x
            load_payload = qualified_coop.ThreadData(
                items_per_thread, dtype=types.int32
            )
            qualified_coop.load(
                qualified_coop.this_warp(),
                load_source,
                load_payload,
                algorithm=algorithm,
                valid_items=valid_items,
                oob_default=types.int32(-17),
                offset=offset,
            )
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            for item in range(items_per_thread):
                index = thread * items_per_thread + item
                observed[index] = load_payload[item]
                payload[item] = store_source[index]
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                offset=offset,
            )

        return kernel

    def common_kernel(algorithm: str):
        @compiler_cuda.jit(chip="sm_90")
        def kernel(
            load_source,
            store_source,
            observed,
            destination,
            valid_items,
            offset,
            items_per_thread,
        ):
            thread = compiler_cuda.threadIdx.x
            load_payload = root_coop.ThreadData(
                items_per_thread, dtype=types.int32
            )
            root_coop.load(
                root_coop.this_warp(),
                load_source,
                load_payload,
                algorithm=algorithm,
                valid_items=valid_items,
                oob_default=types.int32(-17),
                offset=offset,
            )
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            for item in range(items_per_thread):
                index = thread * items_per_thread + item
                observed[index] = load_payload[item]
                payload[item] = store_source[index]
            root_coop.store(
                root_coop.this_warp(),
                destination,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                offset=offset,
            )

        return kernel

    dispatchers = [
        (qualified, algorithm, factory(algorithm))
        for qualified, factory in (
            (False, common_kernel),
            (True, qualified_kernel),
        )
        for algorithm in ("direct", "transpose")
    ]

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.int32[::1],
        types.int32[::1],
        types.int32,
        types.int64,
        types.IntegerLiteral(items_per_thread),
    )
    launch_config_key = _production_launch_config_key()
    for qualified, algorithm, dispatcher in dispatchers:
        result = dispatcher._compile_launch_config_signature(
            signature,
            launch_config_key,
        )
        assert isinstance(result.metadata["ltoir"], bytes)
        assert result.metadata["ltoir"]
        assert isinstance(result.metadata["cubin"], bytes)
        assert result.metadata["cubin"]
        assert result.metadata["linked_external_link_items"]

        ptx = next(iter(dispatcher.inspect_lto_ptx().values()))
        assert ".visible .entry" in ptx
        if algorithm == "transpose":
            assert ".shared" in ptx
        else:
            assert ".shared" not in ptx


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_production_routes_compile_common_and_qualified_logical_warp_kernels(
    monkeypatch: pytest.MonkeyPatch, items_per_thread
) -> None:
    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    compiler_cuda = _production_compile_environment(monkeypatch)

    def qualified_kernel(algorithm: str):
        @compiler_cuda.jit(chip="sm_90")
        def kernel(source, destination, valid_items, offset, items_per_thread):
            group = qualified_coop.this_warp().group_by(8)
            payload = qualified_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            qualified_coop.load(
                group,
                source,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                oob_default=types.int32(-17),
                offset=offset,
            )
            qualified_coop.store(
                group,
                destination,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                offset=offset,
            )

        return kernel

    def common_kernel(algorithm: str):
        @compiler_cuda.jit(chip="sm_90")
        def kernel(source, destination, valid_items, offset, items_per_thread):
            group = root_coop.this_warp().group_by(8)
            payload = root_coop.ThreadData(
                items_per_thread,
                dtype=types.int32,
            )
            root_coop.load(
                group,
                source,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                oob_default=types.int32(-17),
                offset=offset,
            )
            root_coop.store(
                group,
                destination,
                payload,
                algorithm=algorithm,
                valid_items=valid_items,
                offset=offset,
            )

        return kernel

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.int32,
        types.int64,
        types.IntegerLiteral(items_per_thread),
    )
    launch_config_key = _production_launch_config_key()
    for algorithm in ("direct", "transpose"):
        for factory in (common_kernel, qualified_kernel):
            dispatcher = factory(algorithm)
            result = dispatcher._compile_launch_config_signature(
                signature,
                launch_config_key,
            )
            assert result.metadata["ltoir"]
            assert result.metadata["cubin"]
            assert result.metadata["linked_external_link_items"]
            ptx = next(iter(dispatcher.inspect_lto_ptx().values()))
            assert "bar.sync" not in ptx
            if algorithm == "transpose":
                assert ".shared" in ptx
            else:
                assert ".shared" not in ptx


@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_warp_scalar_literal_store_compiles_with_destination_dtype(
    monkeypatch: pytest.MonkeyPatch,
    qualified: bool,
) -> None:
    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    compiler_cuda = _production_compile_environment(monkeypatch)
    if qualified:

        @compiler_cuda.jit(chip="sm_90")
        def kernel(destination):
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                23,
                algorithm="direct",
            )

    else:

        @compiler_cuda.jit(chip="sm_90")
        def kernel(destination):
            root_coop.store(
                root_coop.this_warp(),
                destination,
                23,
                algorithm="direct",
            )

    result = kernel._compile_launch_config_signature(
        types.void(types.int32[::1]),
        _production_launch_config_key(),
    )
    assert result.metadata["ltoir"]
    assert result.metadata["cubin"]


@pytest.mark.parametrize(
    "qualified", (False, True), ids=("common", "qualified")
)
def test_warp_scalar_runtime_expression_rejects_implicit_narrowing(
    monkeypatch: pytest.MonkeyPatch,
    qualified: bool,
) -> None:
    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as root_coop

    compiler_cuda = _production_compile_environment(monkeypatch)
    if qualified:

        @compiler_cuda.jit(chip="sm_90")
        def kernel(source, destination):
            thread = compiler_cuda.threadIdx.x
            qualified_coop.store(
                qualified_coop.this_warp(),
                destination,
                source[thread] + 1,
                algorithm="direct",
            )

    else:

        @compiler_cuda.jit(chip="sm_90")
        def kernel(source, destination):
            thread = compiler_cuda.threadIdx.x
            root_coop.store(
                root_coop.this_warp(),
                destination,
                source[thread] + 1,
                algorithm="direct",
            )

    signature = types.void(types.int32[::1], types.int32[::1])
    with pytest.raises(TypeError, match="does not match payload dtype"):
        kernel._compile_launch_config_signature(
            signature,
            _production_launch_config_key(),
        )
