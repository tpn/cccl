# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check Load/Store provider source, argument validation, and ABI metadata."""

from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


class _StringAlgorithm(str, Enum):
    DIRECT = "direct"


@pytest.fixture(autouse=True)
def _fixed_provider_compute_capability(monkeypatch):
    """Keep provider identity stable without querying a CUDA device."""
    from cuda.coop.numba_mlir import _types

    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _parameter_names(specialization):
    return [
        [type(parameter).__name__ for parameter in overload]
        for overload in specialization.parameters
    ]


def _block_provider_metadata():
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    return {
        "storage_abi": StorageABI.NONE,
        "execution_scope": SynchronizationScope.BLOCK,
        "synchronization_scope": SynchronizationScope.NONE,
    }


def test_block_load_store_adapters_preserve_offset_overloads():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import (
        make_block_load_specialization,
        make_block_store_specialization,
    )
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._lowering._load_store import (
        _load_store_value_abis,
    )

    load = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            valid_items=ArgumentBinding.runtime(),
            oob_default=ArgumentBinding.runtime(),
        )
    ).materialize(
        make_block_load_specialization(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            algorithm="direct",
            valid_items=True,
            oob_default=True,
            include_full_tile=True,
            include_pointer_offset=True,
        ),
        **_block_provider_metadata(),
    )
    assert _parameter_names(load) == [
        ["Pointer", "Array"],
        [
            "Pointer",
            "Array",
            "BoundedInteger",
            "ExactValue",
        ],
        [
            "Pointer",
            "Array",
            "BoundedInteger",
            "ExactValue",
            "PointerOffset",
        ],
        ["Pointer", "Array", "PointerOffset"],
    ]

    store = NumbaMlirCoreAdapter().materialize(
        make_block_store_specialization(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=True,
        ),
        **_block_provider_metadata(),
    )
    assert _parameter_names(store) == [
        ["Pointer", "Array"],
        ["Pointer", "Array", "PointerOffset"],
    ]


def test_static_movement_controls_are_absent_from_numba_runtime_abi():
    pytest.importorskip("numba_cuda_mlir")
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import (
        make_block_load_specialization,
        make_block_store_specialization,
    )
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import PointerOffset, algo_coalesce_key

    adapter = NumbaMlirCoreAdapter()
    load = adapter.materialize(
        make_block_load_specialization(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            valid_items=ArgumentBinding.static(17),
            oob_default=ArgumentBinding.static(-1),
            include_pointer_offset=ArgumentBinding.static(3),
        ),
        **_block_provider_metadata(),
    )
    assert len(load.parameters) == 1
    method = load.parameters[0]
    offset = method[-1]
    assert isinstance(offset, PointerOffset)
    assert offset.static_value == 3

    source = load._source_code()[0]
    assert ".Load((src + 3)," in source
    assert "::cuda::std::int64_t offset" not in source
    assert ", 17, -1);" in source

    static_four = adapter.materialize(
        make_block_store_specialization(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=ArgumentBinding.static(4),
        ),
        **_block_provider_metadata(),
    )
    static_five = adapter.materialize(
        make_block_store_specialization(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=ArgumentBinding.static(5),
        ),
        **_block_provider_metadata(),
    )
    assert static_four.mangled_name(static_four.parameters[0]) != (
        static_five.mangled_name(static_five.parameters[0])
    )
    assert algo_coalesce_key(static_four) != algo_coalesce_key(static_five)

    runtime = adapter.materialize(
        make_block_store_specialization(
            dtype=types.int32,
            block_dim=(32, 1, 1),
            items_per_thread=2,
            algorithm="direct",
            include_pointer_offset=ArgumentBinding.runtime(),
        ),
        **_block_provider_metadata(),
    )
    runtime_source = runtime._source_code()[0]
    assert "::cuda::std::int64_t offset" in runtime_source
    assert ".Store((dst + offset)," in runtime_source


def test_runtime_valid_items_is_checked_before_cub_integer_narrowing():
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop._core.block import make_block_load_specialization
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._lowering._load_store import (
        _load_store_value_abis,
    )
    from cuda.coop.numba_mlir._types import BoundedInteger, algo_coalesce_key

    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            valid_items=ArgumentBinding.runtime(),
        )
    )
    load = adapter.materialize(
        make_block_load_specialization(
            dtype=types.int32,
            block_dim=(16, 2, 1),
            items_per_thread=3,
            algorithm="direct",
            valid_items=ArgumentBinding.runtime(),
        ),
        **_block_provider_metadata(),
    )
    valid_items = load.parameters[0][-1]
    assert isinstance(valid_items, BoundedInteger)
    assert valid_items.dtype() == types.int64
    assert valid_items.provider_dtype == types.int32
    assert (valid_items.minimum, valid_items.maximum) == (0, 96)

    source = load._source_code()[0]
    declaration = "::cuda::std::int64_t num_valid_items"
    guard = "if (num_valid_items < 0 || num_valid_items > 96)"
    trap = 'asm volatile("trap;" : : :);'
    narrowing = (
        "::cuda::std::int32_t checked_num_valid_items = "
        "static_cast<::cuda::std::int32_t>(num_valid_items);"
    )
    invocation = ".Load(src, *reinterpret_cast<"
    assert source.index(declaration) < source.index(guard)
    assert source.index(guard) < source.index(trap) < source.index(narrowing)
    assert source.index(narrowing) < source.index(invocation)
    assert "checked_num_valid_items);" in source

    narrower_adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=types.int32,
            block_dim=(16, 1, 1),
            items_per_thread=3,
            valid_items=ArgumentBinding.runtime(),
        )
    )
    narrower_tile = narrower_adapter.materialize(
        make_block_load_specialization(
            dtype=types.int32,
            block_dim=(16, 1, 1),
            items_per_thread=3,
            algorithm="direct",
            valid_items=ArgumentBinding.runtime(),
        ),
        **_block_provider_metadata(),
    )
    assert algo_coalesce_key(load) != algo_coalesce_key(narrower_tile)
    assert load.mangled_name(load.parameters[0]) != narrower_tile.mangled_name(
        narrower_tile.parameters[0]
    )


def test_arbitrary_primitive_declares_bounded_and_exact_value_abis():
    from numba_cuda_mlir import types

    from cuda.coop._core import (
        FLOAT32,
        INT32,
        Algorithm,
        SynchronizationScope,
        Value,
    )
    from cuda.coop.numba_mlir._compiler._operations import StorageABI
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import (
        BoundedInteger,
        ExactValue,
        algo_coalesce_key,
    )
    from cuda.coop.numba_mlir._types import (
        Value as BackendValue,
    )

    specialization = Algorithm(
        struct_name="FuturePrimitive",
        method_name="Run",
        c_name="test_future_primitive",
        includes=(),
        template_parameters=(),
        parameters=(
            (
                Value(INT32, name="remaining"),
                Value(FLOAT32, name="fill"),
            ),
        ),
        template_arguments={},
        metadata={"primitive": "not_load_or_store"},
    )
    bounded = BoundedInteger(types.int32, minimum=-2, maximum=7)
    exact = ExactValue(types.float32)
    algorithm = NumbaMlirCoreAdapter(
        value_abis={"remaining": bounded, "fill": exact}
    ).materialize(
        specialization,
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )

    lowered_bounded, lowered_exact = algorithm.parameters[0]
    assert lowered_bounded is not bounded
    assert lowered_exact is not exact
    assert lowered_bounded.parameter_name == "remaining"
    assert lowered_exact.parameter_name == "fill"
    assert bounded.provider_dtype == types.int32
    assert bounded.dtype() == types.int64

    source = algorithm._source_code()[0]
    guard = "if (remaining < -2 || remaining > 7)"
    trap = 'asm volatile("trap;" : : :);'
    narrowing = (
        "::cuda::std::int32_t checked_remaining = "
        "static_cast<::cuda::std::int32_t>(remaining);"
    )
    assert source.index(guard) < source.index(trap) < source.index(narrowing)
    assert "checked_remaining, fill);" in source

    wider = NumbaMlirCoreAdapter(
        value_abis={
            "remaining": BoundedInteger(types.int32, minimum=-2, maximum=8),
            "fill": exact,
        }
    ).materialize(
        specialization,
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )
    assert algo_coalesce_key(algorithm) != algo_coalesce_key(wider)
    assert algorithm.mangled_name(
        algorithm.parameters[0]
    ) != wider.mangled_name(wider.parameters[0])

    load_like = Algorithm(
        struct_name="LegacyLikePrimitive",
        method_name="Run",
        c_name="test_load_like_primitive",
        includes=(),
        template_parameters=(),
        parameters=(
            (
                Value(INT32, name="num_valid_items"),
                Value(FLOAT32, name="oob_default"),
            ),
        ),
        template_arguments={},
        metadata={"primitive": "load"},
    )
    without_declarations = NumbaMlirCoreAdapter().materialize(
        load_like,
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.NONE,
        synchronization_scope=SynchronizationScope.NONE,
    )
    assert all(
        type(parameter) is BackendValue
        for parameter in without_declarations.parameters[0]
    )


def test_value_abi_declarations_fail_closed_during_materialization():
    from numba_cuda_mlir import types

    from cuda.coop._core import (
        INT32,
        Algorithm,
        Pointer,
        SynchronizationScope,
        Value,
    )
    from cuda.coop.numba_mlir._compiler._operations import StorageABI
    from cuda.coop.numba_mlir._lowering._core import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import ExactValue

    def specialization(parameter):
        return Algorithm(
            struct_name="FuturePrimitive",
            method_name="Run",
            c_name="test_value_abi_validation",
            includes=(),
            template_parameters=(),
            parameters=((parameter,),),
            template_arguments={},
        )

    def materialize(parameter, value_abis):
        return NumbaMlirCoreAdapter(value_abis=value_abis).materialize(
            specialization(parameter),
            storage_abi=StorageABI.NONE,
            execution_scope=SynchronizationScope.NONE,
            synchronization_scope=SynchronizationScope.NONE,
        )

    with pytest.raises(ValueError, match="unknown .* value ABI"):
        materialize(
            Value(INT32, name="value"), {"missing": ExactValue(types.int32)}
        )
    with pytest.raises(ValueError, match="require scalar Value parameters"):
        materialize(
            Pointer(INT32, name="value"),
            {"value": ExactValue(types.int32)},
        )
    with pytest.raises(ValueError, match="provider dtypes do not match"):
        materialize(
            Value(INT32, name="value"),
            {"value": ExactValue(types.float32)},
        )
    with pytest.raises(ValueError, match="output roles do not match"):
        materialize(
            Value(INT32, name="value", is_output=True),
            {"value": ExactValue(types.int32)},
        )
    with pytest.raises(TypeError, match="backend Value instances"):
        materialize(Value(INT32, name="value"), {"value": object()})


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize(
    "algorithm",
    [True, 0, "stripd", "warp_transpose", object()],
)
def test_invalid_physical_warp_algorithm_fails_before_materialization(
    monkeypatch, operation, algorithm
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "invalid Warp algorithm reached provider materialization"
        ),
    )
    factory = getattr(_load_store, f"warp_{operation}")
    error_type = TypeError if not isinstance(algorithm, str) else ValueError

    with pytest.raises(error_type):
        factory(
            types.int32,
            threads_in_warp=32,
            threads_per_block=(64, 1, 1),
            algorithm=algorithm,
        )


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("threads", [0, 3, 31, 64, True, 32.0])
def test_warp_provider_rejects_invalid_width_before_materialization(
    monkeypatch, operation, threads
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "invalid Warp width reached provider materialization"
        ),
    )
    factory = getattr(_load_store, f"warp_{operation}")

    with pytest.raises(
        (TypeError, ValueError),
        match=r"logical width|integer|positive",
    ):
        factory(
            types.int32,
            threads_in_warp=threads,
            threads_per_block=(64, 1, 1),
        )


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize(
    ("algorithm", "error_type", "message"),
    [
        (True, TypeError, "algorithm must be a string"),
        (0, TypeError, "algorithm must be a string"),
        (_StringAlgorithm.DIRECT, TypeError, "algorithm must be a string"),
        ("stripd", ValueError, "expected one of"),
        (object(), TypeError, "algorithm must be a string"),
    ],
)
def test_invalid_block_algorithm_values_fail_before_provider_materialization(
    monkeypatch, operation, algorithm, error_type, message
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "invalid algorithm reached provider materialization"
        ),
    )
    factory = getattr(_load_store, operation)
    with pytest.raises(error_type, match=message):
        factory(types.int32, threads_per_block=32, algorithm=algorithm)


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("dtype_name", ["boolean", "float16", "complex64"])
def test_unsupported_dtypes_fail_before_provider_materialization(
    monkeypatch, operation, dtype_name
):
    from numba_cuda_mlir import types

    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "unsupported dtype reached provider materialization"
        ),
    )
    factory = getattr(_load_store, operation)

    with pytest.raises(TypeError, match="supports dtypes"):
        factory(getattr(types, dtype_name), threads_per_block=32)


@pytest.mark.parametrize(
    "oob_default",
    [
        pytest.param(True, id="bool"),
        pytest.param(np.float16(1), id="float16"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param(object(), id="object"),
        pytest.param(float("inf"), id="nonfinite"),
        pytest.param(1 << 65, id="outside-64-bit"),
    ],
)
def test_direct_provider_rejects_invalid_static_oob_default_before_materialization(  # noqa: E501 - Preserve descriptive test name.
    monkeypatch,
    oob_default,
):
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "invalid oob_default reached provider materialization"
        ),
    )

    with pytest.raises((TypeError, ValueError), match="oob_default"):
        _load_store.load(
            types.int32,
            threads_per_block=32,
            num_valid_items=ArgumentBinding.static(31),
            oob_default=ArgumentBinding.static(oob_default),
        )


@pytest.mark.parametrize("operation", ["load", "store"])
def test_static_valid_items_beyond_the_exact_tile_fail_before_nvrtc(
    monkeypatch, operation
):
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "invalid tile reached materialization"
        ),
    )
    factory = getattr(_load_store, operation)

    with pytest.raises(ValueError, match="block tile size"):
        factory(
            types.int32,
            threads_per_block=32,
            items_per_thread=2,
            num_valid_items=ArgumentBinding.static(65),
        )


@pytest.mark.parametrize("operation", ["load", "store"])
def test_negative_static_offsets_fail_before_provider_materialization(
    monkeypatch, operation
):
    from numba_cuda_mlir import types

    from cuda.coop._core import ArgumentBinding
    from cuda.coop.numba_mlir._lowering import _load_store

    monkeypatch.setattr(
        _load_store.NumbaMlirCoreAdapter,
        "materialize",
        lambda *args, **kwargs: pytest.fail(
            "negative offset reached provider materialization"
        ),
    )
    factory = getattr(_load_store, operation)

    with pytest.raises(ValueError, match="offset must be nonnegative"):
        factory(
            types.int32,
            threads_per_block=32,
            offset=ArgumentBinding.static(-1),
        )
