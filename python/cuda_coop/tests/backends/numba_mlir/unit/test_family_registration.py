# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check core family registration and provider synchronization metadata."""

from dataclasses import dataclass

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


@dataclass(frozen=True)
class _FakeSemantics:
    """Provide semantic identity for a temporary operation family."""

    token: str

    @property
    def semantic_key(self):
        return ("fake-family", self.token)

    @property
    def result_visibility(self):
        from cuda.coop._core import ResultVisibility

        return ResultVisibility.PER_MEMBER

    @property
    def returns_value(self):
        return True


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        (
            {"classifications": None},
            TypeError,
            "classifications must be callable",
        ),
        ({"planner": None}, TypeError, "planner must be callable"),
        (
            {"group_kinds": frozenset()},
            ValueError,
            "group_kinds must not be empty",
        ),
        (
            {"group_kinds": frozenset({"not_a_group"})},
            ValueError,
            "group_kinds contains unsupported values",
        ),
        (
            {"unsupported_group_message": " "},
            ValueError,
            "unsupported_group_message must be a non-empty string",
        ),
    ],
)
def test_core_family_registration_rejects_invalid_contracts(
    override,
    error,
    message,
):
    from cuda.coop._core.group import _dispatch

    arguments = {
        "classifications": lambda operation: (),
        "planner": lambda call, group, launch, operation: object(),
        "group_kinds": frozenset({"block"}),
        "unsupported_group_message": "test family requires a block",
    }
    arguments.update(override)

    with pytest.raises(error, match=message):
        _dispatch._register_group_operation_family(_FakeSemantics, **arguments)


def test_factory_registration_rejects_noncallable_provider():
    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir._compiler._operations import (
        StorageABI,
        register_factory,
    )

    with pytest.raises(TypeError, match="lowering factory must be callable"):
        register_factory(
            object(),
            operation="_test_invalid_factory",
            namespace="test",
            storage_abi=StorageABI.NONE,
            execution_scope=SynchronizationScope.NONE,
            synchronization_scope=SynchronizationScope.NONE,
        )


def test_storage_free_provider_uses_default_constructor_and_zero_storage():
    from numba_cuda_mlir import types

    from cuda.coop._core import SynchronizationScope
    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler._nvrtc import CompilerIdentity
    from cuda.coop.numba_mlir._compiler._operations import StorageABI

    algorithm = _types.Algorithm(
        struct_name="StorageFreeProvider",
        method_name="Run",
        c_name="test_storage_free_provider",
        includes=(),
        template_parameters=(),
        parameters=((_types.Value(types.int32),),),
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.NONE,
        template_arguments={},
    )
    source, _, storage_symbols, _ = algorithm._source_code(
        compile_identity=CompilerIdentity(
            cc=90, rdc=True, code="lto", compiler_options=()
        )
    )

    assert "StorageFreeProvider().Run" not in source
    assert "algorithm_t_" in source
    assert "().Run(param_0);" in source
    assert "TempStorage" not in source
    assert "temp_storage" not in source
    assert storage_symbols == ()
