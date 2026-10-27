# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile the two calls sharing one CUB asynchronous-copy context."""

from cuda.coop._core import SynchronizationScope
from cuda.coop._core.block.load_to_shared import (
    LOAD_TO_SHARED_INCLUDES,
    load_to_shared_cpp_type,
    load_to_shared_type_definition,
    make_block_load_to_shared_specialization,
)

from .._compiler import _nvrtc
from .._compiler._operations import StorageABI
from .._types import _current_compile_target, make_invocable_from_specialization
from ._core import NumbaMlirCoreAdapter


def make_shared_copy_invocables(dtype, block_dim, num_items, compile_context):
    """Return issue/wait providers and their actual C++ context layouts.

    The context resides in per-thread storage across both calls. Its shared
    barrier has a separate allocation that ordinary primitives cannot reuse.
    Query both layouts from the same headers and target as the providers;
    never approximate the private CUB object's size or reconstruct it at wait.
    """

    adapter = NumbaMlirCoreAdapter()
    cpp_type = load_to_shared_cpp_type(
        adapter.cpp_type(dtype), block_dim, num_items
    )
    source = "\n".join(
        [f"#include <{header}>" for header in LOAD_TO_SHARED_INCLUDES]
        + [load_to_shared_type_definition().code]
    )
    _, (_, layouts) = _nvrtc.compile_with_layouts(
        cpp=source,
        layout_types=(f"{cpp_type}::Context", f"{cpp_type}::TempStorage"),
        code="lto",
        rdc=True,
        context=compile_context,
        **_current_compile_target().nvrtc_kwargs(),
    )
    invocables = []
    for phase in ("issue", "wait"):
        core = make_block_load_to_shared_specialization(
            dtype=adapter.core_dtype(dtype),
            block_dim=block_dim,
            num_items=num_items,
            phase=phase,
        )
        specialization = adapter.materialize(
            core,
            storage_abi=StorageABI.NONE,
            execution_scope=SynchronizationScope.BLOCK,
            synchronization_scope=SynchronizationScope.NONE,
        )
        invocables.append(make_invocable_from_specialization(specialization))
    return (*invocables, *layouts)
