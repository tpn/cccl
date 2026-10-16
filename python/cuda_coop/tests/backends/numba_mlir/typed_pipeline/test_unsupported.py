# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A cooperative primitive the typed pipeline lacks is a typing error.

It never falls back to the planner, and it never surfaces as Numba's generic
"untyped global name" message.
"""

from __future__ import annotations

import pytest

# Primitives outside the typed pipeline's first supported surface.
_REJECTED = [
    ("cuda.coop", "scan"),
    ("cuda.coop", "sum"),
    ("cuda.coop", "TempStorage"),
    ("cuda.coop", "this_warp"),
    ("cuda.coop.numba_mlir", "radix_sort_keys"),
    ("cuda.coop.numba_mlir", "exchange"),
    ("cuda.coop.numba_mlir", "TempStorage"),
    ("cuda.coop.numba_mlir", "this_grid"),
]


@pytest.mark.parametrize("module_name, name", _REJECTED)
def test_rejected_primitive_names_itself(module_name, name):
    import importlib

    from numba_cuda_mlir import compiler, cuda, types
    from numba_cuda_mlir.numba_cuda.core.errors import TypingError

    primitive = getattr(importlib.import_module(module_name), name)

    @cuda.jit
    def kernel(x):
        primitive(x)

    with pytest.raises(TypingError) as info:
        compiler.compile_mlir(kernel, types.void(types.float32[::1]))
    message = str(info.value)
    assert f"{module_name}.{name} is not supported on the typed pipeline" in (
        message
    )


def test_every_kernel_callable_primitive_has_a_template():
    from numba_cuda_mlir.extending import typing_registry

    from cuda.coop.numba_mlir._pipeline._templates import kernel_primitives

    primitives = kernel_primitives()
    names = set(primitives.values())
    assert {"cuda.coop.load", "cuda.coop.numba_mlir.store"} <= names
    assert "cuda.coop.register" not in names
    registered = {key for key, _ in typing_registry.globals}
    missing = [n for obj, n in primitives.items() if obj not in registered]
    assert not missing
