# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend activation registers the typed pipeline by default."""

from __future__ import annotations

from ._support import run_python

_REPORT = """
import sys
import cuda.coop.numba_mlir
from numba_cuda_mlir import _whole_function_planners as wfp
from numba_cuda_mlir.numba_cuda.core.rewrites import rewrite_registry

planners = [p.__name__ for p in wfp._planner_registry._planners]
rewrites = [
    r.__name__ for r in rewrite_registry.rewrites["after-inference"]
]
backend = "cuda.coop.numba_mlir"
print("planners", planners)
print("planner-imported", f"{backend}._compiler._planner" in sys.modules)
print("pipeline-imported", f"{backend}._pipeline" in sys.modules)
print("call-resolution", "CoopCallResolution" in rewrites)
"""


def test_default_registers_call_resolution_without_a_planner():
    result = run_python(_REPORT)
    assert result.returncode == 0, result.stderr
    assert "planners []" in result.stdout
    assert "planner-imported False" in result.stdout
    assert "pipeline-imported True" in result.stdout
    assert "call-resolution True" in result.stdout
