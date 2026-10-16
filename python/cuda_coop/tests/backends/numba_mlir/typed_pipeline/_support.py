# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared helpers for the typed-pipeline tests.

The pipeline is registered when ``cuda.coop.numba_mlir`` activates.
Activation tests start a fresh interpreter per case.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Compile-only tests pin the target and stand in for the launch's block.
CHIP = "sm_120"
BLOCK = (128, 1, 1)

ALGORITHMS = (
    "direct",
    "striped",
    "vectorize",
    "transpose",
    "warp_transpose",
    "warp_transpose_timesliced",
)
STORAGE_FREE = ALGORITHMS[:3]
TRANSPOSING = ALGORITHMS[3:]


def f32_in_out():
    """The signature ``(float32[::1], float32[::1])``."""

    from numba_cuda_mlir import types

    return types.void(types.float32[::1], types.float32[::1])


def run_python(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter."""

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
