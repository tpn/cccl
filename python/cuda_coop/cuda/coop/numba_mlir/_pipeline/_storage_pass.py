# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan each kernel's cooperative calls once, after Numba typing.

The pass runs as an after-inference rewrite, after typing and before
lowering. It sees the whole typed function at once, which is what the
planner had to reconstruct from untyped IR. For each kernel it:

* skips functions compiled as device functions (``targetoptions["device"]``);
* collects the cooperative call sites in program order;
* reads the launch configuration, only when a site needs block dimensions;
* completes each site's request, compiles all of them in one NVRTC bundle,
  and plans their scratch regions, backing and barriers;
* swaps the scratch placeholder for a constant-size ``cuda.shared.array``
  and sets ``launch_bounds``;
* stores the plan in ``state.metadata["coop_plan"]`` for lowering.

The rewrite registry is process-wide, so the pass runs for every function
Numba-CUDA-MLIR compiles once the typed pipeline is active.
"""

from __future__ import annotations

from . import _bridges as nb
from . import _debug
from ._plan import KernelPlan


class CoopStoragePass(nb.Rewrite):
    """The after-inference rewrite that owns every whole-kernel decision."""

    def __init__(self, state=None):
        super().__init__(state)
        self.state = state
        self._planned = False

    def match(self, func_ir, block, typemap, calltypes):
        if self._planned:
            return False
        self._planned = True
        targetoptions = self.state.metadata.get("targetoptions") or {}
        if targetoptions.get("device"):
            return False
        plan = KernelPlan(name=self.state.func_id.func_name)
        self.state.metadata["coop_plan"] = plan
        _debug.record(plan, func_ir, typemap)
        return False

    def apply(self):
        raise AssertionError("CoopStoragePass.match never selects a block")


def register() -> None:
    """Register the pass with the after-inference rewrite registry."""

    nb.register_after_inference_rewrite(CoopStoragePass)


__all__ = ["CoopStoragePass", "register"]
