# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve each kernel's cooperative calls once, after Numba typing.

Typing gives each cooperative call a request that knows nothing of the
launch. ``CoopCallResolution`` turns every request into a concrete provider
call, and lowering only emits it. The resolution runs as one
after-inference rewrite, after typing and before lowering, and sees the
whole typed function at once. It runs four stages in order, each on the
previous stages' results:

1. site analysis (``_sites``) checks the typed IR and builds each site's
   request;
2. specialization planning (``_specialize``) reads the launch's block, only
   when there are sites, and specializes the providers;
3. storage analysis (``_storage``) compiles them in one bundle and lays out
   their scratch;
4. storage materialization (``_materialize``) rewrites the IR.

Only the last stage changes the IR, so a failure in any other leaves the
function as typing left it. The driver applies the ``launch_bounds`` that
stage 2 chose, stores what lowering needs in ``state.metadata["coop_plan"]``
and reports the ``KernelPlan`` to ``_debug.record``.

The rewrite registry is process-wide, so the resolution runs for every
function Numba-CUDA-MLIR compiles once the typed pipeline is active.
"""

from __future__ import annotations

from typing import Any

from . import _bridges as nb
from . import _debug, _materialize, _sites, _specialize, _storage
from ._materialize import CallEntry, CoopPlan
from ._plan import KernelPlan, SitePlan


def _describe(binding: Any) -> str:
    if binding.kind.value == "runtime":
        return "runtime"
    value = binding.value
    return str(value.item() if hasattr(value, "item") else value)


def _site_plan(index, site, block, region, barrier_after) -> SitePlan:
    request = site.request
    value = ()
    if site.scalar is not None:
        value = (
            (
                "value",
                "runtime"
                if site.scalar.value is None
                else f"{site.scalar.value.item()!r}",
            ),
        )
    return SitePlan(
        index=index,
        operation=site.kind,
        line=site.line,
        request=(
            ("algorithm", request.algorithm),
            ("dtype", str(request.dtype)),
            ("items_per_thread", request.items_per_thread),
            ("block", "x".join(map(str, block))),
            *value,
            *((name, _describe(binding)) for name, binding in request.controls),
        ),
        region=region,
        barrier_after=barrier_after,
    )


def _compile_target(targetoptions: dict) -> tuple[Any, Any, Any]:
    """Return the kernel's NVRTC target, compile context and identity."""

    from .._compiler import _nvrtc

    target = _nvrtc.CompileTarget.from_targetoptions(targetoptions)
    if target is None:
        target = _nvrtc.CompileTarget.from_targetoptions(
            {
                **targetoptions,
                "chip": nb.resolve_gpu_target(targetoptions)["chip"],
            }
        )
    context = _nvrtc.resolve_compile_context()
    identity = _nvrtc.compiler_identity(
        context=context, rdc=True, code="lto", **target.nvrtc_kwargs()
    )
    return target, context, identity


def _resolve_sites(
    state, func_ir, typemap, calltypes, sites, plan, targetoptions
) -> dict[str, CallEntry]:
    launch = nb.require_launch_config(state)
    block = tuple(launch["block"]) + (1,) * (3 - len(launch["block"]))
    target, context, identity = _compile_target(targetoptions)

    specialization = _specialize.specialize(
        sites,
        block,
        target=target,
        context=context,
        targetoptions=targetoptions,
    )
    storage, bundle = _storage.analyze(
        specialization, sites.user_shared_bytes, identity, context
    )
    bounds = specialization.launch_bounds
    if bounds.rebind:
        nb.rebind_targetoptions(state, launch_bounds=bounds.value)
    calls = _materialize.materialize(
        func_ir,
        typemap,
        calltypes,
        state.typingctx,
        sites.sites,
        specialization,
        storage,
        bundle,
    )

    plan.block, plan.block_source = block, "launch"
    plan.launch_bounds = bounds.value
    plan.bundle = bundle.specializations
    plan.backing = storage.backing
    if storage.region is not None:
        plan.regions.append(storage.region)
    for index, (site, scratch, barrier) in enumerate(
        zip(
            sites.sites,
            storage.scratch,
            specialization.barrier_after,
            strict=True,
        )
    ):
        region = storage.region.name if scratch else None
        plan.sites.append(_site_plan(index, site, block, region, barrier))
    return calls


def resolve(state: Any, func_ir: Any, typemap: dict, calltypes: Any) -> None:
    """Resolve every cooperative call of the function ``state`` compiles."""

    targetoptions = state.metadata.get("targetoptions") or {}
    sites = _sites.analyze(
        func_ir,
        typemap,
        calltypes,
        args=state.args,
        targetoptions=targetoptions,
        func_id=state.func_id,
    )
    if sites is None:
        return
    plan = KernelPlan(name=state.func_id.func_name)
    coop_plan = CoopPlan(plan)
    if sites.sites:
        coop_plan.calls = _resolve_sites(
            state, func_ir, typemap, calltypes, sites, plan, targetoptions
        )
    state.metadata["coop_plan"] = coop_plan
    _debug.record(plan, func_ir, typemap)


class CoopCallResolution(nb.Rewrite):
    """The after-inference rewrite that owns every whole-kernel decision.

    All of the work happens in the first ``match``, which rewrites the
    function in place and always returns ``False``.
    """

    def __init__(self, state=None):
        super().__init__(state)
        self.state = state
        self._resolved = False

    def match(self, func_ir, block, typemap, calltypes):
        if self._resolved or "coop_plan" in self.state.metadata:
            return False
        self._resolved = True
        resolve(self.state, func_ir, typemap, calltypes)
        return False

    def apply(self):
        raise AssertionError("CoopCallResolution.match never selects a block")


def register() -> None:
    """Register the resolution with the after-inference rewrite registry."""

    nb.register_after_inference_rewrite(CoopCallResolution)


__all__ = ["CoopCallResolution", "register", "resolve"]
