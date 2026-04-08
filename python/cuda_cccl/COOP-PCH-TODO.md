# COOP PCH TODO

- [x] Create a dedicated PCH exploration worktree from `4776-single-phase-cuda-coop-v2`.
- [x] Map the current coop NVRTC -> LTO-IR -> linker path and existing bundle behavior.
- [x] Add per-compile NVRTC telemetry suitable for bundle/PCH comparisons.
- [x] Extend NVRTC dumps so rewritten sources, logs, metadata, and generated PCH
      headers can be inspected easily.
- [x] Add an env-gated experimental PCH mode with canonical include-header
      rewriting.
- [x] Add a fixed prologue PCH mode for workload-specific experiments.
- [x] Replace hard-coded wrapped-type helper names (`storage_t`,
      `construct`, `assign`) with stable generated names.
- [x] Add targeted unit coverage for telemetry and canonical PCH rewriting.
- [x] Add a GPU integration test that proves canonical PCH reuse on the Mamba
      `traits_gpu_dataclass` path.
- [x] Benchmark a repeated simple single-primitive workload and two Mamba-style
      workloads with and without bundling/PCH.
- [x] Write up findings and recommendations in `COOP-PCH-NOTES.md`.
- [ ] Decide whether the experimental PCH path should stay in-tree as a hidden
      debug knob or move to a narrower prototype branch after review.
- [ ] If the umbrella-header idea remains interesting, try a follow-up
      experiment with `CCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK` and any
      additional missing NVRTC dependencies needed to make `cub/cub.cuh`
      compile; guard-disabling alone still failed on `cuda::stream_ref`.
- [ ] Generalize the fixed-prologue approach so it can be synthesized safely
      from source shape instead of being hard-coded for the current Mamba
      single-phase kernel.
- [ ] Explore a smarter header-bucketing strategy for bundled multi-primitive
      paths where source-local canonical headers remain too fragmented.
