# COOP PCH Work Log

## 2026-03-18

- Request: explore NVRTC PCH support for `cuda.coop`, starting from the
  `4776-single-phase-cuda-coop-v2` branch in a new worktree, then instrument
  current NVRTC overhead, benchmark likely workloads, and capture the work in a
  ledger/report.
- Worktree:
  - created `/home/trentn/src/cccl-4776-single-phase-v2-pch`
  - branch: `4776-single-phase-cuda-coop-v2-pch`
- Research:
  - reviewed `CONTRIBUTING.md`, `ci-overview.md`, and the existing
    `SINGLE-PHASE-(NOTES|TODO|LOG).md` files for local workflow/style
  - inspected `cuda/coop/_types.py`, `cuda/coop/_rewrite/__init__.py`, and
    `cuda/coop/_nvrtc.py` to map the current source-generation, bundling, and
    LTO-IR linking path
  - verified on the live machine that NVRTC 13.1 exposes the new PCH APIs and
    that real coop-generated LTO sources can create and reuse `.pch` files
- Manual findings before implementation:
  - identical coop sources do reuse an auto-created PCH and show a large
    second-compile drop
  - Mamba bundle sources did not reuse automatically because their initial
    include order differed
  - rewriting the leading include block into a generated header terminated with
    `#pragma nv_hdrstop` was sufficient to make those two sources share one PCH
- Code changes:
  - `cuda/coop/_nvrtc.py`
    - added structured per-compile telemetry records and JSONL trace output
    - added richer dump sidecars (`*_compiled.cu`, `*_pch.h`, `.log`, `.json`)
    - added env-gated PCH modes: `off`, `auto`, `canonical`
    - implemented canonical include-header rewriting with optional header
      override support
  - `tests/coop/test_nvrtc_compile_count.py`
    - extended the NVRTC stub for trace/PCH metadata
    - added unit coverage for trace output and canonical PCH rewriting
  - `tests/coop/test_nvrtc_compile_count_gpu.py`
    - made subprocess tests bootstrap the local worktree package explicitly
    - added a GPU integration test proving canonical PCH reuse on the Mamba
      `traits_gpu_dataclass` path
  - `benchmarks/coop/bench_nvrtc_overhead.py`
    - added a dedicated benchmark driver for compile-time overhead experiments
    - runs isolated subprocesses across simple and Mamba workloads with bundle
      and PCH toggles
- Benchmark highlights:
  - repeated `block_sum_batch`: canonical PCH about `2.85x` faster than
    baseline
  - `mamba_traits_gpu_dataclass`: `bundle + canonical PCH` about `2.77x`
    faster than baseline and about `1.31x` faster than `bundle` alone
  - `mamba_single_phase_bleeding_edge_qol`: `bundle` alone remained best;
    `bundle + canonical PCH` created four distinct PCHs and reused none
  - explicit umbrella-header experiment with `cub/cub.cuh` failed under NVRTC
    because CUB rejects umbrella-header inclusion in this mode unless its
    compatibility guard is disabled
- Validation:
  - `python -m py_compile ...` on the edited Python files
  - unit tests: `tests/coop/test_nvrtc_compile_count.py` (`6 passed`)
  - GPU test: `tests/coop/test_nvrtc_compile_count_gpu.py -k pch_canonical_reuse`
    (`1 passed, 2 deselected`)
  - multi-repeat benchmark run: `benchmarks/coop/bench_nvrtc_overhead.py --repeats 3`
- Deliverables:
  - `COOP-PCH-NOTES.md`
  - `COOP-PCH-TODO.md`
  - this log

## 2026-03-18 (follow-up: umbrella guard disable + Mamba prologue)

- Request: experiment with disabling the CUB umbrella-header guard and also try
  a broader Mamba-specific PCH so more than just the raw include lines are
  cached.
- Manual umbrella-header result:
  - added `-DCCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK` and retried a simple
    `BlockReduce` shim with `#include <cub/cub.cuh>`
  - the original guard error went away, but compilation still failed in this
    environment with `error: namespace "cuda" has no member "stream_ref"`
  - conclusion: guard-disabling alone is not enough to make the umbrella header
    a viable NVRTC PCH candidate here
- Code changes:
  - `cuda/coop/_nvrtc.py`
    - added `NUMBA_CCCL_COOP_NVRTC_EXTRA_DEFINES`
    - added `NUMBA_CCCL_COOP_NVRTC_PCH=prologue`
    - added `NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH`
    - implemented fixed-prologue rewriting using the longest matching leading
      source subsequence already covered by the supplied header
  - `tests/coop/test_nvrtc_compile_count.py`
    - added unit coverage for prologue mode rewriting and extra defines
  - `benchmarks/coop/bench_nvrtc_overhead.py`
    - added `bundle_pch_mamba_prologue`
    - added benchmark-time generation of a fixed Mamba prologue header
- Manual raw NVRTC prototype:
  - proved that a fixed header containing the shared Mamba front-end state
    (load/store/scan includes, `storage_t`, `construct`/`assign`, and scan-op
    decls) can make all four residual bundled single-phase Mamba compiles reuse
    one PCH
- Benchmark follow-up:
  - `mamba_single_phase_bleeding_edge_qol` with `bundle_pch_mamba_prologue`
    measured `1619.6 ms` over 3 repeats
  - that beat:
    - baseline (`4153.7 ms`)
    - `bundle_only` (`2477.4 ms`)
    - `bundle_pch_canonical` (`3253.5 ms`)
  - effective speedups:
    - about `2.56x` vs baseline
    - about `1.54x` vs `bundle_only`
- Scope caveat:
  - the fixed Mamba prologue header did not generalize cleanly to the
    `traits_gpu_dataclass` Mamba path because that source still carried a
    conflicting `storage_t` definition after the stripped prologue
- Notes/TODO updates:
  - updated `COOP-PCH-NOTES.md` with the follow-up findings
  - updated `COOP-PCH-TODO.md` to track broader generalization work

## 2026-03-20 (stable wrapper names)

- Request: address the hard-coded wrapped-type names (`storage_t`,
  `construct`, `assign`) instead of treating the earlier conflict as a
  Mamba-only special case.
- Code changes:
  - `cuda/coop/_types.py`
    - replaced the literal `storage_t` wrapper type with a stable generated
      `cccl_storage_<hash>` name derived from the wrapped numba type
    - replaced generic `construct` / `assign` helper declarations with stable
      generated names and compiled them with explicit `abi_name`s
    - replaced the old `"storage_t"` string checks with helper predicates so
      operator/UDF plumbing works with non-generic wrapper type names
  - `cuda/coop/block/_block_merge_sort.py`
  - `cuda/coop/block/_block_radix_sort.py`
    - switched wrapper-detection checks to use `numba_type_requires_wrapper()`
  - `cuda/coop/_nvrtc.py`
    - refined fixed-prologue rewriting so include coverage is order-insensitive
      and body matching works after stripping shared include blocks
  - `tests/coop/test_nvrtc_compile_count.py`
    - added a unit test ensuring wrapper names are no longer bare
      `storage_t` / `construct` / `assign`
  - `tests/coop/test_nvrtc_compile_count_gpu.py`
    - added a GPU integration test for prologue-PCH reuse on
      `traits_gpu_dataclass`
- Validation:
  - `tests/coop/test_nvrtc_compile_count.py` now passes with 8 tests
  - `tests/coop/test_nvrtc_compile_count_gpu.py -k "pch_canonical_reuse or pch_prologue_reuse_traits"`
    passes (`2 passed, 2 deselected`)
- Benchmark follow-up (3 repeats, Mamba workloads only):
  - `mamba_single_phase_bleeding_edge_qol`
    - `bundle_pch_mamba_prologue`: `1288.8 ms`
    - `bundle_only`: `2470.0 ms`
    - `bundle_pch_canonical`: `3256.4 ms`
  - `mamba_traits_gpu_dataclass`
    - `bundle_pch_canonical`: `1110.8 ms`
    - `bundle_pch_mamba_prologue`: `1117.1 ms`
  - conclusion:
    - stable wrapper names were necessary to make the fixed prologue path work
      robustly
    - the fixed Mamba prologue is now clearly best for the bleeding-edge
      single-phase kernel
    - canonical PCH remains slightly better for the traits path
