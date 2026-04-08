---
# vim:set tw=78
title: "CUDA Coop NVRTC PCH Investigation"
categories:
  - cuda.coop
  - nvrtc
author: "Codex"
date: 03/18/2026
description: |
    Investigation notes and benchmark results for applying NVRTC precompiled
    headers to cuda.coop's on-the-fly LTO-IR C-shim compilation path.
format:
  html:
    css: extend.css
    code-annotations: select
    code-line-numbers: true
    grid:
      gutter-width: 1rem
---

# Goal

Evaluate whether NVRTC's CUDA 12.8+ precompiled header (PCH) support can reduce
`cuda.coop`'s JIT overhead, especially for:

1. repeated small single-phase kernels that each compile a simple one-shot
   primitive, and
2. larger Mamba-style kernels that exercise multiple cooperative primitives and
   already benefit from batched NVRTC LTO bundling.

# Executive Summary

- Real NVRTC PCH support works with our `code="lto"` path on CUDA 13.1.
- The low-risk path is not a hard-coded umbrella header. It is a canonicalized
  generated header built from the source's leading `#include` block, terminated
  with `#pragma nv_hdrstop`, and compiled behind an env-gated experimental path.
- For repeated simple kernels, canonical PCH is a clear win: about `2.85x`
  faster than baseline in the batch benchmark below.
- For the current `traits_gpu_dataclass` Mamba path, `bundle + canonical PCH`
  is also a clear win: about `2.82x` faster than baseline and about `1.34x`
  faster than `bundle` alone.
- For the current bleeding-edge single-phase Mamba kernel, a fixed Mamba
  prologue PCH beats `bundle` alone: about `3.20x` faster than baseline and
  about `1.92x` faster than `bundle` alone.
- After stabilizing the generated wrapper type and helper names, the fixed
  Mamba prologue path also works on `traits_gpu_dataclass`, but canonical PCH
  still edges it out there.
- The umbrella candidate `#include <cub/cub.cuh>` is currently blocked by CUB's
  NVRTC compatibility guard unless `CCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK`
  is also defined, and even with that guard disabled it still failed in this
  environment because `cuda::stream_ref` was unavailable in the umbrella include
  path. I did not promote that path to the default experiment matrix.

# Current cuda.coop NVRTC/LTO-IR Flow

## Source generation

The on-the-fly C-shim path lives primarily in
`cuda/coop/_types.py` and `cuda/coop/_rewrite/__init__.py`.

At a high level:

1. A coop primitive specialization materializes C++ source with:
   - leading `#include` directives for the relevant CUB headers,
   - optional type-wrapper definitions for user-defined types,
   - optional UDF forward declarations,
   - algorithm typedefs and exported size/alignment constants,
   - an `extern "C" __device__` shim that wraps the CUB call.
2. `cuda.coop._nvrtc.compile(..., code="lto")` compiles that source to LTO-IR.
3. The resulting `numba.cuda.LTOIR` object is attached to the current kernel's
   code library for device linking.

For example, a simple `coop.block.sum(..., items_per_thread=1, dim=128)` kernel
produces a small shim like:

```cpp
#include <cuda/std/cstdint>
#include <cub/block/block_reduce.cuh>

using block_reduce_0_t = cub::BlockReduce<...>;
using block_reduce_0_temp_storage_t = typename block_reduce_0_t::TempStorage;

extern "C" __device__ void block_reduce_0(int& src, int& result) {
    __shared__ block_reduce_0_temp_storage_t temp_storage;
    result = block_reduce_0_t(temp_storage).Sum(src);
    __syncthreads();
}
```

## Bundling

The existing bundling path is already important context for PCH:

- `CoopNodeRewriter.ensure_ltoir_bundle()` collects all algorithms seen during
  lowering.
- `prepare_ltoir_bundle()` deduplicates compatible algorithms, merges their
  includes, type definitions, and UDF declarations, and emits a single bundled
  LTO compilation unit.
- `gpu_dataclass()` also uses bundled compilation when it needs temp-storage
  sizing metadata.

That means the baseline PCH question is not "can PCH replace bundling?" It is
"does PCH still help the NVRTC compiles that remain after bundling?"

# What Changed In This Worktree

## New NVRTC telemetry

`cuda/coop/_nvrtc.py` now records one structured record per cache-miss compile:

- elapsed wall time in milliseconds,
- source and rewritten-source byte counts and SHA1s,
- full NVRTC option list,
- PCH mode, PCH create status, heap sizes,
- whether the compile created or reused a PCH,
- optional dump/log/metadata paths.

New helpers:

- `_nvrtc.reset_compile_records()`
- `_nvrtc.get_compile_records()`

New persistent trace hook:

- `NUMBA_CCCL_COOP_NVRTC_TRACE_PATH=/path/to/trace.jsonl`

## Improved dump artifacts

The old dump behavior only wrote the original generated `.cu` source.

When dumps are enabled now:

- `*_lto.cu` or `*_ptx.cu`: original generated source,
- `*_compiled.cu`: rewritten source actually handed to NVRTC, if different,
- `*_pch.h`: generated canonical PCH header, if used,
- `*.log`: NVRTC program log,
- `*.json`: metadata summary for that compile.

Existing toggles still work:

- `NUMBA_CCCL_COOP_NVRTC_DUMP=1`
- `NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/path/to/dir`

## Experimental PCH toggles

The PCH path is env-gated and default-off:

- `NUMBA_CCCL_COOP_NVRTC_PCH=off|auto|canonical|prologue`
- `NUMBA_CCCL_COOP_NVRTC_PCH_DIR=/tmp/cccl_nvrtc_pch`
- `NUMBA_CCCL_COOP_NVRTC_PCH_HEADERS=header1;header2;...`
- `NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH=/path/to/header.h`
- `NUMBA_CCCL_COOP_NVRTC_EXTRA_DEFINES=NAME1;NAME2[=VALUE]`

Modes:

- `auto`: use NVRTC PCH against the source as generated.
- `canonical`: extract the leading include block, emit a generated header with
  those includes plus `#pragma nv_hdrstop`, replace the source preamble with a
  single `#include "cccl_coop_pch_<hash>.h"`, and let NVRTC auto-create/reuse
  the resulting `.pch`.
- `prologue`: reuse a caller-supplied header file as a fixed PCH prologue and
  strip the longest matching leading source block already covered by that
  header. This is the mode used for the Mamba-specific follow-up experiment.

`canonical` is the useful mode for coop because it normalizes otherwise noisy
include ordering between generated sources.

# Benchmark Setup

Environment used for the data below:

- GPU: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- compute capability: `sm_120`
- NVRTC: `13.1`
- repeats per configuration: `3`

Bench harness:

`python/cuda_cccl/benchmarks/coop/bench_nvrtc_overhead.py`

Workloads:

1. `block_sum_batch`
   - six simple single-phase kernels
   - each kernel uses exactly one primitive: `coop.block.sum`
   - dimensions: `32, 64, 128, 256, 512, 1024`
2. `mamba_single_phase_bleeding_edge_qol`
   - the existing bleeding-edge single-phase Mamba test kernel
3. `mamba_traits_gpu_dataclass`
   - the existing trait-structured Mamba path

Configurations:

- `baseline`: bundling off, PCH off
- `bundle_only`: bundling on, PCH off
- `pch_canonical_only`: bundling off, canonical PCH on
- `bundle_pch_canonical`: bundling on, canonical PCH on
- `bundle_pch_mamba_prologue`: bundling on, fixed Mamba prologue PCH

# Results

## block_sum_batch

| config | mean compile count | mean compile ms | stdev ms | speedup vs baseline | mean pch used |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 6.0 | 3945.2 | 7.8 | 1.00x | 0.0 |
| bundle_only | 6.0 | 3959.0 | 17.2 | 1.00x | 0.0 |
| bundle_pch_canonical | 6.0 | 1384.5 | 6.8 | 2.85x | 5.0 |
| pch_canonical_only | 6.0 | 1397.6 | 9.6 | 2.82x | 5.0 |

```text
baseline                  100.0% |############################| 3945.2 ms
bundle_only               100.3% |############################| 3959.0 ms
bundle_pch_canonical       35.1% |##########                  | 1384.5 ms
pch_canonical_only         35.4% |##########                  | 1397.6 ms
```

Interpretation:

- Bundling is irrelevant here because each kernel has only one primitive.
- Canonical PCH works exactly as hoped: one PCH create, five PCH reuses.
- The simple repeated-kernel case is the cleanest win for PCH.

## mamba_single_phase_bleeding_edge_qol

| config | mean compile count | mean compile ms | stdev ms | speedup vs baseline | mean pch used |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 7.0 | 4130.4 | 7.6 | 1.00x | 0.0 |
| bundle_only | 4.0 | 2470.0 | 5.4 | 1.67x | 0.0 |
| bundle_pch_canonical | 4.0 | 3256.4 | 16.3 | 1.27x | 0.0 |
| bundle_pch_mamba_prologue | 4.0 | 1288.8 | 6.3 | 3.20x | 3.0 |
| pch_canonical_only | 7.0 | 2593.4 | 9.1 | 1.59x | 4.0 |

```text
baseline                  100.0% |############################| 4130.4 ms
bundle_only                59.8% |#################           | 2470.0 ms
bundle_pch_canonical       78.8% |######################      | 3256.4 ms
bundle_pch_mamba_prologue   31.2% |#########                   | 1288.8 ms
pch_canonical_only         62.8% |##################          | 2593.4 ms
```

Interpretation:

- Canonical PCH helps the non-bundled path significantly:
  - three PCH creates,
  - four PCH reuses.
- But `bundle + canonical PCH` is worse than `bundle` alone because the four
  remaining compiles all ended up with distinct canonical header hashes:
  - `cccl_coop_pch_56e31023524f81ae.h`
  - `cccl_coop_pch_b96c4065c49ed7f5.h`
  - `cccl_coop_pch_eea0b8360d5f9ad2.h`
  - `cccl_coop_pch_63b9ccc4949a3f58.h`
- Result: four PCH creations, zero reuses. In that shape, PCH adds cost instead
  of removing it.
- A fixed Mamba prologue PCH worked much better:
  - one PCH create,
  - three PCH reuses,
  - about `1.92x` faster than `bundle_only`.
- The reason it wins is that the prologue header deliberately precompiles the
  shared Mamba front-end state:
  - block load/store/scan includes,
  - `storage_t`,
  - `construct` / `assign`,
  - scan-op and prefix-callback forward declarations.

## mamba_traits_gpu_dataclass

| config | mean compile count | mean compile ms | stdev ms | speedup vs baseline | mean pch used |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 5.0 | 3132.5 | 5.0 | 1.00x | 0.0 |
| bundle_only | 2.0 | 1482.8 | 3.1 | 2.11x | 0.0 |
| bundle_pch_canonical | 2.0 | 1110.8 | 5.6 | 2.82x | 1.0 |
| bundle_pch_mamba_prologue | 2.0 | 1117.1 | 13.5 | 2.80x | 1.0 |
| pch_canonical_only | 5.0 | 3353.4 | 1.4 | 0.93x | 1.0 |

```text
baseline                  100.0% |############################| 3132.5 ms
bundle_only                47.3% |#############               | 1482.8 ms
bundle_pch_canonical       35.5% |##########                  | 1110.8 ms
bundle_pch_mamba_prologue   35.7% |##########                  | 1117.1 ms
pch_canonical_only        107.1% |############################| 3353.4 ms
```

Interpretation:

- Bundling already removes most of the redundant NVRTC work here.
- The remaining two bundled compiles share the same canonical include header, so
  `bundle + canonical PCH` gives the best result:
  - one PCH create,
  - one PCH reuse.
- After the stable wrapper-name change, the fixed Mamba prologue header also
  works on this path, but it is slightly slower than the lighter canonical PCH
  because there are only two residual bundled compiles to amortize.
- PCH without bundling loses badly because it still pays for too many distinct
  sources.

# The `cub/cub.cuh` Umbrella Candidate

The repo's umbrella header is `cub/cub.cuh` rather than `cub/cub.h`.

I tried the direct umbrella-header experiment via:

```text
NUMBA_CCCL_COOP_NVRTC_PCH=canonical
NUMBA_CCCL_COOP_NVRTC_PCH_HEADERS=cuda/std/cstdint;cub/cub.cuh
```

That failed immediately under NVRTC with CUB's guard:

```text
#error "Including <cub/cub.cuh> is not supported when compiling with NVRTC.
Include the specific device header instead ..."
```

Even after explicitly adding
`-DCCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK`, the umbrella path still failed
for a simple `BlockReduce` shim in this environment with:

```text
error: namespace "cuda" has no member "stream_ref"
```

So the umbrella path is not currently a drop-in experiment here.

This is one reason the canonical "extract actual generated includes" strategy is
the safer first experiment for coop.

# Conclusions

## What looks promising

- Keep bundling as the primary structural optimization for multi-primitive
  kernels.
- Keep canonical PCH as the general-purpose experimental second-layer
  optimization, especially
  for:
  - repeated single-primitive kernel compiles,
  - `gpu_dataclass` + bundled-kernel paths where the same bundled header set is
    compiled more than once.
- For specific high-value kernels like the bleeding-edge single-phase Mamba
  kernel, a fixed workload-specific prologue PCH can outperform both canonical
  PCH and bundle-only runs.

## What does not look ready

- A blanket "always enable PCH" switch.
- A hard-coded `cub/cub.cuh` umbrella-header PCH path.
- `bundle + canonical PCH` for every workload shape. The bleeding-edge Mamba
  single-phase path shows that bundling can reduce compile count enough that the
  remaining compiles are too heterogeneous for source-local canonical PCH reuse
  to pay off.

## Recommended direction

1. Keep the new PCH path default-off.
2. Use the new telemetry to identify workloads where bundled or non-bundled
   sources actually share a canonical header hash.
3. For kernels with a stable shared front-end state, consider a fixed
   workload-specific prologue header instead of source-local include extraction.
4. Treat `gpu_dataclass`, repeated single-primitive kernels, and the current
   bleeding-edge single-phase Mamba kernel as the best near-term PCH targets.

# Reproduction

Benchmark:

```bash
python python/cuda_cccl/benchmarks/coop/bench_nvrtc_overhead.py \
  --repeats 3 \
  --output-json /tmp/coop_pch_bench_full.json \
  --output-markdown /tmp/coop_pch_bench_full.md

python python/cuda_cccl/benchmarks/coop/bench_nvrtc_overhead.py \
  --repeats 3 \
  --workloads mamba_single_phase_bleeding_edge_qol \
  --output-json /tmp/coop_pch_mamba_prologue_full.json \
  --output-markdown /tmp/coop_pch_mamba_prologue_full.md
```

Useful env toggles for ad-hoc exploration:

```bash
NUMBA_CCCL_COOP_BUNDLE_LTOIR=1
NUMBA_CCCL_COOP_NVRTC_PCH=canonical
NUMBA_CCCL_COOP_NVRTC_TRACE_PATH=/tmp/coop_nvrtc_trace.jsonl
NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/tmp/coop_nvrtc_dump

NUMBA_CCCL_COOP_NVRTC_PCH=prologue
NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH=/tmp/mamba_pch.h
NUMBA_CCCL_COOP_NVRTC_EXTRA_DEFINES=CCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK
```

GPU validation performed for this work:

```bash
python -m py_compile \
  python/cuda_cccl/cuda/coop/_nvrtc.py \
  python/cuda_cccl/tests/coop/test_nvrtc_compile_count.py \
  python/cuda_cccl/tests/coop/test_nvrtc_compile_count_gpu.py \
  python/cuda_cccl/benchmarks/coop/bench_nvrtc_overhead.py

python - <<'PY'
import importlib
import sys
import pytest
ROOT = '/home/trentn/src/cccl-4776-single-phase-v2-pch/python/cuda_cccl'
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ not in ('_cuda_cccl_editable', '__editable___cuda_tile_9_9_99_finder')
]
sys.path[:] = [ROOT, ROOT + '/tests/coop'] + [
    p for p in sys.path
    if 'tile-interop/cutile-python/simt-interop/cccl/python/cuda_cccl' not in p
    and '__editable__.cuda_tile' not in p
]
importlib.invalidate_caches()
raise SystemExit(pytest.main(['-q', 'python/cuda_cccl/tests/coop/test_nvrtc_compile_count.py']))
PY

python - <<'PY'
import importlib
import sys
import pytest
ROOT = '/home/trentn/src/cccl-4776-single-phase-v2-pch/python/cuda_cccl'
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ not in ('_cuda_cccl_editable', '__editable___cuda_tile_9_9_99_finder')
]
sys.path[:] = [ROOT, ROOT + '/tests/coop'] + [
    p for p in sys.path
    if 'tile-interop/cutile-python/simt-interop/cccl/python/cuda_cccl' not in p
    and '__editable__.cuda_tile' not in p
]
importlib.invalidate_caches()
raise SystemExit(pytest.main([
    '-q',
    'python/cuda_cccl/tests/coop/test_nvrtc_compile_count_gpu.py',
    '-k',
    'pch_canonical_reuse',
]))
PY
```
