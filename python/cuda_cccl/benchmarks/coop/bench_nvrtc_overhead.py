# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

RESULT_PREFIX = "__COOP_NVRTC_RESULT__="


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def package_root() -> Path:
    return repo_root() / "python" / "cuda_cccl"


def bootstrap_local_cccl(root: Path) -> None:
    import importlib

    package = root / "python" / "cuda_cccl"
    tests = package / "tests" / "coop"
    blocked_substrings = (
        "tile-interop/cutile-python/simt-interop/cccl/python/cuda_cccl",
        "__editable__.cuda_tile",
    )
    blocked_meta_path = {
        "_cuda_cccl_editable",
        "__editable___cuda_tile_9_9_99_finder",
    }

    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ not in blocked_meta_path
    ]
    sys.path[:] = [str(package), str(tests)] + [
        path
        for path in sys.path
        if not any(token in path for token in blocked_substrings)
    ]
    importlib.invalidate_caches()


def run_block_sum_batch():
    import numpy as np
    from numba import cuda

    from cuda import coop

    thread_counts = [32, 64, 128, 256, 512, 1024]
    for threads in thread_counts:

        @cuda.jit
        def kernel(d_in, d_out):
            total = coop.block.sum(
                d_in[cuda.threadIdx.x],
                items_per_thread=1,
                dim=threads,
            )
            if cuda.threadIdx.x == 0:
                d_out[0] = total

        d_in = cuda.to_device(np.ones(threads, dtype=np.int32))
        d_out = cuda.device_array(1, dtype=np.int32)
        kernel[1, threads](d_in, d_out)

    cuda.synchronize()


def run_mamba_variant(kernel_variant: str):
    from test_mamba_selective_scan_fwd import test_mamba_selective_scan_fwd_simple

    test_mamba_selective_scan_fwd_simple(kernel_variant)


WORKLOADS = {
    "block_sum_batch": run_block_sum_batch,
    "mamba_traits_gpu_dataclass": lambda: run_mamba_variant("traits_gpu_dataclass"),
    "mamba_single_phase_bleeding_edge_qol": lambda: run_mamba_variant(
        "single_phase_bleeding_edge_qol"
    ),
}

CONFIGS = [
    {
        "name": "baseline",
        "bundle": "0",
        "pch": "off",
        "pch_headers": None,
        "description": "Bundle off, PCH off",
    },
    {
        "name": "bundle_only",
        "bundle": "1",
        "pch": "off",
        "pch_headers": None,
        "description": "Bundle on, PCH off",
    },
    {
        "name": "pch_canonical_only",
        "bundle": "0",
        "pch": "canonical",
        "pch_headers": None,
        "description": "Bundle off, canonical include-PCH",
    },
    {
        "name": "bundle_pch_canonical",
        "bundle": "1",
        "pch": "canonical",
        "pch_headers": None,
        "description": "Bundle on, canonical include-PCH",
    },
    {
        "name": "bundle_pch_mamba_prologue",
        "bundle": "1",
        "pch": "prologue",
        "pch_headers": None,
        "pch_prologue": "mamba",
        "workloads": (
            "mamba_single_phase_bleeding_edge_qol",
            "mamba_traits_gpu_dataclass",
        ),
        "description": "Bundle on, fixed Mamba prologue PCH",
    },
]


def worker_main(args: argparse.Namespace) -> int:
    bootstrap_local_cccl(Path(args.repo_root))

    from numba import cuda

    from cuda.coop import _nvrtc

    _nvrtc.compile_impl.cache_clear()
    _nvrtc.reset_compile_counter()
    _nvrtc.reset_compile_records()
    _nvrtc._set_compile_counter_enabled(True)

    WORKLOADS[args.worker]()

    records = [record.to_dict() for record in _nvrtc.get_compile_records()]
    result = {
        "workload": args.worker,
        "compile_count": len(records),
        "attempted_compile_count": _nvrtc.get_compile_counter(),
        "total_compile_ms": round(
            sum(record["elapsed_ms"] for record in records),
            3,
        ),
        "pch_created_count": sum(1 for record in records if record["pch_created"]),
        "pch_used_count": sum(1 for record in records if record["pch_used"]),
        "device_name": str(cuda.get_current_device()),
        "records": records,
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0


def parse_worker_result(output: str):
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f"Worker did not emit a result payload:\n{output}")


def write_mamba_prologue_header(path: Path) -> None:
    bootstrap_local_cccl(repo_root())

    from mamba_selective_scan_fwd import float2_type

    from cuda.coop._types import numba_type_to_wrapper

    wrapper = numba_type_to_wrapper(float2_type, methods=float2_type.methods)
    path.write_text(
        "#include <cuda/std/cstdint>\n"
        "#include <cub/block/block_load.cuh>\n"
        "#include <cub/block/block_scan.cuh>\n"
        "#include <cub/block/block_store.cuh>\n" + wrapper.code + "\n"
        'extern "C" __device__ void Fssm_scan_op_Float2__Float2_Float2('
        "void*, const void*, const void*);\n"
        'extern "C" __device__ void F__call___Float2__Float2('
        "char *state, void*, const void*);\n"
        "#pragma nv_hdrstop\n",
        encoding="utf-8",
    )


def run_one(workload: str, config: dict[str, object], script_path: Path):
    env = os.environ.copy()
    env["CCCL_REPO_ROOT"] = str(repo_root())
    env["NUMBA_CCCL_COOP_BUNDLE_LTOIR"] = str(config["bundle"])
    env["NUMBA_CCCL_COOP_NVRTC_PCH"] = str(config["pch"])
    if config["pch_headers"] is None:
        env.pop("NUMBA_CCCL_COOP_NVRTC_PCH_HEADERS", None)
    else:
        env["NUMBA_CCCL_COOP_NVRTC_PCH_HEADERS"] = str(config["pch_headers"])

    with tempfile.TemporaryDirectory(prefix="cccl-coop-pch-") as tempdir:
        tempdir_path = Path(tempdir)
        if config.get("pch_prologue") == "mamba":
            header_path = tempdir_path / "mamba_pch.h"
            write_mamba_prologue_header(header_path)
            env["NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH"] = str(header_path)
        else:
            env.pop("NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH", None)

        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--worker",
                workload,
                "--repo-root",
                str(repo_root()),
            ],
            env=env,
            cwd=str(repo_root()),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Worker failed for workload={workload}, config={config['name']}.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return parse_worker_result(proc.stdout + proc.stderr)


def summarize_runs(raw_runs):
    grouped = {}
    for entry in raw_runs:
        key = (entry["workload"], entry["config"]["name"])
        grouped.setdefault(key, []).append(entry["result"])

    summary = []
    for (workload, config_name), results in grouped.items():
        compile_counts = [result["compile_count"] for result in results]
        total_compile_ms = [result["total_compile_ms"] for result in results]
        pch_created = [result["pch_created_count"] for result in results]
        pch_used = [result["pch_used_count"] for result in results]

        summary.append(
            {
                "workload": workload,
                "config_name": config_name,
                "runs": len(results),
                "mean_compile_count": statistics.mean(compile_counts),
                "mean_total_compile_ms": statistics.mean(total_compile_ms),
                "stdev_total_compile_ms": (
                    statistics.stdev(total_compile_ms)
                    if len(total_compile_ms) > 1
                    else 0.0
                ),
                "mean_pch_created_count": statistics.mean(pch_created),
                "mean_pch_used_count": statistics.mean(pch_used),
            }
        )
    return summary


def render_bar(value: float, baseline: float, width: int = 28) -> str:
    if baseline <= 0:
        return ""
    ratio = min(value / baseline, 1.0)
    filled = max(1, round(ratio * width))
    return "#" * filled + " " * (width - filled)


def render_markdown(raw_runs, summary_rows, repeats: int):
    config_map = {config["name"]: config for config in CONFIGS}
    device_name = raw_runs[0]["result"]["device_name"] if raw_runs else "unknown"
    lines = [
        "# CUDA Coop NVRTC Overhead Benchmark",
        "",
        f"- Device: `{device_name}`",
        f"- Repeats per configuration: `{repeats}`",
        "- Metrics: cache-miss NVRTC compile count and summed NVRTC compile wall time",
        "",
        "## Configurations",
        "",
    ]
    for config in CONFIGS:
        lines.append(f"- `{config['name']}`: {config['description']}")

    workloads = sorted({row["workload"] for row in summary_rows})
    for workload in workloads:
        rows = [row for row in summary_rows if row["workload"] == workload]
        rows.sort(key=lambda row: row["config_name"])
        baseline = next(
            row["mean_total_compile_ms"]
            for row in rows
            if row["config_name"] == "baseline"
        )

        lines.extend(
            [
                "",
                f"## {workload}",
                "",
                "| config | mean compile count | mean compile ms | stdev ms | speedup vs baseline | mean pch used |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            speedup = baseline / row["mean_total_compile_ms"]
            lines.append(
                "| "
                f"{row['config_name']} | "
                f"{row['mean_compile_count']:.1f} | "
                f"{row['mean_total_compile_ms']:.1f} | "
                f"{row['stdev_total_compile_ms']:.1f} | "
                f"{speedup:.2f}x | "
                f"{row['mean_pch_used_count']:.1f} |"
            )

        lines.extend(["", "```text"])
        for row in rows:
            bar = render_bar(row["mean_total_compile_ms"], baseline)
            percent = (row["mean_total_compile_ms"] / baseline) * 100.0
            lines.append(
                f"{row['config_name']:<24} {percent:6.1f}% |{bar}| "
                f"{row['mean_total_compile_ms']:.1f} ms"
            )
        lines.append("```")

        best = min(rows, key=lambda row: row["mean_total_compile_ms"])
        best_cfg = config_map[best["config_name"]]
        lines.extend(
            [
                "",
                f"Best config: `{best['config_name']}` "
                f"({best_cfg['description']}, {baseline / best['mean_total_compile_ms']:.2f}x vs baseline).",
            ]
        )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=sorted(WORKLOADS), default=None)
    parser.add_argument("--repo-root", default=str(repo_root()))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--workloads",
        nargs="*",
        choices=sorted(WORKLOADS),
        default=sorted(WORKLOADS),
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args(argv)

    if args.worker is not None:
        return worker_main(args)

    script_path = Path(__file__).resolve()
    raw_runs = []
    for workload in args.workloads:
        for config in CONFIGS:
            allowed_workloads = config.get("workloads")
            if allowed_workloads is not None and workload not in allowed_workloads:
                continue
            for repeat_idx in range(args.repeats):
                print(
                    f"[bench] workload={workload} config={config['name']} "
                    f"repeat={repeat_idx + 1}/{args.repeats}",
                    file=sys.stderr,
                )
                result = run_one(workload, config, script_path)
                raw_runs.append(
                    {
                        "workload": workload,
                        "config": config,
                        "repeat": repeat_idx,
                        "result": result,
                    }
                )

    summary_rows = summarize_runs(raw_runs)
    markdown = render_markdown(raw_runs, summary_rows, args.repeats)
    if args.output_json is not None:
        Path(args.output_json).write_text(
            json.dumps(
                {
                    "raw_runs": raw_runs,
                    "summary": summary_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.output_markdown is not None:
        Path(args.output_markdown).write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
