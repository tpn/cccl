import json
import os
import subprocess
import sys

import pytest
from numba import cuda


def _bootstrap_local_cccl():
    return r"""
import importlib
import os
import sys

REPO_ROOT = os.getcwd()
ROOT = os.path.join(REPO_ROOT, "python", "cuda_cccl")
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__
    not in ("_cuda_cccl_editable", "__editable___cuda_tile_9_9_99_finder")
]
sys.path[:] = [ROOT, os.path.join(ROOT, "tests", "coop")] + [
    p
    for p in sys.path
    if "tile-interop/cutile-python/simt-interop/cccl/python/cuda_cccl" not in p
    and "__editable__.cuda_tile" not in p
]
importlib.invalidate_caches()
"""


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_mamba_nvrtc_compile_count_drop():
    script = (
        _bootstrap_local_cccl()
        + r"""
import os
import sys

from cuda.coop import _nvrtc
from test_mamba_selective_scan_fwd import test_mamba_selective_scan_fwd_simple

_nvrtc.reset_compile_counter()
_nvrtc._set_compile_counter_enabled(True)

# Run the actual kernel test
_test = test_mamba_selective_scan_fwd_simple
_test("traits_gpu_dataclass")

print("__NVRTC_COUNT__=" + str(_nvrtc.get_compile_counter()))
"""
    )

    def run(bundle, dump_dir=None):
        env = os.environ.copy()
        env["NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT"] = "1"
        env["NUMBA_CCCL_COOP_BUNDLE_LTOIR"] = bundle
        if dump_dir is not None:
            env["NUMBA_CCCL_COOP_NVRTC_DUMP_DIR"] = dump_dir
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Subprocess failed (bundle={bundle}):\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        out = result.stdout + result.stderr
        count = None
        for line in out.splitlines():
            if line.startswith("__NVRTC_COUNT__="):
                count = int(line.split("=", 1)[1])
        if count is None:
            raise AssertionError(f"Missing NVRTC count in output: {out}")
        return count

    count_off = run("0")
    count_on = run("1")

    assert count_on < count_off


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_mamba_nvrtc_dump_bundle_only(tmp_path):
    script = (
        _bootstrap_local_cccl()
        + r"""
import os
import sys

from cuda.coop import _nvrtc
from test_mamba_selective_scan_fwd import test_mamba_selective_scan_fwd_simple

_nvrtc.reset_compile_counter()
_nvrtc._set_compile_counter_enabled(True)

# Run the actual kernel test
_test = test_mamba_selective_scan_fwd_simple
_test("traits_gpu_dataclass")

print("__NVRTC_COUNT__=" + str(_nvrtc.get_compile_counter()))
"""
    )

    def run(bundle, dump_dir):
        env = os.environ.copy()
        env["NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT"] = "1"
        env["NUMBA_CCCL_COOP_BUNDLE_LTOIR"] = bundle
        env["NUMBA_CCCL_COOP_NVRTC_DUMP_DIR"] = dump_dir
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Subprocess failed (bundle={bundle}):\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    dump_dir = tmp_path / "nvrtc_dump"
    dump_dir.mkdir()
    run("1", str(dump_dir))

    lto_files = [p for p in dump_dir.iterdir() if p.name.endswith("_lto.cu")]
    # gpu_dataclass bundle + kernel bundle
    assert len(lto_files) == 2


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_mamba_nvrtc_pch_canonical_reuse():
    script = (
        _bootstrap_local_cccl()
        + r"""
import json

from cuda.coop import _nvrtc
from test_mamba_selective_scan_fwd import test_mamba_selective_scan_fwd_simple

_nvrtc.compile_impl.cache_clear()
_nvrtc.reset_compile_counter()
_nvrtc.reset_compile_records()
_nvrtc._set_compile_counter_enabled(True)

test_mamba_selective_scan_fwd_simple("traits_gpu_dataclass")

records = [record.to_dict() for record in _nvrtc.get_compile_records()]
print("__NVRTC_RECORDS__=" + json.dumps(records))
"""
    )

    env = os.environ.copy()
    env["NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT"] = "1"
    env["NUMBA_CCCL_COOP_BUNDLE_LTOIR"] = "1"
    env["NUMBA_CCCL_COOP_NVRTC_PCH"] = "canonical"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    records = None
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith("__NVRTC_RECORDS__="):
            records = json.loads(line.split("=", 1)[1])
            break

    if records is None:
        raise AssertionError(
            f"Missing NVRTC records in output:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    assert any(record["pch_created"] for record in records)
    assert any(record["pch_used"] for record in records)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_mamba_nvrtc_pch_prologue_reuse_traits(tmp_path):
    root = os.path.join(os.getcwd(), "python", "cuda_cccl")
    sys.path.insert(0, root)
    from pathlib import Path

    from benchmarks.coop.bench_nvrtc_overhead import write_mamba_prologue_header

    header_path = tmp_path / "mamba_pch.h"
    write_mamba_prologue_header(Path(header_path))

    script = (
        _bootstrap_local_cccl()
        + r"""
import json

from cuda.coop import _nvrtc
from test_mamba_selective_scan_fwd import test_mamba_selective_scan_fwd_simple

_nvrtc.compile_impl.cache_clear()
_nvrtc.reset_compile_counter()
_nvrtc.reset_compile_records()
_nvrtc._set_compile_counter_enabled(True)

test_mamba_selective_scan_fwd_simple("traits_gpu_dataclass")

records = [record.to_dict() for record in _nvrtc.get_compile_records()]
print("__NVRTC_RECORDS__=" + json.dumps(records))
"""
    )

    env = os.environ.copy()
    env["NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT"] = "1"
    env["NUMBA_CCCL_COOP_BUNDLE_LTOIR"] = "1"
    env["NUMBA_CCCL_COOP_NVRTC_PCH"] = "prologue"
    env["NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH"] = str(header_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    records = None
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith("__NVRTC_RECORDS__="):
            records = json.loads(line.split("=", 1)[1])
            break

    if records is None:
        raise AssertionError(
            f"Missing NVRTC records in output:\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    assert any(record["pch_created"] for record in records)
    assert any(record["pch_used"] for record in records)
