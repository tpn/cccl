# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass

from cuda.bindings import nvrtc

from ._caching import disk_cache
from ._common import check_in, version


def CHECK_NVRTC(err, prog):
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * logsize
        err = nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"NVRTC error: {log.decode('ascii')}")


_NVRTC_COMPILE_COUNTER = 0
_NVRTC_COMPILE_COUNTER_ENABLED = None
_NVRTC_COMPILE_RECORDS = []
_NVRTC_DUMP_COUNTER = 0
_PCH_MIN_VERSION = version(12, 8)


@dataclass(frozen=True)
class NvrtcCompileRecord:
    index: int
    cc: int
    code: str
    rdc: bool
    elapsed_ms: float
    source_bytes: int
    source_sha1: str
    compiled_source_bytes: int
    compiled_source_sha1: str
    options: tuple[str, ...]
    pch_mode: str
    pch_header_path: str | None
    pch_status: str | None
    pch_created: bool
    pch_used: bool
    pch_heap_size: int | None
    pch_heap_size_required: int | None
    log: str
    dump_source_path: str | None = None
    dump_compiled_source_path: str | None = None
    dump_log_path: str | None = None
    dump_metadata_path: str | None = None

    def to_dict(self):
        return asdict(self)


def _is_compile_counter_enabled():
    global _NVRTC_COMPILE_COUNTER_ENABLED
    if _NVRTC_COMPILE_COUNTER_ENABLED is None:
        val = os.environ.get("NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT")
        _NVRTC_COMPILE_COUNTER_ENABLED = val is not None and val.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return _NVRTC_COMPILE_COUNTER_ENABLED


def _get_dump_dir():
    dump_dir = os.environ.get("NUMBA_CCCL_COOP_NVRTC_DUMP_DIR")
    if dump_dir:
        return dump_dir
    dump_enabled = os.environ.get("NUMBA_CCCL_COOP_NVRTC_DUMP")
    if dump_enabled and dump_enabled.lower() in ("1", "true", "yes", "on"):
        return "/tmp/cccl_nvrtc"
    return None


def _get_trace_path():
    return os.environ.get("NUMBA_CCCL_COOP_NVRTC_TRACE_PATH")


def _normalize_pch_mode(mode):
    if mode is None:
        return "off"
    mode = mode.strip().lower()
    if mode in ("", "0", "false", "no", "off"):
        return "off"
    if mode in ("1", "true", "yes", "on", "auto"):
        return "auto"
    if mode in ("canonical", "rewrite"):
        return "canonical"
    if mode in ("prologue", "fixed"):
        return "prologue"
    raise ValueError(
        "Unsupported NUMBA_CCCL_COOP_NVRTC_PCH mode. "
        "Expected off, auto, canonical, or prologue."
    )


def _get_pch_mode(nvrtc_version):
    mode = _normalize_pch_mode(os.environ.get("NUMBA_CCCL_COOP_NVRTC_PCH"))
    if mode != "off" and nvrtc_version < _PCH_MIN_VERSION:
        return "off"
    return mode


def _get_pch_dir():
    dump_dir = _get_dump_dir()
    default_dir = os.path.join(tempfile.gettempdir(), "cccl_nvrtc_pch")
    if dump_dir is not None:
        default_dir = os.path.join(dump_dir, "pch")
    return os.environ.get("NUMBA_CCCL_COOP_NVRTC_PCH_DIR", default_dir)


def _get_pch_headers_override():
    raw = os.environ.get("NUMBA_CCCL_COOP_NVRTC_PCH_HEADERS")
    if raw is None:
        return None
    headers = []
    seen = set()
    for item in re.split(r"[;,]", raw):
        header = item.strip()
        if not header or header in seen:
            continue
        seen.add(header)
        headers.append(header)
    return tuple(headers)


def _get_pch_prologue_path():
    path = os.environ.get("NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH")
    if not path:
        return None
    return os.path.abspath(path)


def _get_extra_defines():
    raw = os.environ.get("NUMBA_CCCL_COOP_NVRTC_EXTRA_DEFINES")
    if raw is None:
        return ()
    defines = []
    seen = set()
    for item in re.split(r"[;,]", raw):
        define = item.strip()
        if not define or define in seen:
            continue
        seen.add(define)
        defines.append(define)
    return tuple(defines)


def _dump_source(cpp, cc, code):
    dump_dir = _get_dump_dir()
    if dump_dir is None:
        return
    os.makedirs(dump_dir, exist_ok=True)
    global _NVRTC_DUMP_COUNTER
    _NVRTC_DUMP_COUNTER += 1
    suffix = "lto" if code == "lto" else "ptx"
    stem = f"nvrtc_{_NVRTC_DUMP_COUNTER:04d}_cc{cc}_{suffix}"
    path = os.path.join(dump_dir, f"{stem}.cu")
    with open(path, "w", encoding="utf-8") as f:
        f.write(cpp)
    return {
        "dump_dir": dump_dir,
        "stem": stem,
        "source_path": path,
    }


def _dump_extra_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _split_include_preamble(cpp):
    includes = []
    body_lines = []
    in_preamble = True
    for line in cpp.splitlines(keepends=True):
        stripped = line.strip()
        if in_preamble and (not stripped or stripped.startswith("#include")):
            if stripped.startswith("#include"):
                includes.append(stripped)
            continue
        in_preamble = False
        body_lines.append(line)
    return includes, "".join(body_lines)


def _make_pch_header_line(header):
    if header.startswith("#include"):
        return header
    if header.startswith("<") or header.startswith('"'):
        return f"#include {header}"
    return f"#include <{header}>"


def _canonicalize_pch_source(cpp, pch_dir):
    includes, body = _split_include_preamble(cpp)
    headers_override = _get_pch_headers_override()
    if headers_override is None:
        header_lines = sorted(set(includes))
    else:
        header_lines = [_make_pch_header_line(header) for header in headers_override]

    if not header_lines:
        return cpp, None, None

    os.makedirs(pch_dir, exist_ok=True)
    header_text = "\n".join(header_lines) + "\n#pragma nv_hdrstop\n"
    digest = hashlib.sha1(header_text.encode("utf-8")).hexdigest()[:16]
    header_name = f"cccl_coop_pch_{digest}.h"
    header_path = os.path.join(pch_dir, header_name)
    if not os.path.exists(header_path):
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(header_text)
    rewritten = f'#include "{header_name}"\n'
    rewritten += body
    return rewritten, header_path, header_text


def _rewrite_with_pch_prologue(cpp, header_path):
    with open(header_path, "r", encoding="utf-8") as f:
        header_text = f.read()

    header_includes, header_body = _split_include_preamble(header_text)
    source_includes, source_body = _split_include_preamble(cpp)

    if source_includes and not set(source_includes).issubset(set(header_includes)):
        raise ValueError(
            "NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH did not cover the leading "
            "source includes."
        )

    header_lines = header_body.splitlines(keepends=True)
    source_lines = source_body.splitlines(keepends=True)
    header_index = 0
    matched = 0
    for source_line in source_lines:
        while (
            header_index < len(header_lines)
            and header_lines[header_index] != source_line
        ):
            header_index += 1
        if header_index == len(header_lines):
            break
        matched += 1
        header_index += 1

    if matched == 0 and not source_includes:
        raise ValueError(
            "NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH did not match the beginning "
            "of the generated source."
        )

    rewritten = f'#include "{os.path.basename(header_path)}"\n'
    rewritten += "".join(source_lines[matched:])
    return rewritten, header_path, header_text


def _get_program_log(prog):
    err, logsize = nvrtc.nvrtcGetProgramLogSize(prog)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS or logsize <= 1:
        return ""
    log = b" " * logsize
    (err,) = nvrtc.nvrtcGetProgramLog(prog, log)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return ""
    return log.decode("utf-8", errors="replace").rstrip("\0")


def _unwrap_nvrtc_result(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def _nvrtc_result_name(result):
    result = _unwrap_nvrtc_result(result)
    return getattr(result, "name", str(result))


def _collect_pch_metadata(prog, pch_mode):
    if pch_mode == "off":
        return None
    status_result = nvrtc.nvrtcGetPCHCreateStatus(prog)
    pch_status = _nvrtc_result_name(status_result)

    pch_heap_size = None
    err, size = nvrtc.nvrtcGetPCHHeapSize()
    if err == nvrtc.nvrtcResult.NVRTC_SUCCESS:
        pch_heap_size = size

    pch_heap_size_required = None
    err, size = nvrtc.nvrtcGetPCHHeapSizeRequired(prog)
    if err == nvrtc.nvrtcResult.NVRTC_SUCCESS:
        pch_heap_size_required = size

    return {
        "pch_status": pch_status,
        "pch_heap_size": pch_heap_size,
        "pch_heap_size_required": pch_heap_size_required,
    }


def _append_compile_record(record):
    _NVRTC_COMPILE_RECORDS.append(record)

    trace_path = _get_trace_path()
    if trace_path is None:
        return

    trace_dir = os.path.dirname(trace_path)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    with open(trace_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), sort_keys=True))
        f.write("\n")


def _set_compile_counter_enabled(enabled):
    global _NVRTC_COMPILE_COUNTER_ENABLED
    _NVRTC_COMPILE_COUNTER_ENABLED = enabled


def reset_compile_counter():
    global _NVRTC_COMPILE_COUNTER
    _NVRTC_COMPILE_COUNTER = 0


def get_compile_counter():
    return _NVRTC_COMPILE_COUNTER


def reset_compile_records():
    _NVRTC_COMPILE_RECORDS.clear()


def get_compile_records():
    return tuple(_NVRTC_COMPILE_RECORDS)


# cpp is the C++ source code
# cc = 800 for Ampere, 900 Hopper, etc
# rdc is true or false
# code is lto or ptx
# @cache
@functools.lru_cache(maxsize=32)  # Always enabled
@disk_cache  # Optional, see caching.py
def compile_impl(cpp, cc, rdc, code, nvrtc_path, nvrtc_version, pch_mode, pch_dir):
    dump_info = _dump_source(cpp, cc, code)
    if _is_compile_counter_enabled():
        global _NVRTC_COMPILE_COUNTER
        _NVRTC_COMPILE_COUNTER += 1
    check_in("rdc", rdc, [True, False])
    check_in("code", code, ["lto", "ptx"])
    check_in("pch_mode", pch_mode, ["off", "auto", "canonical", "prologue"])

    opts = [b"--std=c++17"]

    # TODO: move this to a module-level import (after docs env modernization).
    from cuda.cccl import get_include_paths

    include_paths = get_include_paths()
    # print(f"NVRTC include paths: {include_paths}")
    # include_paths.cub = '/home/trentn/src/cccl/cub'
    # include_paths.thrust = '/home/trentn/src/cccl/thrust'
    # include_paths.libcudacxx = '/home/trentn/src/cccl/libcudacxx'
    for path in include_paths.as_tuple():
        if path is not None:
            opts += [f"--include-path={path}".encode("ascii")]
    opts += [f"--gpu-architecture=compute_{cc}".encode("ascii")]
    if rdc:
        opts += [b"--relocatable-device-code=true"]

    if code == "lto":
        opts += [b"-dlto"]

    # Some strange linking issues
    opts += [b"-DCCCL_DISABLE_BF16_SUPPORT"]
    for define in _get_extra_defines():
        opts += [f"-D{define}".encode("ascii")]

    compiled_cpp = cpp
    pch_header_path = None
    pch_header_text = None
    if pch_mode != "off":
        os.makedirs(pch_dir, exist_ok=True)
        if pch_mode == "canonical":
            compiled_cpp, pch_header_path, pch_header_text = _canonicalize_pch_source(
                cpp, pch_dir
            )
            if pch_header_path is not None:
                opts += [f"--include-path={pch_dir}".encode("ascii")]
        elif pch_mode == "prologue":
            prologue_path = _get_pch_prologue_path()
            if prologue_path is None:
                raise ValueError(
                    "NUMBA_CCCL_COOP_NVRTC_PCH=prologue requires "
                    "NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH."
                )
            compiled_cpp, pch_header_path, pch_header_text = _rewrite_with_pch_prologue(
                cpp, prologue_path
            )
            opts += [f"--include-path={os.path.dirname(prologue_path)}".encode("ascii")]
        opts += [
            b"-pch",
            f"-pch-dir={pch_dir}".encode("ascii"),
            b"-pch-messages=true",
            b"-pch-verbose=true",
        ]

    if dump_info is not None:
        if compiled_cpp != cpp:
            dump_info["compiled_source_path"] = _dump_extra_text(
                os.path.join(dump_info["dump_dir"], f"{dump_info['stem']}_compiled.cu"),
                compiled_cpp,
            )
        if pch_header_path is not None and pch_header_text is not None:
            dump_info["pch_header_path"] = _dump_extra_text(
                os.path.join(dump_info["dump_dir"], f"{dump_info['stem']}_pch.h"),
                pch_header_text,
            )

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(
        str.encode(compiled_cpp), b"code.cu", 0, [], []
    )
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram error: {err}")

    start = time.perf_counter()
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    log = _get_program_log(prog)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"NVRTC error: {log}")

    pch_metadata = _collect_pch_metadata(prog, pch_mode) or {}
    pch_created = "creating precompiled header file" in log
    pch_used = "using precompiled header file" in log

    record = NvrtcCompileRecord(
        index=len(_NVRTC_COMPILE_RECORDS) + 1,
        cc=cc,
        code=code,
        rdc=rdc,
        elapsed_ms=elapsed_ms,
        source_bytes=len(cpp.encode("utf-8")),
        source_sha1=hashlib.sha1(cpp.encode("utf-8")).hexdigest(),
        compiled_source_bytes=len(compiled_cpp.encode("utf-8")),
        compiled_source_sha1=hashlib.sha1(compiled_cpp.encode("utf-8")).hexdigest(),
        options=tuple(opt.decode("utf-8", errors="replace") for opt in opts),
        pch_mode=pch_mode,
        pch_header_path=(
            dump_info.get("pch_header_path")
            if dump_info is not None and dump_info.get("pch_header_path") is not None
            else pch_header_path
        ),
        pch_status=pch_metadata.get("pch_status"),
        pch_created=pch_created,
        pch_used=pch_used,
        pch_heap_size=pch_metadata.get("pch_heap_size"),
        pch_heap_size_required=pch_metadata.get("pch_heap_size_required"),
        log=log,
        dump_source_path=None if dump_info is None else dump_info["source_path"],
        dump_compiled_source_path=(
            None if dump_info is None else dump_info.get("compiled_source_path")
        ),
        dump_log_path=None,
        dump_metadata_path=None,
    )

    if dump_info is not None:
        if log:
            dump_info["log_path"] = _dump_extra_text(
                os.path.join(dump_info["dump_dir"], f"{dump_info['stem']}.log"),
                log,
            )
        dump_info["metadata_path"] = os.path.join(
            dump_info["dump_dir"], f"{dump_info['stem']}.json"
        )
        record = NvrtcCompileRecord(
            **{
                **record.to_dict(),
                "dump_log_path": dump_info.get("log_path"),
                "dump_metadata_path": dump_info["metadata_path"],
            }
        )
        _dump_extra_text(
            dump_info["metadata_path"],
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
        )

    _append_compile_record(record)

    if code == "lto":
        err, ltoSize = nvrtc.nvrtcGetLTOIRSize(prog)
        CHECK_NVRTC(err, prog)

        lto = b" " * ltoSize
        (err,) = nvrtc.nvrtcGetLTOIR(prog, lto)
        CHECK_NVRTC(err, prog)

        (err,) = nvrtc.nvrtcDestroyProgram(prog)
        CHECK_NVRTC(err, prog)

        return lto

    elif code == "ptx":
        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        CHECK_NVRTC(err, prog)

        ptx = b" " * ptxSize
        (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
        CHECK_NVRTC(err, prog)

        (err,) = nvrtc.nvrtcDestroyProgram(prog)
        CHECK_NVRTC(err, prog)

        return ptx.decode("ascii")


def compile(**kwargs):
    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion error: {err}")
    nvrtc_version = version(major, minor)
    pch_mode = _get_pch_mode(nvrtc_version)
    pch_dir = _get_pch_dir()
    return nvrtc_version, compile_impl(
        **kwargs,
        nvrtc_path=nvrtc.__file__,
        nvrtc_version=nvrtc_version,
        pch_mode=pch_mode,
        pch_dir=pch_dir,
    )
