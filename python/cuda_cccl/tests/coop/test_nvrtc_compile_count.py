import json
import types

from cuda.coop import _nvrtc, _types


class _DummyNvrtc:
    __file__ = "dummy_nvrtc"
    last_source = None
    last_filename = None
    last_options = ()

    class nvrtcResult:
        NVRTC_SUCCESS = 0
        NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13
        NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14
        NVRTC_ERROR_PCH_CREATE = 15

    @classmethod
    def reset(cls):
        cls.last_source = None
        cls.last_filename = None
        cls.last_options = ()

    @staticmethod
    def nvrtcVersion():
        return (0, 13, 1)

    @staticmethod
    def nvrtcCreateProgram(src, name, *args, **kwargs):
        _DummyNvrtc.last_source = src.decode("utf-8")
        _DummyNvrtc.last_filename = name.decode("utf-8")
        return (0, object())

    @staticmethod
    def nvrtcCompileProgram(prog, num_opts, opts):
        _DummyNvrtc.last_options = tuple(
            opt.decode("utf-8") if isinstance(opt, bytes) else opt for opt in opts
        )
        return (0,)

    @staticmethod
    def nvrtcGetLTOIRSize(*args, **kwargs):
        return (0, 1)

    @staticmethod
    def nvrtcGetLTOIR(*args, **kwargs):
        return (0,)

    @staticmethod
    def nvrtcGetPTXSize(*args, **kwargs):
        return (0, 1)

    @staticmethod
    def nvrtcGetPTX(*args, **kwargs):
        return (0,)

    @staticmethod
    def nvrtcDestroyProgram(*args, **kwargs):
        return (0,)

    @staticmethod
    def nvrtcGetProgramLogSize(*args, **kwargs):
        return (0, 0)

    @staticmethod
    def nvrtcGetProgramLog(*args, **kwargs):
        return (0,)

    @staticmethod
    def nvrtcGetPCHCreateStatus(*args, **kwargs):
        return (_DummyNvrtc.nvrtcResult.NVRTC_SUCCESS,)

    @staticmethod
    def nvrtcGetPCHHeapSize(*args, **kwargs):
        return (0, 4096)

    @staticmethod
    def nvrtcGetPCHHeapSizeRequired(*args, **kwargs):
        return (0, 8192)


class _DummyDevice:
    compute_capability = (8, 0)


class _DummyObjectCode:
    def __init__(self, name):
        self.name = name

    @classmethod
    def from_ltoir(cls, blob, name):
        return cls(name)


class _DummyLinker:
    def __init__(self, obj, options=None):
        self.obj = obj
        self.options = options

    def link(self, kind):
        return types.SimpleNamespace(code=self._ptx.encode("utf-8"))


class _DummyLinkerOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyPrimitive:
    def __init__(self):
        self.is_child = False


class _DummyAlgo:
    def __init__(self, name_prefix):
        self.includes = []
        self.type_definitions = []
        self.parameters = [[]]
        self.primitive = _DummyPrimitive()
        self.names = types.SimpleNamespace(
            target_name=name_prefix,
            temp_storage_bytes=f"{name_prefix}_temp_storage_bytes",
            temp_storage_alignment=f"{name_prefix}_temp_storage_alignment",
            algorithm_struct_size=f"{name_prefix}_struct_size",
            algorithm_struct_alignment=f"{name_prefix}_struct_alignment",
        )
        self.source_code = "#include <cuda/std/cstdint>\nusing dummy_t = int;\n"

    @property
    def temp_storage_bytes(self):
        return self._temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        return self._temp_storage_alignment

    @property
    def algorithm_struct_size(self):
        return self._algorithm_struct_size

    @property
    def algorithm_struct_alignment(self):
        return self._algorithm_struct_alignment


def _install_nvrtc_stub(monkeypatch):
    monkeypatch.setattr(_nvrtc, "nvrtc", _DummyNvrtc)
    _nvrtc.compile_impl.cache_clear()
    _nvrtc.reset_compile_counter()
    _nvrtc.reset_compile_records()
    _nvrtc._set_compile_counter_enabled(True)
    _DummyNvrtc.reset()


def test_nvrtc_compile_counter_counts_cache_misses(monkeypatch):
    _install_nvrtc_stub(monkeypatch)

    _nvrtc.compile(cpp="x", cc=80, rdc=True, code="lto")
    _nvrtc.compile(cpp="x", cc=80, rdc=True, code="lto")

    assert _nvrtc.get_compile_counter() == 1

    _nvrtc.compile(cpp="y", cc=80, rdc=True, code="lto")
    assert _nvrtc.get_compile_counter() == 2


def test_bundle_uses_single_nvrtc_compile(monkeypatch):
    _install_nvrtc_stub(monkeypatch)

    monkeypatch.setattr(_types.cuda, "get_current_device", lambda: _DummyDevice())
    monkeypatch.setattr(_types, "ObjectCode", _DummyObjectCode)
    monkeypatch.setattr(_types, "Linker", _DummyLinker)
    monkeypatch.setattr(_types, "LinkerOptions", _DummyLinkerOptions)
    monkeypatch.setattr(_types, "_get_source_code_rewriter", lambda: None)

    ptx = (
        ".global .align 4 .u32 algo_a_temp_storage_bytes = 64;\n"
        ".global .align 4 .u32 algo_a_temp_storage_alignment = 8;\n"
        ".global .align 4 .u32 algo_a_struct_size = 16;\n"
        ".global .align 4 .u32 algo_a_struct_alignment = 4;\n"
        ".global .align 4 .u32 algo_b_temp_storage_bytes = 96;\n"
        ".global .align 4 .u32 algo_b_temp_storage_alignment = 16;\n"
        ".global .align 4 .u32 algo_b_struct_size = 32;\n"
        ".global .align 4 .u32 algo_b_struct_alignment = 8;\n"
    )
    monkeypatch.setattr(_DummyLinker, "_ptx", ptx, raising=False)

    algo_a = _DummyAlgo("algo_a")
    algo_b = _DummyAlgo("algo_b")

    _types.prepare_ltoir_bundle([algo_a, algo_b], bundle_name="bundle_count")

    assert _nvrtc.get_compile_counter() == 1


def test_nvrtc_dump_sources(tmp_path, monkeypatch):
    _install_nvrtc_stub(monkeypatch)
    monkeypatch.setenv("NUMBA_CCCL_COOP_NVRTC_DUMP_DIR", str(tmp_path))

    _nvrtc.compile(cpp="x", cc=80, rdc=True, code="lto")

    files = sorted(p for p in tmp_path.iterdir() if p.suffix == ".cu")
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert content == "x"


def test_nvrtc_trace_records(tmp_path, monkeypatch):
    _install_nvrtc_stub(monkeypatch)
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("NUMBA_CCCL_COOP_NVRTC_TRACE_PATH", str(trace_path))

    ticks = iter([1.0, 1.25])
    monkeypatch.setattr(_nvrtc.time, "perf_counter", lambda: next(ticks))

    _nvrtc.compile(cpp="trace", cc=80, rdc=True, code="lto")

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["elapsed_ms"] == 250.0
    assert record["source_sha1"]
    assert record["compiled_source_sha1"]
    assert record["pch_mode"] == "off"


def test_nvrtc_pch_canonical_rewrites_source(tmp_path, monkeypatch):
    _install_nvrtc_stub(monkeypatch)
    pch_dir = tmp_path / "pch"
    monkeypatch.setenv("NUMBA_CCCL_COOP_NVRTC_PCH", "canonical")
    monkeypatch.setenv("NUMBA_CCCL_COOP_NVRTC_PCH_DIR", str(pch_dir))
    monkeypatch.setenv(
        "NUMBA_CCCL_COOP_NVRTC_PCH_HEADERS", "cuda/std/cstdint;cub/cub.cuh"
    )

    source = (
        "#include <cuda/std/cstdint>\n"
        "#include <cub/block/block_reduce.cuh>\n"
        "int value = 0;\n"
    )
    _nvrtc.compile(cpp=source, cc=80, rdc=True, code="lto")

    assert _DummyNvrtc.last_source.startswith('#include "cccl_coop_pch_')
    assert "#include <cub/block/block_reduce.cuh>" not in _DummyNvrtc.last_source
    assert "-pch" in _DummyNvrtc.last_options
    assert f"-pch-dir={pch_dir}" in _DummyNvrtc.last_options
    assert f"--include-path={pch_dir}" in _DummyNvrtc.last_options

    headers = list(pch_dir.glob("cccl_coop_pch_*.h"))
    assert len(headers) == 1
    header_text = headers[0].read_text(encoding="utf-8")
    assert "#include <cuda/std/cstdint>" in header_text
    assert "#include <cub/cub.cuh>" in header_text
    assert "#pragma nv_hdrstop" in header_text

    records = _nvrtc.get_compile_records()
    assert len(records) == 1
    assert records[0].pch_mode == "canonical"
    assert records[0].pch_status == "0"


def test_nvrtc_pch_prologue_rewrites_source(tmp_path, monkeypatch):
    _install_nvrtc_stub(monkeypatch)
    pch_dir = tmp_path / "pch"
    header_path = tmp_path / "mamba_pch.h"
    header_path.write_text(
        "#include <cuda/std/cstdint>\n"
        "#include <cub/block/block_scan.cuh>\n"
        "struct storage_t {};\n"
        "#pragma nv_hdrstop\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NUMBA_CCCL_COOP_NVRTC_PCH", "prologue")
    monkeypatch.setenv("NUMBA_CCCL_COOP_NVRTC_PCH_DIR", str(pch_dir))
    monkeypatch.setenv("NUMBA_CCCL_COOP_NVRTC_PCH_PROLOGUE_PATH", str(header_path))
    monkeypatch.setenv(
        "NUMBA_CCCL_COOP_NVRTC_EXTRA_DEFINES",
        "CCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK",
    )

    source = (
        "#include <cuda/std/cstdint>\n"
        "#include <cub/block/block_scan.cuh>\n"
        "struct storage_t {};\n"
        "using algorithm_t = int;\n"
    )
    _nvrtc.compile(cpp=source, cc=80, rdc=True, code="lto")

    assert _DummyNvrtc.last_source.startswith('#include "mamba_pch.h"\n')
    assert _DummyNvrtc.last_source.endswith("using algorithm_t = int;\n")
    assert "-pch" in _DummyNvrtc.last_options
    assert f"--include-path={tmp_path}" in _DummyNvrtc.last_options
    assert "-DCCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK" in _DummyNvrtc.last_options

    records = _nvrtc.get_compile_records()
    assert len(records) == 1
    assert records[0].pch_mode == "prologue"


def test_type_wrapper_uses_stable_unique_names(monkeypatch):
    from mamba_selective_scan_fwd import float2_type

    monkeypatch.setattr(
        _types.cuda,
        "compile",
        lambda *args, **kwargs: (b"fake_ltoir", None),
    )

    wrapper = _types.numba_type_to_wrapper(float2_type, methods=float2_type.methods)

    assert wrapper.cpp_name.startswith("cccl_storage_")
    assert "struct __align__(4) storage_t" not in wrapper.code
    assert 'extern "C" __device__ void construct' not in wrapper.code
    assert 'extern "C" __device__ void assign' not in wrapper.code
    assert all(lto.name.startswith(wrapper.cpp_name) for lto in wrapper.lto_irs)


def test_gpu_dataclass_bundles_temp_storage(monkeypatch):
    _install_nvrtc_stub(monkeypatch)
    monkeypatch.setattr(_types.cuda, "get_current_device", lambda: _DummyDevice())
    monkeypatch.setattr(_types, "ObjectCode", _DummyObjectCode)
    monkeypatch.setattr(_types, "Linker", _DummyLinker)
    monkeypatch.setattr(_types, "LinkerOptions", _DummyLinkerOptions)
    monkeypatch.setattr(_types, "_get_source_code_rewriter", lambda: None)

    ptx = (
        ".global .align 4 .u32 algo_a_temp_storage_bytes = 64;\n"
        ".global .align 4 .u32 algo_a_temp_storage_alignment = 8;\n"
        ".global .align 4 .u32 algo_a_struct_size = 16;\n"
        ".global .align 4 .u32 algo_a_struct_alignment = 4;\n"
        ".global .align 4 .u32 algo_b_temp_storage_bytes = 96;\n"
        ".global .align 4 .u32 algo_b_temp_storage_alignment = 16;\n"
        ".global .align 4 .u32 algo_b_struct_size = 32;\n"
        ".global .align 4 .u32 algo_b_struct_alignment = 8;\n"
    )
    monkeypatch.setattr(_DummyLinker, "_ptx", ptx, raising=False)

    class DummyPrimitive(_types.BasePrimitive):
        def __init__(self, name_prefix):
            self.specialization = _DummyAlgo(name_prefix)

    import numba
    from numba.core.extending import typeof_impl

    @typeof_impl.register(DummyPrimitive)
    def typeof_dummy_primitive(val, c):
        return numba.types.uintp

    prim_a = DummyPrimitive("algo_a")
    prim_b = DummyPrimitive("algo_b")

    from dataclasses import dataclass

    from cuda.coop._dataclass import gpu_dataclass

    @dataclass
    class Traits:
        a: DummyPrimitive
        b: DummyPrimitive

    traits = Traits(prim_a, prim_b)

    _nvrtc.reset_compile_counter()
    _nvrtc._set_compile_counter_enabled(True)

    gpu_dataclass(traits, compute_temp_storage=True)

    assert _nvrtc.get_compile_counter() == 1
    assert traits.temp_storage_bytes_sum == 160
    assert traits.temp_storage_bytes_max == 96
    assert traits.temp_storage_alignment == 16
