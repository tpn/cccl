# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Preserve one CUB copy context across separate issue and wait calls."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from cutlass import Int32, Int64, Uint8, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.cute.ffi import ffi

from ..._core import BindingKind
from ..._core.block.load_to_shared import (
    LOAD_TO_SHARED_INCLUDES,
    load_to_shared_cpp_type,
    make_block_load_to_shared_specialization,
)
from .._compiler import _bundle, _rendering, _state, _storage, _target
from .._compiler._types import TYPE_SPECIFICATIONS
from .._group_load_store import _classify_integer_binding, _resolve_group
from .._temp_storage import TempStorage
from .._thread_data import _make_rmem_tensor
from ._load_store import _memory_pointer, _resolve_memory_type
from ._load_store_layout import static_layout_elements

_SCOPE = "cuda.coop.cutlass.load_to_shared_async"


@dataclass(frozen=True)
class _CopyRequest:
    value_type: type
    block_dim: tuple[int, int, int]
    num_items: int
    phase: str
    kind: str = "cub_load_to_shared"

    @property
    def implementation(self):
        return make_block_load_to_shared_specialization(
            dtype=TYPE_SPECIFICATIONS[self.value_type].token,
            block_dim=self.block_dim,
            num_items=self.num_items,
            phase=self.phase,
        )

    @property
    def cpp_type(self):
        return load_to_shared_cpp_type(
            TYPE_SPECIFICATIONS[self.value_type].cpp_type,
            self.block_dim,
            self.num_items,
        )

    @property
    def symbol_name(self):
        dims = "_".join(map(str, self.block_dim))
        token = TYPE_SPECIFICATIONS[self.value_type].token
        return (
            f"cuda_coop_cutlass_shared_{self.phase}_{token}_"
            f"{dims}_{self.num_items}"
        )


def _render_copy(request):
    implementation = request.cpp_type
    if request.phase == "wait":
        return [
            f"int {request.symbol_name}(unsigned char* context) {{",
            f"  return {implementation}::Wait(context);",
            "}",
        ]
    dtype = TYPE_SPECIFICATIONS[request.value_type].cpp_type
    return [
        f"void {request.symbol_name}(unsigned char* context,",
        "    unsigned int barrier,",
        f"    const {dtype}* source, long long source_count,",
        f"    {dtype}* buffer, long long buffer_count, long long offset) {{",
        "  auto* barrier_ptr = reinterpret_cast<unsigned char*>(",
        "      __cvta_shared_to_generic(static_cast<size_t>(barrier)));",
        f"  {implementation}::Issue(context, barrier_ptr, source,",
        "      source_count, buffer, buffer_count, offset);",
        "}",
    ]


def _entry_block():
    kernel = _storage._active_cuda_kernel_op()
    block = ir.InsertionPoint.current.block
    if block != kernel.regions[0].blocks[0]:
        raise DSLRuntimeError(
            f"{_SCOPE} issue and wait require the kernel entry block; "
            "runtime branches and loops are unsupported."
        )
    return block


class _PendingSharedLoad:
    def __init__(self, context, buffer, request, entry_block):
        self._context = context
        self._buffer = buffer
        self._request = request
        self._entry_block = entry_block
        self._consumed = False

    def __extract_mlir_values__(self):
        raise DSLRuntimeError(
            f"{_SCOPE} pending copies cannot cross runtime control flow."
        )

    def wait(self):
        if _entry_block() != self._entry_block:
            raise DSLRuntimeError(
                f"{_SCOPE} wait must be in the issuing kernel entry block."
            )
        if self._consumed:
            raise DSLRuntimeError(
                f"{_SCOPE} a pending copy must be waited on exactly once."
            )
        displacement = ffi(
            name=self._request.symbol_name,
            params_types=[llvm.PointerType.get(0)],
            return_type=Int32,
        )(self._context.iterator.llvm_ptr)
        result = cute.recast_tensor(
            cute.make_tensor(
                self._buffer.iterator + displacement,
                # CuTe requires a dynamic extent for an empty view.
                cute.make_layout(self._request.num_items or Int32(0)),
            ),
            self._request.value_type,
        )
        self._consumed = True
        return result


def validate_pending_loads(session):
    """Reject an unfinished copy before compiling or launching its kernel."""
    for pending in session.pending_shared_loads:
        if not pending._consumed:
            raise DSLRuntimeError(f"{_SCOPE} requires a matching wait().")


def _require_memory(value, space, name):
    pointer = getattr(value, "iterator", value)
    if getattr(pointer, "memspace", None) != space:
        raise TypeError(f"{_SCOPE} {name} must use {space.name} memory")
    shape = getattr(value, "shape", None)
    if shape is not None and cute.rank(shape) != 1:
        raise TypeError(f"{_SCOPE} {name} must be one-dimensional")
    return pointer


def provider_load_to_shared_async(
    group, source, shared_buffer, *, num_items, offset
):
    entry_block = _entry_block()
    group, launch, _ = _resolve_group(
        group, "direct", None, "load_to_shared_async"
    )
    if group.kind != "block" or group.static_size % 32:
        raise ValueError(f"{_SCOPE} requires a block size divisible by 32")
    if isinstance(num_items, bool) or not isinstance(num_items, Integral):
        raise TypeError(f"{_SCOPE} num_items must be a compile-time integer")
    count = int(num_items)
    if count < 0:
        raise ValueError(f"{_SCOPE} num_items must be nonnegative")
    binding = _classify_integer_binding(offset, name="offset")
    if isinstance(offset, Uint64):
        raise TypeError(f"{_SCOPE} offset does not support uint64")
    value_type = _resolve_memory_type(
        source, primitive_name="load_to_shared_async"
    )
    buffer_type = _resolve_memory_type(
        shared_buffer, primitive_name="load_to_shared_async"
    )
    if buffer_type is not value_type:
        raise TypeError(
            f"{_SCOPE} source and shared buffer dtypes must match: "
            f"{value_type.__name__} and {buffer_type.__name__}"
        )
    _require_memory(source, cute.AddressSpace.gmem, "source")
    buffer_pointer = _require_memory(
        shared_buffer, cute.AddressSpace.smem, "shared_buffer"
    )
    if buffer_pointer.alignment < 16:
        raise ValueError(f"{_SCOPE} shared_buffer requires 16-byte alignment")
    itemsize = value_type.width // 8
    num_bytes = count * itemsize
    required_bytes = ((num_bytes + 15) // 16 * 16 + 16) if count else 0
    if required_bytes > (1 << 31) - 1:
        raise ValueError(f"{_SCOPE} padded byte count must fit a signed int")
    capacity = static_layout_elements(shared_buffer)
    if capacity is None:
        raise TypeError(f"{_SCOPE} shared_buffer requires a static capacity")
    if capacity * itemsize < required_bytes:
        raise ValueError(
            f"{_SCOPE} shared_buffer has insufficient padded capacity"
        )
    source_count = static_layout_elements(source)
    static_offset = (
        binding.value if binding.kind is BindingKind.STATIC else None
    )
    if (
        source_count is not None
        and static_offset is not None
        and static_offset + count > source_count
    ):
        raise ValueError(f"{_SCOPE} source range exceeds its capacity")
    source_pointer = _memory_pointer(
        source, primitive_name="load_to_shared_async"
    )
    destination_pointer = _memory_pointer(
        shared_buffer, primitive_name="load_to_shared_async"
    )
    block_dim = launch.exact_block_dim
    issue = _CopyRequest(value_type, block_dim, count, "issue")
    wait = _CopyRequest(value_type, block_dim, count, "wait")
    arch = _target.resolve_nvrtc_arch(
        _SCOPE, lambda: _target.configured_gpu_arch(_state._get_cute_dsl)
    )
    compilation = _bundle.compile_bundle_source_with_layouts(
        _rendering.render_bundle_source((issue, wait)),
        arch=arch,
        required_headers=LOAD_TO_SHARED_INCLUDES,
        layout_probes=(
            _rendering.make_scratch_layout_probe(
                "context", f"typename {issue.cpp_type}::Context"
            ),
            _rendering.make_scratch_layout_probe(
                "barrier", f"typename {issue.cpp_type}::TempStorage"
            ),
        ),
    )
    context_layout = compilation.layouts["context"]
    context = _make_rmem_tensor(
        context_layout.size_in_bytes, Uint8, context_layout.alignment
    )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(issue)
        _state.register_request(wait)
        barrier = TempStorage(auto_sync=False)
        barrier_address, _, _ = _storage.register_deferred_temp_storage_event(
            barrier,
            primitive_name="load_to_shared_async",
            requirement_key=None,
            reservation_layout=compilation.layouts["barrier"],
        )
        from cutlass import Uint32

        ffi(
            name=issue.symbol_name,
            params_types=[
                llvm.PointerType.get(0),
                Uint32,
                llvm.PointerType.get(0),
                Int64,
                llvm.PointerType.get(0),
                Int64,
                Int64,
            ],
            return_type=None,
        )(
            context.iterator.llvm_ptr,
            barrier_address,
            source_pointer,
            Int64(-1 if source_count is None else source_count),
            destination_pointer,
            Int64(capacity),
            Int64(offset),
        )
        pending = _PendingSharedLoad(context, shared_buffer, wait, entry_block)
        _state.active_bundle_session().pending_shared_loads.append(pending)
        return pending
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise


_rendering.register_bundle_renderer(
    "cub_load_to_shared",
    render=_render_copy,
    include_lines=tuple(
        f"#include <{header}>" for header in LOAD_TO_SHARED_INCLUDES
    ),
    cccl_headers=tuple((header, header) for header in LOAD_TO_SHARED_INCLUDES),
)
