# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe a split CUB global-to-shared transfer for compiler adapters.

The issue and wait calls share a per-thread context and a dedicated block-wide
barrier. Adapters query the actual C++ layouts, keep both allocations alive,
and enforce a single wait in the same collective control-flow region. Neither
call uses the ordinary implicit temporary-storage ABI.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

from .._algorithm import Algorithm, TypeDefinition
from .._types import (
    INT32,
    INT64,
    UINT8,
    Dependency,
    Pointer,
    Reference,
    TemplateParameter,
    Value,
)
from ._common import normalize_block_dim

LOAD_TO_SHARED_INCLUDES = (
    "cub/block/block_load_to_shared.cuh",
    "cuda/std/__new/device_new.h",
    "cuda/std/cstdint",
    "cuda/std/limits",
    "cuda/std/span",
    "cuda/std/utility",
)

_CPP_NAME = "CudaCoopBlockLoadToShared"
_TEMPLATE_NAMES = ("T", "BlockDimX", "BlockDimY", "BlockDimZ", "NumItems")
_DEFINITION = TypeDefinition(
    name=_CPP_NAME,
    code=r"""
namespace cub {
template <typename T, int BlockDimX, int BlockDimY, int BlockDimZ, int NumItems>
struct CudaCoopBlockLoadToShared
{
  static_assert(NumItems >= 0, "num_items must be non-negative");
  static_assert(NumItems <= (::cuda::std::numeric_limits<int>::max() - 31) / static_cast<int>(sizeof(T)),
                "load_to_shared_async padded byte count must fit an int");
  using Loader = ::cub::detail::BlockLoadToShared<BlockDimX, BlockDimY, BlockDimZ>;
  using TempStorage = typename Loader::TempStorage;
  static constexpr int BufferBytes = ::cub::detail::LoadToSharedBufferSizeBytes<T>(NumItems);
  static constexpr int BufferAlignment = ::cub::detail::LoadToSharedBufferAlignBytes<T>();

  struct Context
  {
    Loader loader;
    const int first;
    typename Loader::CommitToken token;

    _CCCL_DEVICE_API _CCCL_FORCEINLINE
    Context(TempStorage& barrier, const T* source, T* buffer)
        : loader(barrier)
        , first(Copy(source, buffer))
        , token(loader.Commit())
    {}

    _CCCL_DEVICE_API _CCCL_FORCEINLINE int Copy(const T* source, T* buffer)
    {
      const auto values = loader.CopyAsync(
        ::cuda::std::span<char>{reinterpret_cast<char*>(buffer), BufferBytes},
        ::cuda::std::span<const T>{source, NumItems});
      // An empty view has no data and needs no leading source-alignment pad.
      if constexpr (NumItems == 0)
      {
        return 0;
      }
      return static_cast<int>(values.data() - buffer);
    }
  };

  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void Issue(
    unsigned char* state, unsigned char* barrier, const T* source,
    long long source_count, T* buffer, long long buffer_count, long long offset)
  {
    using Address = ::cuda::std::uintptr_t;
    const auto state_address = reinterpret_cast<Address>(state);
    const auto barrier_address = reinterpret_cast<Address>(barrier);
    const auto source_address = reinterpret_cast<Address>(source);
    const auto buffer_address = reinterpret_cast<Address>(buffer);
    constexpr auto max_address = ::cuda::std::numeric_limits<Address>::max();
    // A negative source count is used only for native pointers without an extent.
    // Their caller retains the allocation-bounds precondition.
    if (offset < 0 || source_count < -1 || buffer_count < 0
        || (source_count >= 0 && (source_count < NumItems || offset > source_count - NumItems))
        || buffer_count < (static_cast<long long>(BufferBytes) + static_cast<long long>(sizeof(T)) - 1)
                            / static_cast<long long>(sizeof(T))
        || source_address % alignof(T) != 0 || buffer_address % BufferAlignment != 0
        || state_address % alignof(Context) != 0 || barrier_address % alignof(TempStorage) != 0
        || state == nullptr || !__isLocal(state) || barrier == nullptr || !__isShared(barrier)
        || (source != nullptr && !__isGlobal(source))
        || (buffer != nullptr && !__isShared(buffer))
        || (NumItems != 0 && (source == nullptr || buffer == nullptr))
        || blockDim.x != BlockDimX || blockDim.y != BlockDimY || blockDim.z != BlockDimZ)
    {
      asm volatile("trap;");
      return;
    }
    const auto available_items = (max_address - source_address) / sizeof(T);
    if (static_cast<unsigned long long>(offset) > available_items)
    {
      asm volatile("trap;");
      return;
    }
    if constexpr (NumItems > 0)
    {
      if (NumItems > available_items - static_cast<unsigned long long>(offset))
      {
        asm volatile("trap;");
        return;
      }
    }
    const auto adjusted_source = reinterpret_cast<const T*>(
      source_address + static_cast<Address>(offset) * sizeof(T));
    ::new (state) Context(*reinterpret_cast<TempStorage*>(barrier), adjusted_source, buffer);
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE static int Wait(unsigned char* state)
  {
    auto* context = reinterpret_cast<Context*>(state);
    context->loader.Wait(::cuda::std::move(context->token));
    // Invalidate synchronizes the block before and after releasing the barrier.
    context->loader.Invalidate();
    const int first = context->first;
    context->~Context();
    return first;
  }
};
}
""",  # noqa: E501 - Embedded C++ source.
)


def _normalize_shape(
    block_dim: Any, num_items: Any
) -> tuple[tuple[int, int, int], int]:
    shape = normalize_block_dim(block_dim)
    threads = shape[0] * shape[1] * shape[2]
    if threads > 1024 or threads % 32:
        raise ValueError(
            "load_to_shared_async requires a block size divisible by 32 "
            "and no larger than 1024"
        )
    if (
        not isinstance(num_items, Integral)
        or isinstance(num_items, bool)
        or not 0 <= num_items <= 2**31 - 1
    ):
        raise ValueError("num_items must be a non-negative 32-bit integer")
    return shape, int(num_items)


def load_to_shared_type_definition() -> TypeDefinition:
    """Return the facade containing the retained loader and commit token."""

    return _DEFINITION


def load_to_shared_cpp_type(
    cpp_dtype: str, block_dim: Any, num_items: Any
) -> str:
    """Name a facade specialization for layout probes and direct wrappers.

    ``cpp_dtype`` comes from the adapter's validated C++ type mapping. Probe
    ``Context`` and ``TempStorage`` under this type for their sizes/alignments.
    """

    shape, count = _normalize_shape(block_dim, num_items)
    arguments = ", ".join((cpp_dtype, *(str(v) for v in shape), str(count)))
    return f"::cub::{_CPP_NAME}<{arguments}>"


def make_block_load_to_shared_specialization(
    *, dtype: Any, block_dim: Any, num_items: Any, phase: str
) -> Algorithm:
    """Bind an issue/wait entry point with explicit state and barrier pointers.

    ``issue`` takes state, barrier, source, source count, destination,
    destination count, and source element offset. A source count of ``-1`` is a
    pointer with caller-guaranteed bounds. ``wait`` takes the same state pointer
    and returns the first element of the completed view within the destination.
    """

    shape, count = _normalize_shape(block_dim, num_items)
    if phase == "issue":
        parameters = (
            Pointer(UINT8, name="state", is_inout=True, is_return=False),
            Pointer(UINT8, name="barrier", is_inout=True, is_return=False),
            Pointer(Dependency("T"), name="source"),
            Value(INT64, name="source_count"),
            Pointer(
                Dependency("T"),
                name="buffer",
                is_output=True,
                is_return=False,
            ),
            Value(INT64, name="buffer_count"),
            Value(INT64, name="offset"),
        )
    elif phase == "wait":
        parameters = (
            Pointer(UINT8, name="state", is_inout=True, is_return=False),
            Reference(INT32, name="first", is_output=True, is_return=True),
        )
    else:
        raise ValueError("phase must be 'issue' or 'wait'")
    return Algorithm(
        struct_name=_CPP_NAME,
        method_name=phase.title(),
        c_name=f"block_load_to_shared_{phase}",
        includes=LOAD_TO_SHARED_INCLUDES,
        template_parameters=tuple(
            TemplateParameter(name) for name in _TEMPLATE_NAMES
        ),
        parameters=(parameters,),
        template_arguments=dict(zip(_TEMPLATE_NAMES, (dtype, *shape, count))),
        type_definitions=(_DEFINITION,),
        metadata={"scope": "block", "primitive": "load_to_shared_async"},
    )


__all__ = [
    "LOAD_TO_SHARED_INCLUDES",
    "load_to_shared_cpp_type",
    "load_to_shared_type_definition",
    "make_block_load_to_shared_specialization",
]
