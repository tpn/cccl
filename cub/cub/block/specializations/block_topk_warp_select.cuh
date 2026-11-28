// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/topk_helpers.cuh>
#include <cub/util_arch.cuh>
#include <cub/warp/specializations/warp_redux.cuh>
#include <cub/warp/warp_shuffle_sort.cuh>

#include <cuda/__functional/minimum.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN
namespace detail
{
//! @brief Select the largest k float keys and their int values from a partial 64-thread tile.
//!
//! Each thread holds two consecutive input pairs in blocked order. The network finds the cutoff
//! using keys alone, then compacts the winning pairs into the first k blocked slots in unspecified order.
class block_topk_warp_select
{
  using bits_t = ::cuda::std::uint32_t;

  struct pair
  {
    bits_t bits;
    int value;
  };

public:
  struct TempStorage
  {
    bits_t sorted[warp_threads];
    pair original[warp_threads];
    pair output[warp_threads];
  };

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr bool supports(int k, int num_valid) noexcept
  {
    return 0 < k && k <= warp_threads && k < num_valid && num_valid <= 3 * warp_threads;
  }

private:
  template <typename Compare>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static bits_t sort_column(bits_t rank, Compare compare)
  {
    bits_t keys[1] = {rank};
    WarpShuffleSort<bits_t, 1>{}.Sort(keys, compare);
    return keys[0];
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static bits_t warp_min(bits_t value)
  {
    if (const auto result = _CUB::detail::warp_redux(value, ~0u, ::cuda::minimum<>{}))
    {
      return *result;
    }
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int stride = warp_threads / 2; stride > 0; stride /= 2)
    {
      const bits_t peer = ::cuda::device::warp_shuffle_xor(value, stride);
      value             = (::cuda::std::min) (value, peer);
    }
    return value;
  }

  // Try the normalized zero rank as a cutoff. Both signed zeros and positive NaN encodings qualify.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE static bool
  select_at_zero(TempStorage& storage, pair first, pair second, int k, int lane, bits_t prior)
  {
    constexpr bits_t sign_bit = bits_t{1} << 31;
    const bool take_first     = first.bits <= sign_bit;
    const bool take_second    = second.bits <= sign_bit;
    const auto first_mask     = __ballot_sync(~0u, take_first);
    const auto second_mask    = __ballot_sync(~0u, take_second);
    const int first_count     = ::cuda::std::popcount(first_mask);
    const int warp_count      = first_count + ::cuda::std::popcount(second_mask);
    // Each warp contributes min(warp_count, 32) votes. For k < 32 equality is exact.
    if (__syncthreads_count(lane < warp_count) != k)
    {
      return false;
    }
    const int base = threadIdx.x < warp_threads ? 0 : k - warp_count;
    if (take_first)
    {
      storage.output[base + ::cuda::std::popcount(first_mask & prior)] = first;
    }
    if (take_second)
    {
      storage.output[base + first_count + ::cuda::std::popcount(second_mask & prior)] = second;
    }
    __syncthreads();
    return true;
  }

  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void
  read_output(const TempStorage& storage, float (&keys)[2], int (&values)[2], int k)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < 2; ++i)
    {
      const int slot = static_cast<int>(threadIdx.x) * 2 + i;
      if (slot < k)
      {
        const pair result = storage.output[slot];
        keys[i]           = ::cuda::std::bit_cast<float>(result.bits);
        values[i]         = result.value;
      }
    }
  }

public:
  //! @pre All 64 threads in the one-dimensional block must participate with the same k and num_valid.
  //!      The valid input is the first num_valid blocked slots, with ``1 <= k <= 32`` and ``k < num_valid <= 96``.
  //! @smemreuse
  template <bool CheckZero>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE static void
  select_pairs(TempStorage& storage, float (&keys)[2], int (&values)[2], int k, int num_valid)
  {
    _CCCL_ASSERT(supports(k, num_valid), "warp selection requires 1 <= k <= 32 and k < num_valid <= 96");
    const int tid      = static_cast<int>(threadIdx.x);
    const int lane     = tid % warp_threads;
    const bits_t prior = ::cuda::ptx::get_sreg_lanemask_lt();
    // The lowest NaN encoding may also be valid. Padding only participates in cutoff discovery.
    constexpr pair padding{~bits_t{0}, 0};
    const pair first  = 2 * tid < num_valid ? pair{::cuda::std::bit_cast<bits_t>(keys[0]), values[0]} : padding;
    const pair second = 2 * tid + 1 < num_valid ? pair{::cuda::std::bit_cast<bits_t>(keys[1]), values[1]} : padding;
    if constexpr (CheckZero)
    {
      if (k < warp_threads && select_at_zero(storage, first, second, k, lane, prior))
      {
        read_output(storage, keys, values, k);
        return;
      }
    }

    const bits_t first_rank  = _CUB::detail::topk::twiddle_in_bits<float, true>(first.bits);
    const bits_t second_rank = _CUB::detail::topk::twiddle_in_bits<float, true>(second.bits);
    bits_t kept{};
    if (tid < warp_threads)
    {
      // Match two oppositely sorted runs. Their pairwise maxima are the largest 32
      // inputs of the first warp in bitonic order. Merge them into descending order.
      const bits_t descending = sort_column(first_rank, ::cuda::std::greater<>{});
      const bits_t ascending  = sort_column(second_rank, ::cuda::std::less<>{});
      kept                    = (::cuda::std::max) (descending, ascending);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int stride = warp_threads / 2; stride > 0; stride /= 2)
      {
        const bits_t peer = ::cuda::device::warp_shuffle_xor(kept, stride);
        kept              = (lane & stride) == 0 ? (::cuda::std::max) (kept, peer) : (::cuda::std::min) (kept, peer);
      }
    }
    else
    {
      // Gather the remaining blocked inputs into a third column, one pair per lane.
      const pair even        = ::cuda::device::warp_shuffle_idx(first, lane / 2);
      const pair odd         = ::cuda::device::warp_shuffle_idx(second, lane / 2);
      const pair item        = (lane & 1) == 0 ? even : odd;
      storage.original[lane] = item;
      storage.sorted[lane] =
        sort_column(_CUB::detail::topk::twiddle_in_bits<float, true>(item.bits), ::cuda::std::less<>{});
    }
    __syncthreads();

    if (tid < warp_threads)
    {
      // For descending runs A and B, max(A[i], B[k-1-i]) for 0 <= i < k gives
      // the k winners in unspecified order. B is stored ascending here.
      const bits_t peer   = storage.sorted[(lane + warp_threads - k) % warp_threads];
      const bits_t winner = (::cuda::std::max) (kept, peer);
      const bits_t cutoff = warp_min(lane < k ? winner : ~bits_t{0});
      const pair inputs[] = {first, second, storage.original[lane]};
      const bits_t ranks[] = {first_rank, second_rank, _CUB::detail::topk::twiddle_in_bits<float, true>(inputs[2].bits)};
      const bool valid[] = {2 * lane < num_valid, 2 * lane + 1 < num_valid, 2 * warp_threads + lane < num_valid};
      bits_t greater[3], equal[3];
      int num_greater = 0;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < 3; ++i)
      {
        greater[i] = __ballot_sync(~0u, valid[i] && ranks[i] > cutoff);
        equal[i]   = __ballot_sync(~0u, valid[i] && ranks[i] == cutoff);
        num_greater += ::cuda::std::popcount(greater[i]);
      }
      // Give every strict winner a slot. Fill the remaining slots with distinct ties.
      int greater_base = 0;
      int equal_base   = num_greater;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < 3; ++i)
      {
        const bool better = ranks[i] > cutoff;
        const int slot =
          (better ? greater_base : equal_base) + ::cuda::std::popcount(prior & (better ? greater[i] : equal[i]));
        if (valid[i] && ranks[i] >= cutoff && slot < k)
        {
          storage.output[slot] = inputs[i];
        }
        greater_base += ::cuda::std::popcount(greater[i]);
        equal_base += ::cuda::std::popcount(equal[i]);
      }
      __syncwarp();
      read_output(storage, keys, values, k);
    }
  }
};
} // namespace detail
CUB_NAMESPACE_END
