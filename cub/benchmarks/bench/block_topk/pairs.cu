// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_topk.cuh>

#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include <algorithm>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <nvbench_helper.cuh>

using algorithm = cub::detail::block_topk_algorithm;
using algorithms =
  nvbench::enum_type_list<algorithm::air, algorithm::warp_select, algorithm::warp_select_with_zero_check>;
using k_values     = nvbench::enum_type_list<16, 31, 32>;
using valid_counts = nvbench::enum_type_list<80, 93, 96>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  algorithm,
  [](algorithm value) {
    switch (value)
    {
      case algorithm::air:
        return "air";
      case algorithm::warp_select:
        return "warp_select";
      default:
        return "warp_select_with_zero_check";
    }
  },
  [](auto) {
    return std::string{};
  })

inline constexpr int block_threads = 64;
inline constexpr int thread_items  = 2;
inline constexpr int tile_items    = block_threads * thread_items;

template <algorithm Algorithm, int K, int ValidItems>
__global__ void select_pairs(const float* input, float* scores, int* indices)
{
  using block_topk = cub::detail::block_topk<float, block_threads, thread_items, int, Algorithm>;
  __shared__ typename block_topk::TempStorage storage;

  float keys[thread_items];
  int values[thread_items];
  const auto tile = static_cast<cuda::std::int64_t>(blockIdx.x);
  for (int i = 0; i < thread_items; ++i)
  {
    const int slot = threadIdx.x * thread_items + i;
    keys[i]        = input[tile * tile_items + slot];
    values[i]      = slot;
  }

  block_topk{storage}.template max_pairs<false>(keys, values, K, ValidItems);

  for (int i = 0; i < thread_items; ++i)
  {
    const int slot = threadIdx.x * thread_items + i;
    if (slot < K)
    {
      scores[tile * K + slot]  = keys[i];
      indices[tile * K + slot] = values[i];
    }
  }
}

std::vector<float> make_input(cuda::std::int64_t tiles, int k, int valid_items, const std::string& distribution)
{
  std::vector<float> input(tiles * tile_items, std::numeric_limits<float>::max());
  std::mt19937 rng{123};
  std::normal_distribution<float> normal;
  for (cuda::std::int64_t tile = 0; tile < tiles; ++tile)
  {
    const auto first = input.begin() + tile * tile_items;
    if (distribution == "normal")
    {
      std::generate_n(first, valid_items, [&] {
        return normal(rng);
      });
      continue;
    }

    // All controlled inputs have identical winning pairs after the shuffle. Only the losing keys differ.
    // common_prefix needs the last byte. Both of the other inputs separate winners in the first byte,
    // but only sign_separated has exactly k nonnegative keys.
    for (int rank = 0; rank < valid_items; ++rank)
    {
      const auto offset = static_cast<cuda::std::uint32_t>(rank);
      float key         = cuda::std::bit_cast<float>(0x3f800000u + offset);
      if (rank < valid_items - k)
      {
        if (distribution == "sign_separated")
        {
          key = -key;
        }
        else if (distribution == "positive_separated")
        {
          key = cuda::std::bit_cast<float>(0x3d000000u + offset);
        }
      }
      first[rank] = key;
    }
    std::shuffle(first, first + valid_items, rng);
  }
  return input;
}

template <algorithm Algorithm, int K, int ValidItems>
void pairs(nvbench::state& state,
           nvbench::type_list<nvbench::enum_type<Algorithm>, nvbench::enum_type<K>, nvbench::enum_type<ValidItems>>)
{
  const auto tiles = state.get_int64("Tiles{io}");
  if (tiles < 1 || tiles > std::numeric_limits<int>::max())
  {
    state.skip("Tile count must fit the grid's x dimension.");
    return;
  }
  const auto host = make_input(tiles, K, ValidItems, state.get_string("Distribution"));
  thrust::device_vector<float> input(host.begin(), host.end());
  thrust::device_vector<float> scores(tiles * K, thrust::no_init);
  thrust::device_vector<int> indices(tiles * K, thrust::no_init);

  const float* in = thrust::raw_pointer_cast(input.data());
  float* out      = thrust::raw_pointer_cast(scores.data());
  int* idx        = thrust::raw_pointer_cast(indices.data());

  state.add_element_count(tiles * ValidItems);
  state.add_global_memory_reads<float>(tiles * tile_items);
  state.add_global_memory_writes<float>(tiles * K, "Scores");
  state.add_global_memory_writes<int>(tiles * K, "Indices");
  // Time complete tile selections, including their score/index stores. Replaying the same
  // read-only inputs also lets NVBench measure the short kernels with warm caches.
  state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& launch) {
    select_pairs<Algorithm, K, ValidItems>
      <<<static_cast<int>(tiles), block_threads, 0, launch.get_stream()>>>(in, out, idx);
  });
}

NVBENCH_BENCH_TYPES(pairs, NVBENCH_TYPE_AXES(algorithms, k_values, valid_counts))
  .set_name("base")
  .set_type_axes_names({"Algorithm{ct}", "K{ct}", "ValidItems{ct}"})
  .add_int64_axis("Tiles{io}", {4096})
  .add_string_axis("Distribution", {"normal", "sign_separated", "common_prefix", "positive_separated"});
