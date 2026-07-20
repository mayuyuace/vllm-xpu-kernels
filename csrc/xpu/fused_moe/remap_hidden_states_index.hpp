#pragma once

#include <sycl/sycl.hpp>

#include "csrc/utils.h"
#include "csrc/dispatch_utils.h"

namespace vllm {
namespace fused_moe {

class RowsPerExpertCount {
 public:
  RowsPerExpertCount(
      int* expert_map,
      int* rows_per_expert,
      void* topk_ids,
      bool is_topk_ids_int32,
      int* unpermuted_row_to_permuted_row,
      const int num_rows,
      const int TopK,
      const int local_experts_num,
      sycl::local_accessor<int32_t, 1> local_counts)
      : expert_map(expert_map),
        rows_per_expert(rows_per_expert),
        topk_ids(topk_ids),
        is_topk_ids_int32(is_topk_ids_int32),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        num_rows(num_rows),
        TopK(TopK),
        local_experts_num(local_experts_num),
        local_counts(local_counts) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;

  static inline sycl::nd_range<1>
  get_nd_range(const int num_rows, const int TopK) {
    int group_nums = (num_rows * TopK + GroupWorkItem - 1) / GroupWorkItem;
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(group_nums);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto global_id = item.get_global_linear_id();
    auto local_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    // ===== Phase 1: init SLM =====
    for (int i = local_id; i < local_experts_num; i += local_range) {
      local_counts[i] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // ===== Phase 2: local atomic =====
    if (global_id < num_rows * TopK) {
      int global_expert_id =
          is_topk_ids_int32 ? reinterpret_cast<int32_t*>(topk_ids)[global_id]
                            : reinterpret_cast<int64_t*>(topk_ids)[global_id];
      int local_expert_id = global_expert_id;
      if (expert_map != nullptr) {
        local_expert_id = expert_map[global_expert_id];
      }

      if (local_expert_id == -1) {
        unpermuted_row_to_permuted_row[global_id] = -1;
      } else {
        auto local_atomic = sycl::atomic_ref<
            int,
            sycl::memory_order_relaxed,
            sycl::memory_scope_work_group,
            sycl::access::address_space::local_space>(
            local_counts[local_expert_id]);
        int local_old = local_atomic.fetch_add(1);

        unpermuted_row_to_permuted_row[global_id] = local_old;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    // ===== Phase 3: global atomic =====
    for (int i = local_id; i < local_experts_num; i += local_range) {
      int count = local_counts[i];
      if (count > 0) {
        auto global_atomic = sycl::atomic_ref<
            int,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>(rows_per_expert[i]);
        int base = global_atomic.fetch_add(count);
        local_counts[i] = base;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    // ===== Phase 4: fix unpermuted_row_to_permuted_row =====
    if (global_id < num_rows * TopK) {
      int global_expert_id =
          is_topk_ids_int32 ? reinterpret_cast<int32_t*>(topk_ids)[global_id]
                            : reinterpret_cast<int64_t*>(topk_ids)[global_id];
      int local_expert_id = global_expert_id;
      if (expert_map != nullptr) {
        local_expert_id = expert_map[global_expert_id];
      }

      if (local_expert_id != -1) {
        // local_old + base_offset = global_offset
        unpermuted_row_to_permuted_row[global_id] +=
            local_counts[local_expert_id];
      }
    }
  }

 private:
  int* expert_map;
  int* rows_per_expert;
  void* topk_ids;
  bool is_topk_ids_int32;
  int* unpermuted_row_to_permuted_row;
  const int num_rows;
  const int TopK;
  const int local_experts_num;
  sycl::local_accessor<int32_t, 1> local_counts;
};

class RemapHiddenStatesIndex {
 public:
  RemapHiddenStatesIndex(
      sycl::local_accessor<int32_t, 1>& slm,
      int* expert_map,
      int* unpermuted_row_to_permuted_row,
      int* permuted_row_to_unpermuted_row,
      int* rows_per_expert,
      void* topk_ids,
      bool is_topk_ids_int32,
      const int num_rows,
      const int TopK,
      const int total_experts_num,
      const int local_experts_num)
      : slm(slm),
        expert_map(expert_map),
        unpermuted_row_to_permuted_row(unpermuted_row_to_permuted_row),
        permuted_row_to_unpermuted_row(permuted_row_to_unpermuted_row),
        rows_per_expert(rows_per_expert),
        topk_ids(topk_ids),
        is_topk_ids_int32(is_topk_ids_int32),
        num_rows(num_rows),
        TopK(TopK),
        total_experts_num(total_experts_num),
        local_experts_num(local_experts_num) {}

  static constexpr int GroupWorkItem = 256;
  static constexpr int WARP_SIZE = 32;
  static constexpr int EXCLUSIVE_SIZE = 1024;

  static inline sycl::nd_range<1>
  get_nd_range(const int num_rows, const int TopK) {
    int group_nums = (num_rows * TopK + GroupWorkItem - 1) / GroupWorkItem;
    sycl::range<1> local(GroupWorkItem);
    sycl::range<1> group(group_nums);
    return sycl::nd_range<1>(local * group, local);
  }

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto local_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);
    auto group_id = item.get_group(0);
    auto global_id = item.get_global_linear_id();

    int32_t* expert_cumsum_ptr = static_cast<int32_t*>(
        slm.template get_multi_ptr<sycl::access::decorated::no>().get());

    if (local_id == 0) {
      expert_cumsum_ptr[0] = 0;
    }
    for (int i = local_id; i < local_experts_num - 1; i += local_range) {
      expert_cumsum_ptr[i + 1] = rows_per_expert[i];
    }

    item.barrier(sycl::access::fence_space::local_space);

    sycl::joint_inclusive_scan(
        item.get_group(),
        expert_cumsum_ptr,
        expert_cumsum_ptr + local_experts_num,
        expert_cumsum_ptr,
        sycl::plus<int>{});

    if (global_id >= (size_t)num_rows * TopK) return;

    int global_expert_id =
        is_topk_ids_int32 ? reinterpret_cast<int32_t*>(topk_ids)[global_id]
                          : reinterpret_cast<int64_t*>(topk_ids)[global_id];
    int local_expert_id =
        expert_map != nullptr ? expert_map[global_expert_id] : global_expert_id;

    // Masked slot: kernel1 already wrote -1, keep the sentinel untouched.
    if (local_expert_id == -1) return;

    unpermuted_row_to_permuted_row[global_id] +=
        expert_cumsum_ptr[local_expert_id];
    permuted_row_to_unpermuted_row[unpermuted_row_to_permuted_row[global_id]] =
        global_id;
  }

 private:
  sycl::local_accessor<int32_t, 1> slm;
  int* expert_map;
  int* unpermuted_row_to_permuted_row;
  int* permuted_row_to_unpermuted_row;
  int* rows_per_expert;
  void* topk_ids;
  bool is_topk_ids_int32;
  const int num_rows;
  const int TopK;
  const int total_experts_num;
  const int local_experts_num;
};

void RemapHiddenStatesLauncher(
    int* expert_map,
    int* rows_per_expert,
    int* unpermuted_row_to_permuted_row,
    int* permuted_row_to_unpermuted_row,
    void* topk_ids,
    bool is_topk_ids_int32,
    const int num_rows,
    const int TopK,
    const int total_experts_num,
    const int local_experts_num,
    sycl::queue& queue) {
  TORCH_CHECK(
      (local_experts_num <= (RemapHiddenStatesIndex::EXCLUSIVE_SIZE)),
      "local_experts_num exceeds the maximum supported number");

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_counts(
        sycl::range<1>(local_experts_num), cgh);
    cgh.parallel_for(
        RowsPerExpertCount::get_nd_range(num_rows, TopK),
        RowsPerExpertCount{
            expert_map,
            rows_per_expert,
            topk_ids,
            is_topk_ids_int32,
            unpermuted_row_to_permuted_row,
            num_rows,
            TopK,
            local_experts_num,
            local_counts});
  });

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> slm(
        sycl::range<1>(local_experts_num), cgh);
    cgh.parallel_for(
        RemapHiddenStatesIndex::get_nd_range(num_rows, TopK),
        RemapHiddenStatesIndex{
            slm,
            expert_map,
            unpermuted_row_to_permuted_row,
            permuted_row_to_unpermuted_row,
            rows_per_expert,
            topk_ids,
            is_topk_ids_int32,
            num_rows,
            TopK,
            total_experts_num,
            local_experts_num});
  });
}

}  // namespace fused_moe
}  // namespace vllm

void remap_hidden_states_index(
    const c10::optional<torch::Tensor>& expert_map,  // [total_experts_num]
    torch::Tensor& rows_per_expert,                  // [local_experts_num]
    torch::Tensor& unpermuted_row_to_permuted_row,   // [num_rows, TopK]
    torch::Tensor& permuted_row_to_unpermuted_row,   // [num_rows, TopK]
    torch::Tensor& topk_ids,                         // [num_rows, TopK]
    int64_t total_experts_num,
    int64_t local_experts_num) {
  if (expert_map.has_value()) {
    TORCH_CHECK(
        expert_map->scalar_type() == torch::kInt32, "expert_map must be int32");
  }

  TORCH_CHECK(
      rows_per_expert.scalar_type() == torch::kInt32,
      "rows_per_expert must be int32");

  TORCH_CHECK(
      topk_ids.scalar_type() == torch::kInt64 ||
          topk_ids.scalar_type() == torch::kInt32,
      "topk_ids must be int64 or int32");

  int num_rows = topk_ids.size(0);
  int TopK = topk_ids.size(1);

  if (expert_map.has_value()) {
    TORCH_CHECK(
        expert_map->size(0) == total_experts_num,
        "expert_map must be [total_experts_num]");
  }
  TORCH_CHECK(
      rows_per_expert.size(0) == local_experts_num,
      "rows_per_expert must be [local_experts_num]");
  TORCH_CHECK(
      topk_ids.size(0) == num_rows && topk_ids.size(1) == TopK,
      "topk_ids must be [num_rows, TopK]");

  const at::DeviceGuard device_guard(topk_ids.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  vllm::fused_moe::RemapHiddenStatesLauncher(
      expert_map.has_value() ? reinterpret_cast<int*>(expert_map->data_ptr())
                             : nullptr,
      reinterpret_cast<int*>(rows_per_expert.data_ptr()),
      reinterpret_cast<int*>(unpermuted_row_to_permuted_row.data_ptr()),
      reinterpret_cast<int*>(permuted_row_to_unpermuted_row.data_ptr()),
      reinterpret_cast<void*>(topk_ids.data_ptr()),
      topk_ids.scalar_type() == torch::kInt32,
      num_rows,
      TopK,
      total_experts_num,
      local_experts_num,
      queue);
}