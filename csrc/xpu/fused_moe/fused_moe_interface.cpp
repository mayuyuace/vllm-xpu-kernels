#include "fused_moe_interface.h"
#include "remap_hidden_states_index.hpp"
#include "grouped_gemm.hpp"

void fused_moe_interface(
    torch::Tensor& output,
    torch::Tensor& hidden_states,
    torch::Tensor& w13,
    const c10::optional<at::Tensor>& w13_scale,
    const c10::optional<at::Tensor>& w13_bias,
    torch::Tensor& w2,
    const c10::optional<at::Tensor>& w2_scale,
    const c10::optional<at::Tensor>& w2_bias,
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weights,
    const c10::optional<torch::Tensor>& expert_map,
    int64_t total_experts_num,
    int64_t local_experts_num,
    int64_t inter_size,
    int64_t hidden_size,
    const std::string& activation,
    const c10::optional<double>& gemm1_clamp_limit) {
  auto activation_type = ActivationType::NONE;
  if (activation == "silu") {
    activation_type = ActivationType::SILU;
  } else if (activation == "gelu") {
    activation_type = ActivationType::GELU;
  } else if (activation == "gelu_tanh") {
    activation_type = ActivationType::GELU_TANH;
  } else if (activation == "swigluoai") {
    activation_type = ActivationType::SWIGLUOAI;
  } else if (activation == "relu2_no_mul") {
    activation_type = ActivationType::RELU2_NO_MUL;
  } else if (activation == "swiglustep") {
    activation_type = ActivationType::SWIGLUSTEP;
  } else {
    TORCH_CHECK(false, "Unsupported activation type: ", activation);
  }

  // step 1 remap_hidden_states_index
  at::Tensor rows_per_expert =
      at::zeros({local_experts_num}, topk_ids.options().dtype(at::kInt));
  at::Tensor unpermuted_row_to_permuted_row = at::empty(
      {topk_ids.size(0), topk_ids.size(1)}, topk_ids.options().dtype(at::kInt));
  at::Tensor permuted_row_to_unpermuted_row = at::empty(
      {topk_ids.size(0) * topk_ids.size(1)},
      topk_ids.options().dtype(at::kInt));

  remap_hidden_states_index(
      expert_map,
      rows_per_expert,
      unpermuted_row_to_permuted_row,
      permuted_row_to_unpermuted_row,
      topk_ids,
      total_experts_num,
      local_experts_num);

  // step 2 group_gemm_up
  at::Tensor output_up =
      at::empty({hidden_states.size(0), inter_size}, hidden_states.options());
  group_gemm_up_impl(
      output_up,
      hidden_states,
      w13,
      w13_scale,
      w13_bias,
      rows_per_expert,
      permuted_row_to_unpermuted_row,
      inter_size,
      hidden_size,
      activation_type,
      gemm1_clamp_limit);

  // step 3 group_gemm_down
  group_gemm_down_impl(
      output,
      w2,
      w2_scale,
      w2_bias,
      topk_weights,
      rows_per_expert,
      unpermuted_row_to_permuted_row,
      inter_size,
      hidden_size);
}
