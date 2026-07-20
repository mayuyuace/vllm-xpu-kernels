#pragma once

#include <torch/all.h>

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
    double gemm1_clamp_limit);
