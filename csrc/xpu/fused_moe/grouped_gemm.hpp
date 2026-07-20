#pragma once

#include <torch/all.h>

#include <cute/tensor.hpp>
#include <random>

#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "xe2/xe2_policy.h"
  #include "xe2/xe2_utils.h"
#endif

#include "activation_utils.h"
#include "gemm_up.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;

// type tag to define a unique sycl kernel name
template <typename, typename, typename, typename, char, char, class>
class GemmCuteName;

template <
    char layoutA,
    char layoutB,
    class policy,
    ActivationType Activation,
    bool gemm1_clamp_limit_has_value,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
void MoEGEMMUpLauncher(
    sycl::queue& queue,
    ElementD* outputs,
    const ElementA* activations,
    const ElementB* weights,
    const ElementS* scales,
    const ElementBI* bias,
    const int* rows_per_expert,
    const int* permuted_row_to_unpermuted_row,
    const int num_experts,
    const int gemm_n,
    const int gemm_k,
    const double gemm1_clamp_limit,
    int32_t* atomic_buffer,
    const int group_size) {
  using ElementA_non_CV = cutlass::platform::remove_cv_t<ElementA>;
  auto op = XE_DPAS_TT<8, float, ElementA_non_CV>{};

  using WGTile = typename policy::WGTile;
  using SGLayout = typename policy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  auto MaxThreadsPerWorkgroup = size(MMA{});

  static constexpr int MaxThreadsPerSM = fused_moe::MaxThreadsPerSM;

  TORCH_CHECK(
      MaxThreadsPerSM % MaxThreadsPerWorkgroup == 0,
      "MaxThreadsPerSM must be divisible by MaxThreadsPerWorkgroup")

  sycl::range<3> local(1, 1, MaxThreadsPerWorkgroup);
  sycl::range<3> global(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroup, 1);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<fused_moe::sub_group_size>,
      intelex::grf_size<fused_moe::grf_size>};

  using GmemTiledCopyA = typename policy::GmemTiledCopyA;
  using GmemTiledCopyB = typename policy::GmemTiledCopyB;
  using GmemTiledCopyD = typename policy::GmemTiledCopyD;

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int32_t, 1> local_mem(sycl::range<1>(1), cgh);
    cgh.parallel_for<GemmCuteName<
        ElementA,
        ElementB,
        ElementS,
        ElementD,
        layoutA,
        layoutB,
        policy>>(
        sycl::nd_range<3>{global * local, local}, kernel_props, [=](auto) {
          fused_moe::MoEGEMMUp<
              GmemTiledCopyA,
              GmemTiledCopyB,
              GmemTiledCopyD,
              layoutA,
              layoutB,
              'R',
              MMA,
              Activation,
              gemm1_clamp_limit_has_value>(
              outputs,
              activations,
              weights,
              scales,
              bias,
              rows_per_expert,
              permuted_row_to_unpermuted_row,
              num_experts,
              gemm_n,
              gemm_k,
              gemm1_clamp_limit,
              atomic_buffer,
              group_size,
              local_mem);
        });
  });
}

void group_gemm_up_impl(
    torch::Tensor& output_up,
    torch::Tensor& hidden_states,
    torch::Tensor& w13,
    const c10::optional<at::Tensor>& w13_scale,
    const c10::optional<at::Tensor>& w13_bias,
    torch::Tensor& rows_per_expert,
    torch::Tensor& permuted_row_to_unpermuted_row,
    int64_t inter_size,
    int64_t hidden_size,
    ActivationType activation_type,
    const c10::optional<double>& gemm1_clamp_limit) {
  auto& dpcpp_queue =
      at::xpu::getCurrentXPUStream(hidden_states.device().index()).queue();
  auto A_dtype = hidden_states.dtype();
  auto B_dtype = w13.dtype();
  bool is_weight_fp8 =
      ((B_dtype == at::kFloat8_e4m3fn) || (B_dtype == at::kFloat8_e5m2));
  bool is_B_int4 = (B_dtype == at::kChar) && w13_scale.has_value();
  bool is_B_mxfp4 = (B_dtype == at::kFloat4_e2m1fn_x2) && w13_scale.has_value();

  int N = inter_size;
  int K = hidden_size;

  TORCH_CHECK(N % 8 == 0, "inter_size must be divisible by 8");

  TORCH_CHECK(
      hidden_states.dim() == 2,
      "hidden_states must be 2D [Total_M, hidden_size]");
  TORCH_CHECK(
      w13.dim() == 3,
      "w13 must be 3D [num_experts, hidden_size, 2*inter_size]");
  if (w13_bias.has_value()) {
    TORCH_CHECK(
        w13_bias->dim() == 2,
        "w13_bias must be 2D [num_experts, 2*inter_size]");
  }

  TORCH_CHECK(
      hidden_states.is_contiguous(), "hidden_states must be contiguous");
  TORCH_CHECK(w13.is_contiguous(), "w13 must be contiguous");
  if (w13_bias.has_value()) {
    TORCH_CHECK(w13_bias->is_contiguous(), "w13_bias must be contiguous");
  }

  int A_total_M = hidden_states.size(0);
  int A_K = K;

  int B_E = w13.size(0);
  int B_K = K;
  int B_N = N;

  int group_size = -1;
  int A_avg_M = A_total_M / B_E;

  at::Tensor atomic_buffer = at::empty(
      {static_cast<long>(1)}, hidden_states.options().dtype(at::kInt));

#define MoEGEMMLauncherCallER(                                                 \
    LayoutA,                                                                   \
    LayoutB,                                                                   \
    Policy,                                                                    \
    Activation,                                                                \
    HasClamp,                                                                  \
    ElementA,                                                                  \
    ElementB,                                                                  \
    ElementS)                                                                  \
  MoEGEMMUpLauncher<LayoutA, LayoutB, Policy, Activation, HasClamp>(           \
      dpcpp_queue,                                                             \
      reinterpret_cast<ElementA*>(output_up.data_ptr()),                       \
      reinterpret_cast<ElementA*>(hidden_states.data_ptr()),                   \
      reinterpret_cast<ElementB*>(w13.data_ptr()),                             \
      w13_scale.has_value()                                                    \
          ? reinterpret_cast<ElementS*>(w13_scale->data_ptr())                 \
          : static_cast<ElementS*>(nullptr),                                   \
      w13_bias.has_value() ? reinterpret_cast<ElementA*>(w13_bias->data_ptr()) \
                           : static_cast<ElementA*>(nullptr),                  \
      reinterpret_cast<int*>(rows_per_expert.data_ptr()),                      \
      reinterpret_cast<int*>(permuted_row_to_unpermuted_row.data_ptr()),       \
      num_experts,                                                             \
      N,                                                                       \
      K,                                                                       \
      gemm1_clamp_limit.has_value() ? *gemm1_clamp_limit : 0.0f,               \
      static_cast<int*>(atomic_buffer.data_ptr()),                             \
      group_size);

// Dispatch the runtime clamp optional into the compile-time bool template arg.
#define MoEGEMMLauncherCallER_CLAMP(                                    \
    LayoutA, LayoutB, Policy, Activation, ElementA, ElementB, ElementS) \
  if (gemm1_clamp_limit.has_value()) {                                  \
    MoEGEMMLauncherCallER(                                              \
        LayoutA,                                                        \
        LayoutB,                                                        \
        Policy,                                                         \
        Activation,                                                     \
        true,                                                           \
        ElementA,                                                       \
        ElementB,                                                       \
        ElementS);                                                      \
  } else {                                                              \
    MoEGEMMLauncherCallER(                                              \
        LayoutA,                                                        \
        LayoutB,                                                        \
        Policy,                                                         \
        Activation,                                                     \
        false,                                                          \
        ElementA,                                                       \
        ElementB,                                                       \
        ElementS);                                                      \
  }

// Dispatch the runtime activation_type into the compile-time template argument.
#define MoEGEMMLauncherCallER_ACT(                                             \
    LayoutA, LayoutB, Policy, ElementA, ElementB, ElementS)                    \
  switch (activation_type) {                                                   \
    case ActivationType::SILU:                                                 \
      MoEGEMMLauncherCallER_CLAMP(                                             \
          LayoutA,                                                             \
          LayoutB,                                                             \
          Policy,                                                              \
          ActivationType::SILU,                                                \
          ElementA,                                                            \
          ElementB,                                                            \
          ElementS);                                                           \
      break;                                                                   \
    case ActivationType::GELU:                                                 \
      MoEGEMMLauncherCallER_CLAMP(                                             \
          LayoutA,                                                             \
          LayoutB,                                                             \
          Policy,                                                              \
          ActivationType::GELU,                                                \
          ElementA,                                                            \
          ElementB,                                                            \
          ElementS);                                                           \
      break;                                                                   \
    case ActivationType::GELU_TANH:                                            \
      MoEGEMMLauncherCallER_CLAMP(                                             \
          LayoutA,                                                             \
          LayoutB,                                                             \
          Policy,                                                              \
          ActivationType::GELU_TANH,                                           \
          ElementA,                                                            \
          ElementB,                                                            \
          ElementS);                                                           \
      break;                                                                   \
    case ActivationType::SWIGLUOAI:                                            \
      MoEGEMMLauncherCallER_CLAMP(                                             \
          LayoutA,                                                             \
          LayoutB,                                                             \
          Policy,                                                              \
          ActivationType::SWIGLUOAI,                                           \
          ElementA,                                                            \
          ElementB,                                                            \
          ElementS);                                                           \
      break;                                                                   \
    case ActivationType::RELU2_NO_MUL:                                         \
      MoEGEMMLauncherCallER_CLAMP(                                             \
          LayoutA,                                                             \
          LayoutB,                                                             \
          Policy,                                                              \
          ActivationType::RELU2_NO_MUL,                                        \
          ElementA,                                                            \
          ElementB,                                                            \
          ElementS);                                                           \
      break;                                                                   \
    case ActivationType::SWIGLUSTEP:                                           \
      MoEGEMMLauncherCallER_CLAMP(                                             \
          LayoutA,                                                             \
          LayoutB,                                                             \
          Policy,                                                              \
          ActivationType::SWIGLUSTEP,                                          \
          ElementA,                                                            \
          ElementB,                                                            \
          ElementS);                                                           \
      break;                                                                   \
    default:                                                                   \
      TORCH_CHECK(false, "Unsupported activation type for fused MoE up gemm"); \
  }

#define W16A16LauncherCallER(policy)                                     \
  if (A_dtype == at::kBFloat16) {                                        \
    MoEGEMMLauncherCallER_ACT(                                           \
        'R', 'R', policy, bfloat16_t, bfloat16_t, bfloat16_t);           \
  } else if (A_dtype == at::kHalf) {                                     \
    MoEGEMMLauncherCallER_ACT('R', 'R', policy, half_t, half_t, half_t); \
  }

  using policy = w16a16_policy;
  W16A16LauncherCallER(policy);
}

void group_gemm_down_impl(
    torch::Tensor& output,
    torch::Tensor& w2,
    const c10::optional<at::Tensor>& w2_scale,
    const c10::optional<at::Tensor>& w2_bias,
    torch::Tensor& topk_weights,
    torch::Tensor& rows_per_expert,
    torch::Tensor& unpermuted_row_to_permuted_row,
    int64_t inter_size,
    int64_t hidden_size) {}
