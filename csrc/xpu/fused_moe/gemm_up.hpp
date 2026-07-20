#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/platform/platform.h"
// #include "gemm.hpp"
#include <cute/util/compat.hpp>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace fused_moe {
using namespace cute;

template <typename T, char LayoutKind>
CUTE_DEVICE auto make_moe_tensor(T* ptr, int r, int c) {
  auto shape = make_shape(r, c);
  if constexpr (LayoutKind == 'C')
    return make_tensor(
        make_gmem_ptr(ptr), make_layout(shape, make_stride(_1{}, r)));
  else
    return make_tensor(
        make_gmem_ptr(ptr), make_layout(shape, make_stride(c, _1{})));
}

template <typename T>
inline T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + sycl::exp((float)-x)));
}

template <typename T>
inline T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + sycl::erf(f * ALPHA)));
}

template <typename T>
inline T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + sycl::tanh(inner)));
}

template <typename T>
inline T relu2_no_mul_kernel(const T& x) {
  // square(relu(x))
  const float f = (float)x;
  const float r = f > 0.0f ? f : 0.0f;
  return (T)(r * r);
}

template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) T
swigluoai_and_mul(const T& gate, const T& up, float alpha, float limit) {
  // clamp gate: min=None, max=limit
  const float gate_f = (float)gate;
  const float clamped_gate = gate_f > limit ? limit : gate_f;

  // clamp up: min=-limit, max=limit
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  // glu = gate * sigmoid(gate * alpha)
  const float sigmoid_val = 1.0f / (1.0f + sycl::exp(-clamped_gate * alpha));
  const float glu = clamped_gate * sigmoid_val;

  // (up + 1) * glu
  return (T)((clamped_up + 1.0f) * glu);
}

template <typename T>
[[intel::device_indirectly_callable]] inline __attribute__((always_inline)) T
swiglustep_and_mul(const T& gate, const T& up, float limit) {
  // gate = silu(gate).clamp(max=limit)
  const float gate_f = (float)gate;
  const float silu_gate = gate_f / (1.0f + sycl::exp(-gate_f));
  const float clamped_gate = silu_gate > limit ? limit : silu_gate;

  // up = up.clamp(min=-limit, max=limit)
  const float up_f = (float)up;
  const float clamped_up =
      up_f > limit ? limit : (up_f < -limit ? -limit : up_f);

  return (T)(clamped_gate * clamped_up);
}

template <ActivationType activation_type>
inline float activate_func(float gate, float up) {
  if (activation_type == ActivationType::SILU) {
    return silu_kernel(gate) * up;
  } else if (activation_type == ActivationType::GELU) {
    return gelu_kernel(gate) * up;
  } else if (activation_type == ActivationType::GELU_TANH) {
    return gelu_tanh_kernel(gate) * up;
  } else if (activation_type == ActivationType::SWIGLUOAI) {
    return swigluoai_and_mul(gate, up, 1.702, 7.0);
  } else if (activation_type == ActivationType::RELU2_NO_MUL) {
    return relu2_no_mul_kernel(gate) * up;
  } else if (activation_type == ActivationType::SWIGLUSTEP) {
    return swiglustep_and_mul(gate, up, 7.0);
  } else {
    return gate * up;
  }
}

template <
    ActivationType Activation,
    bool gemm1_clamp_limit_has_value,
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyC,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA,
    typename ElementS,
    typename ElementBI>
CUTE_DEVICE void gemm_up(
    ATensor const& A,   // (M,K)
    BTensor const& B1,  // (N,K)
    BTensor const& B2,  // (N,K)
    const ElementS* Scales,
    const ElementBI* Bias,
    DTensor& C,  // (M,N)
    Coord<int, int, cute::Underscore, int> blk_coord,
    TiledMMA const& mma,
    float gemm1_clamp_limit) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto wg_m = get<0>(blk_coord);
  auto wg_n = get<1>(blk_coord);
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B1.shape());
  Tensor cC = make_identity_tensor(C.shape());

  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N)

  auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(mma, A);
  auto copy_b1 = get_block_2d_copy_B<GmemTiledCopyB>(mma, B1);
  auto copy_b2 = get_block_2d_copy_B<GmemTiledCopyB>(mma, B2);
  auto copy_c = get_block_2d_copy_D<GmemTiledCopyC>(mma, C);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b1 = copy_b1.get_slice(local_id);
  auto thr_copy_b2 = copy_b2.get_slice(local_id);
  auto thr_copy_c = copy_c.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB1 = thr_copy_b1.partition_sg_fragment_D(gB(_, _, 0));
  auto tBrB2 = thr_copy_b2.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB1 = thr_copy_b1.partition_S(gB);
  Tensor tBgB2 = thr_copy_b2.partition_S(gB);

  /* Partition C */
  auto tCrC1 = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC2 = thr_mma.partition_sg_fragment_C(gC);
  auto tCrC_out = thr_copy_c.partition_sg_fragment_S(gC);
  auto tCgC = thr_copy_c.partition_D(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b1 = make_block_2d_prefetch(copy_b1);
  auto prefetch_b2 = make_block_2d_prefetch(copy_b2);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B1 = prefetch_b1.get_slice(local_id);
  auto thr_prefetch_B2 = prefetch_b2.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB1 = thr_prefetch_B1.partition_S(gB);
  auto pBgB2 = thr_prefetch_B2.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrC1);
  clear(tCrC2);

  using ElementB = typename BTensor::element_type;
  static constexpr bool is_B_fp8_type =
      std::is_same_v<ElementB, cutlass::float_e5m2_t> ||
      std::is_same_v<ElementB, cutlass::float_e4m3_t>;

  load_A_to_SLM();

  CUTLASS_PRAGMA_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_b1, pBgB1(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b2, pBgB2(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    // copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    // reorder(tArA, tCrA);
    copy(copy_slm, tAslm(_, _, _, k_tile), tArA);
    reorder(tArA, tCrA);

    copy(copy_b1, tBgB1(_, _, _, k_tile), tBrB1);
    reorder(tBrB1, tCrB1);
    cute::gemm(mma, tCrA, tCrB1, tCrC1);

    copy(copy_b2, tBgB2(_, _, _, k_tile), tBrB2);
    reorder(tBrB2, tCrB2);
    cute::gemm(mma, tCrA, tCrB2, tCrC2);

    if (k_tile_prefetch % 8 == 0) {
      load_A_to_SLM();
    }

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_b2, pBgB2(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b1, pBgB1(_, _, _, k_tile_prefetch));
    }
    barrier_wait(barrier_scope);
  }

  if constexpr (is_B_fp8_type) {
    float B_scale = Scales[0];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC1.size(); ++i) {
      tCrC1(i) *= B_scale;
      tCrC2(i) *= B_scale;
    }
  }

  if (Bias1 != nullptr) {
    static constexpr auto ATOM_M =
        get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N =
        get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

    auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;

    static constexpr auto tile_m = get<0>(wg_tile);
    static constexpr auto tile_n = get<1>(wg_tile);

    // 32 * 64
    static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
    static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

    int sg_local_id = cutlass::get_sub_group_local_id();
    static constexpr int sg_local_range = 16;

    int n_tile_start = wg_n * tile_n;
    int n_sg_start = sg_local_n_coord * SG_N;

    CUTLASS_PRAGMA_UNROLL
    for (int sn = 0; sn < SG_N / sg_local_range; ++sn) {
      int sg_local_n = sn * sg_local_range + sg_local_id;
      float b_float1 = Bias1[n_tile_start + n_sg_start + sg_local_n];
      float b_float2 = Bias2[n_tile_start + n_sg_start + sg_local_n];
      CUTLASS_PRAGMA_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        tCrC1(sn * SG_M + sm) += b_float1;
        tCrC2(sn * SG_M + sm) += b_float2;
      }
    }
  }

  if constexpr (gemm1_clamp_limit_has_value) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC1.size(); ++i) {
      tCrC1(i) = sycl::min(tCrC1(i), gemm1_clamp_limit);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tCrC2.size(); ++i) {
      tCrC2(i) = sycl::max(tCrC2(i), -gemm1_clamp_limit);
      tCrC2(i) = sycl::min(tCrC2(i), gemm1_clamp_limit);
    }
  }

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < tCrC1.size(); ++i) {
    tCrC1(i) = activate_func<Activation>(tCrC1(i), tCrC2(i));
  }

  reorder(tCrC1, tCrC_out);
  copy(copy_c, tCrC_out, tCgC);
}

template <
    class GmemTiledCopyA,
    class GmemTiledCopyB,
    class GmemTiledCopyD,
    char LayoutKindA,
    char LayoutKindB,
    char LayoutKindD,
    class TiledMMA,
    ActivationType Activation,
    bool gemm1_clamp_limit_has_value,
    typename ElementA,
    typename ElementB,
    typename ElementS,
    typename ElementBI,
    typename ElementD>
CUTE_DEVICE void MoEGEMMUp(
    ElementD* Outputs,
    const ElementA* Activations,
    const ElementB* Weights,
    const ElementS* Scales,
    const ElementBI* Bias,
    const int* rows_per_expert,
    const int* permuted_row_to_unpermuted_row,
    const int32_t num_experts,
    const int32_t gemm_n,
    const int32_t gemm_k,
    double gemm1_clamp_limit,
    int32_t* atomic_buffer,
    const int32_t group_size,
    const sycl::local_accessor<int32_t, 1>& slm_mem_const) {
  constexpr char actual_layout_of_B = LayoutKindB ^ ('R' ^ 'C');
  static constexpr bool is_B_int4 = (std::is_same_v<ElementB, uint8_t>) &&
                                    (!std::is_same_v<ElementS, uint8_t>);
  static constexpr bool is_B_mxfp4 = (std::is_same_v<ElementB, uint8_t>) &&
                                     (std::is_same_v<ElementS, uint8_t>);
  static constexpr bool is_B_4bits = std::is_same_v<ElementB, uint8_t>;

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  TiledMMA mma;
  auto wg_tile = mma.tile_mnk();
  auto wg_tile_m = get<0>(wg_tile);
  auto wg_tile_n = get<1>(wg_tile);

  int group_id = item.get_group_linear_id();
  int gemm_n_pad = (gemm_n + wg_tile_n - 1) / wg_tile_n * wg_tile_n;
  int group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
  int group_range = item.get_group_range(1);
  int local_id = item.get_local_linear_id();

  if (group_id == 0 && local_id == 0) {
    auto atm = sycl::atomic_ref<
        int,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>(atomic_buffer[0]);
    atm.store(0);
  }

  int pre_rows = 0;
  int pre_tiles = 0;

  int32_t* slm_mem = static_cast<int32_t*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());

  for (int i = 0; i < num_experts; ++i) {
    int gemm_m = rows_per_expert[i];
    int cumsum_rows_for_experts = pre_rows + gemm_m;
    int cumsum_tiles_for_experts =
        (gemm_m + wg_tile_m - 1) / wg_tile_m + pre_tiles;

    if (group_m_id >= cumsum_tiles_for_experts) {
      pre_rows = cumsum_rows_for_experts;
      pre_tiles = cumsum_tiles_for_experts;
      continue;
    }

    int expert_id = i;
    int64_t B_offset = static_cast<int64_t>(expert_id) *
                       static_cast<int64_t>(gemm_n) *
                       static_cast<int64_t>(gemm_k) * 2;
    if constexpr (is_B_4bits) {
      B_offset /= 2;
    }
    ElementA* ptr_A_curr_batch =
        const_cast<ElementA*>(Activations) + pre_rows * gemm_k;
    ElementB* ptr_w1_curr_batch = const_cast<ElementB*>(Weights) + B_offset;
    ElementB* ptr_w3_curr_batch =
        const_cast<ElementB*>(Weights) + B_offset + gemm_n;
    ElementD* ptr_D_curr_batch = Outputs + pre_rows * gemm_n;
    ElementS* ptr_w1_Scales_curr_batch =
        const_cast<ElementS*>(Scales) + expert_id;
    ElementS* ptr_w3_Scales_curr_batch = ptr_w1_Scales_curr_batch + gemm_n;
    if constexpr (is_B_4bits) {
      ptr_w1_Scales_curr_batch =
          const_cast<ElementS*>(Scales) + B_offset * 2 / group_size;
      ptr_w3_Scales_curr_batch =
          ptr_w1_Scales_curr_batch + gemm_n * 2 / group_size;
    }
    ElementBI* ptr_w1_Bias_curr_batch = nullptr;
    ElementBI* ptr_w3_Bias_curr_batch = nullptr;
    if (Bias != static_cast<ElementBI*>(nullptr)) {
      ptr_w1_Bias_curr_batch =
          const_cast<ElementBI*>(Bias) + expert_id * gemm_n * 2;
      ptr_w3_Bias_curr_batch = ptr_w1_Bias_curr_batch + gemm_n;
    }

    auto A_tensor = make_moe_tensor<ElementA, LayoutKindA>(
        ptr_A_curr_batch, gemm_m, gemm_k);
    auto w13_layout = [&]() {
      if constexpr (actual_layout_of_B == 'C') {
        return make_layout(
            make_shape(gemm_n, gemm_k), make_stride(_1{}, gemm_n * 2));
      } else {
        return make_layout(
            make_shape(gemm_n, gemm_k), make_stride(gemm_k, _1{}));
      }
    }();
    auto B1_tensor = make_tensor(make_gmem_ptr(ptr_w1_curr_batch), w13_layout);
    auto B2_tensor = make_tensor(make_gmem_ptr(ptr_w3_curr_batch), w13_layout);
    auto D_tensor = make_moe_tensor<ElementD, LayoutKindD>(
        ptr_D_curr_batch, gemm_m, gemm_n);

    while (group_m_id < cumsum_tiles_for_experts) {
      int n_coord = (group_id * wg_tile_n) % gemm_n_pad / wg_tile_n;
      int m_coord = (group_m_id - pre_tiles);
      auto tile_coord = make_coord(m_coord, n_coord, _, 0);

      if constexpr (is_B_4bits) {
      } else {
        gemm_up<
            Activation,
            gemm1_clamp_limit_has_value,
            GmemTiledCopyA,
            GmemTiledCopyB,
            GmemTiledCopyD>(
            A_tensor,
            B1_tensor,
            B2_tensor,
            ptr_w1_Scales_curr_batch,
            ptr_w1_Bias_curr_batch,
            D_tensor,
            tile_coord,
            mma,
            gemm1_clamp_limit);
      }

      if (local_id == 0) {
        slm_mem[0] = cutlass::atomicAdd(atomic_buffer, 1);
      }
      item.barrier(sycl::access::fence_space::local_space);
      group_id = group_range + slm_mem[0];
      group_m_id = (group_id * wg_tile_n) / gemm_n_pad;
    }
    pre_rows = cumsum_rows_for_experts;
    pre_tiles = cumsum_tiles_for_experts;
  }
}

}  // namespace fused_moe
