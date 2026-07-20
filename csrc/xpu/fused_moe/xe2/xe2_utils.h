#pragma once

namespace fused_moe {
inline static constexpr int MaxThreadsPerSM = 512;
inline static constexpr int sub_group_size = 16;
inline static constexpr int grf_size = 256;
}  // namespace fused_moe
