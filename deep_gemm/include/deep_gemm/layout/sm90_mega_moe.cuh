#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/common/compile.cuh>

namespace deep_gemm::layout {

static constexpr uint32_t kSm90MXFP4MinL1SFProducerKBlocks = 48;

CUTLASS_HOST_DEVICE constexpr bool should_stage_sm90_mxfp4_weight_sf(
    const bool runs_linear1,
    const uint32_t hidden,
    const uint32_t block_k) {
    return not runs_linear1 or
           hidden / block_k >= kSm90MXFP4MinL1SFProducerKBlocks;
}

struct Sm90MoeWarpgroupLayout {
    bool split_n;
    uint32_t split_m, split_n_count;
    uint32_t block_m, block_n;
};

CUTLASS_HOST_DEVICE constexpr Sm90MoeWarpgroupLayout get_sm90_moe_warpgroup_layout(
    const uint32_t cta_block_m,
    const uint32_t cta_block_n,
    const uint32_t num_epilogue_warpgroups) {
    const bool split_n =
        cta_block_m == 64 and num_epilogue_warpgroups > 1 and
        cta_block_n % num_epilogue_warpgroups == 0 and
        (cta_block_n / num_epilogue_warpgroups == 64 or
         cta_block_n / num_epilogue_warpgroups == 128);
    const uint32_t split_m = split_n ? 1 : num_epilogue_warpgroups;
    const uint32_t split_n_count = split_n ? num_epilogue_warpgroups : 1;
    return {
        split_n,
        split_m,
        split_n_count,
        split_m == 0 ? 0 : cta_block_m / split_m,
        split_n_count == 0 ? 0 : cta_block_n / split_n_count,
    };
}

} // namespace deep_gemm::layout
