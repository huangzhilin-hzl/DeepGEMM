#pragma once

#include <cuda/std/cstdint>

#include <deep_gemm/common/compile.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout {

template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_sm90_num_padded_sf_pool_tokens(
    T num_max_pool_tokens, T block_m) {
    return get_num_sf_ring_tokens(num_max_pool_tokens, block_m);
}

// PR #383's two-launch SM90 kernel uses an arrival-count workspace ABI that is
// intentionally independent from main's ring-buffer MegaMoE workspace.
struct SM90Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;
    uint32_t num_max_pool_tokens;
    uint32_t num_max_pool_blocks;

    // Carry main's cache-line isolation into the SM90 full-pool ABI: barrier
    // counters stay in the first 32 bytes and expert counters start at 128B.
    static constexpr uint64_t kNumBarrierSignalBytes = 128;

    CUTLASS_HOST_DEVICE
    SM90Workspace(void* base,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(
            num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;
        num_bytes += num_experts * sizeof(uint64_t) * 2;
        num_bytes += num_experts_per_rank * sizeof(uint64_t);
        num_bytes += math::align(num_max_pool_blocks, 2u) * sizeof(uint32_t);
        num_bytes += num_max_pool_blocks * sizeof(uint64_t);
        num_bytes += num_experts_per_rank * num_ranks *
                     num_max_recv_tokens_per_expert * sizeof(int);
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);
        return math::align<uint64_t>(num_bytes, 16);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters,
                         "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(
            base,
            (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) +
            phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) +
               rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_arrival_count_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_l2_arrival_mask_ptr(const uint32_t& pool_block_idx = 0) const {
        const auto base = get_l1_arrival_count_ptr(math::align(num_max_pool_blocks, 2u));
        return reinterpret_cast<uint64_t*>(base) + pool_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0,
        const uint32_t& rank_idx = 0,
        const uint32_t& token_idx = 0) const {
        const auto base = get_l2_arrival_mask_ptr(num_max_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
               expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
               rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(
        const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(
            get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

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
