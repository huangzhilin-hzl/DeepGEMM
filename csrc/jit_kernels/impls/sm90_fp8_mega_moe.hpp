#pragma once

#include <mutex>
#include <unordered_set>

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

#include "../heuristics/sm90_mega_moe.hpp"

namespace deep_gemm {

// ============================================================================
// SM90 (Hopper) FP8-activation, MXFP4-weight MegaMoE host runtime
// ----------------------------------------------------------------------------
// This is the SM90 counterpart of `SM100FP8FP4MegaMoERuntime`. The kernel
// itself lives in `deep_gemm/impls/sm90_fp8_mega_moe.cuh` and uses the same
// dispatch/combine contract with an SM90 FP8 TMA/WGMMA implementation.
//
// Differences from SM100 path:
//   * Routed activations are FP8 (e4m3). Routed weights are packed MXFP4 and
//     expanded to FP8 in shared memory before Hopper WGMMA.
//   * Routed MXFP4 uses relative/rebased UE8M0 bytes at K32 granularity plus
//     one FP32 secondary scale per expert. Shared experts retain FP8 weights
//     with per-128-channel FP32 scales; neither uses the SM100 UTCCP layout.
//   * No tensor memory: WGMMA accumulators are register-resident.
//   * One CTA processes each work item; there is no cluster multicast or 2-CTA UMMA.
// ============================================================================

class SM90FP8MXFP4MegaMoERuntime final : public LaunchRuntime<SM90FP8MXFP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_shared_experts, num_topk;
        int num_ranks;
        int num_ring_tokens, num_sf_ring_tokens;
        float activation_clamp;
        bool fast_math;
        bool bf16_scaled_accum;
        MegaMoESM90Config config;

        // Runtime arguments. num_tokens also selects the canonical compile-time
        // MXFP4 scale-overlap bucket during generated-source construction.
        void* y;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormaps for activations and weights. The B producer can stage
        // MXFP4 relative-scale bytes with ordinary global/shared instructions
        // because each row exposes only a four-byte TMA box, illegal on SM90.
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        const float* l1_mxfp4_secondary;
        const uint8_t* l1_mxfp4_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        const float* l2_mxfp4_secondary;
        const uint8_t* l2_mxfp4_weights_sf;
        CUtensorMap tensor_map_shared_l1_acts;
        CUtensorMap tensor_map_shared_l1_acts_sf;
        CUtensorMap tensor_map_shared_l1_weights;
        const float* shared_l1_weights_sf;
        CUtensorMap tensor_map_shared_l1_output;
        CUtensorMap tensor_map_shared_l2_acts;
        CUtensorMap tensor_map_shared_l2_acts_sf;
        CUtensorMap tensor_map_shared_l2_weights;
        const float* shared_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const bool overlap_mxfp4_scale_path =
            args.hidden > 4096 or
            args.num_tokens <= kSM90MoeMaxLatencyOverlapTokens;
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_mxfp4_mega_moe_persistent_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {}
    >);
}};
)",
    args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.sf_pool_stride_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.num_sms, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.config.swap_ab ? "true" : "false",
    args.bf16_scaled_accum ? "true" : "false",
    overlap_mxfp4_scale_path ? "true" : "false",
    args.num_ring_tokens, args.num_sf_ring_tokens, args.num_shared_experts);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        const int num_threads =
            args.config.num_dispatch_threads +
            args.config.num_non_epilogue_threads +
            args.config.num_epilogue_threads;
        DG_HOST_ASSERT(num_threads == 256 and args.config.num_stages == 3 and
                       args.config.num_sms == 2 * device_runtime->get_num_sms() and
                       args.launch_args.grid_dim.first == args.config.num_sms);
        DG_HOST_ASSERT(device_runtime->get_prop()->cooperativeLaunch and
                       "SM90 MXFP4 MegaMoE requires cooperative launch support");

        // The JIT cache owns every loaded handle for process lifetime, so the
        // exact-kernel residency preflight is stable after its first successful
        // launch. Avoid repeating carveout and occupancy driver calls on every
        // decode forward while remaining host-thread safe.
        {
            static std::mutex preflight_mutex;
            static std::unordered_set<KernelHandle> preflighted_kernels;
            const std::lock_guard<std::mutex> lock(preflight_mutex);
            if (preflighted_kernels.count(kernel) == 0) {
                prefer_max_shared_memory_carveout(kernel);
                const int max_active_blocks = get_max_active_blocks_per_sm(
                    kernel, num_threads, args.config.smem_size);
                DG_HOST_ASSERT(max_active_blocks >= 2 and
                               "MXFP4 logical 2x-SM grid requires two resident CTAs per physical SM");
                DG_HOST_ASSERT(args.launch_args.grid_dim.first <=
                                   max_active_blocks * device_runtime->get_prop()->multiProcessorCount and
                               "SM90 MXFP4 MegaMoE cooperative grid is too large for this device");
                preflighted_kernels.insert(kernel);
            }
        }
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.l1_mxfp4_secondary,
            args.l1_mxfp4_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.l2_mxfp4_secondary,
            args.l2_mxfp4_weights_sf,
            args.tensor_map_shared_l1_acts,
            args.tensor_map_shared_l1_acts_sf,
            args.tensor_map_shared_l1_weights,
            args.shared_l1_weights_sf,
            args.tensor_map_shared_l1_output,
            args.tensor_map_shared_l2_acts,
            args.tensor_map_shared_l2_acts_sf,
            args.tensor_map_shared_l2_weights,
            args.shared_l2_weights_sf
        ));
    }
};

static void sm90_fp8_mxfp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_mxfp4_weights_sf, const torch::Tensor& l2_mxfp4_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const torch::Tensor& l1_mxfp4_secondary,
    const torch::Tensor& l2_mxfp4_secondary
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_padded_sf_pool_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto num_sf_ring_tokens = num_padded_sf_pool_tokens;
    const auto shared_intermediate_hidden =
        intermediate_hidden * num_shared_experts;
    DG_HOST_ASSERT(num_shared_experts >= 0);
    DG_HOST_ASSERT(num_shared_experts == 0 or
                   (shared_l1_acts.defined() and shared_l1_acts_sf.defined() and
                    shared_l2_acts.defined() and shared_l2_acts_sf.defined() and
                    shared_l1_weights.defined() and shared_l2_weights.defined() and
                    shared_l1_weights_sf.defined() and shared_l2_weights_sf.defined()));

    // Resolve the production MXFP4 schedule and numerical mode once. The
    // runtime only consumes the resulting complete persistent launch config.
    const int num_sms = device_runtime->get_num_sms();
    const Sm90MoeHeuristicInput heuristic_input {
        num_sms,
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden,
        num_padded_sf_pool_tokens
    };
    const auto config = select_mxfp4_mega_moe_sm90(heuristic_input);

    // Tensormap construction
    // Acts/weights: standard 2D TMA descriptors (FP8 K-major).
    // Activation SF: per-128 channel float for L1, per-64 for L2 (MN-major, no swizzle).
    // Routed weight SF: K32 relative UE8M0 bytes plus a per-expert FP32
    // secondary, both passed as ordinary pointers. Shared-expert weight SF is
    // a block-(128, 128) FP32 pointer.
    constexpr int kGranK = 128;
    constexpr int kL2ActsSFGranK = 64;
    // A BK256 pipeline stage is represented in shared memory as two adjacent
    // independently-swizzled BK128 TMA tiles. Keep the tensor-map box at 128
    // and issue two copies; the kernel config/scheduler still advances by 256.
    const int tma_block_k = std::min(config.block_k, kGranK);
    const int tma_block_n = std::min(config.block_n, 256);
    const int pool_tokens = num_ring_tokens;
    const int sf_stride_tokens = num_sf_ring_tokens;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, pool_tokens,
                                                     tma_block_k, config.block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     128);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        sf_stride_tokens, hidden,
                                                        config.block_m, kGranK,
                                                        1, 0);
    const auto tensor_map_l1_weights = make_tma_2d_desc(
        l1_weights,
        hidden / 2,
        num_experts_per_rank * intermediate_hidden * 2,
        tma_block_k / 2,
        tma_block_n,
        static_cast<int>(l1_weights.stride(-2)),
        64, 0, false, true, true);
    // L1 output (post-SwiGLU FP8): N is halved. The SM90 epilogue writes this
    // staging tile to SMEM as plain row-major bytes, so the TMA store descriptor
    // must use no shared-memory swizzle. Later L2 TMA loads may still swizzle
    // from this row-major global buffer into their own SMEM tile.
    // The default TMA store is issued per warpgroup, each writing a WG_BLOCK_M
    // row tile. In split-N mode, two WGs produce different N halves of the same
    // M rows, then one TMA store writes the full 64x128 post-SwiGLU tile.
    const int num_epilogue_warpgroups_h = config.num_epilogue_threads / 128;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups_h);
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    const int wg_l1_out_block_n = wg_block_n / 2;
    const int l1_output_box_n = wg_layout.split_n ? config.block_n / 2 : wg_l1_out_block_n;
    const int l1_output_box_m = wg_layout.split_n ? config.block_m : wg_block_m;
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, pool_tokens,
                                                       l1_output_box_n, l1_output_box_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       0);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, pool_tokens,
                                                     tma_block_k, config.block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     128);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        sf_stride_tokens, intermediate_hidden,
                                                        config.block_m, kL2ActsSFGranK,
                                                        1, 0);
    const auto tensor_map_l2_weights = make_tma_2d_desc(
        l2_weights,
        intermediate_hidden / 2,
        num_experts_per_rank * hidden,
        tma_block_k / 2,
        tma_block_n,
        static_cast<int>(l2_weights.stride(-2)),
        64, 0, false, true, true);

    // Shared experts retain Hopper's native FP8 K-major weight path. Their
    // FP32 block-(128, 128) weight scales stay in natural row-major storage
    // and are passed as ordinary pointers rather than TMA descriptors.
    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ?
        make_tma_2d_desc(
            shared_l1_acts,
            hidden, num_max_tokens_per_rank,
            tma_block_k, config.block_m,
            static_cast<int>(shared_l1_acts.stride(-2)),
            128) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ?
        make_tma_sf_desc(
            cute::UMMA::Major::MN, shared_l1_acts_sf,
            static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
            config.block_m, kGranK,
            1, 0) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ?
        make_tma_2d_desc(
            shared_l1_weights,
            hidden, shared_intermediate_hidden * 2,
            tma_block_k, tma_block_n,
            static_cast<int>(shared_l1_weights.stride(-2)),
            128) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ?
        make_tma_2d_desc(
            shared_l2_acts,
            shared_intermediate_hidden, num_max_tokens_per_rank,
            l1_output_box_n, l1_output_box_m,
            static_cast<int>(shared_l2_acts.stride(-2)),
            0) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ?
        make_tma_2d_desc(
            shared_l2_acts,
            shared_intermediate_hidden, num_max_tokens_per_rank,
            tma_block_k, config.block_m,
            static_cast<int>(shared_l2_acts.stride(-2)),
            128) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ?
        make_tma_sf_desc(
            cute::UMMA::Major::MN, shared_l2_acts_sf,
            static_cast<int>(shared_l2_acts_sf.size(0)),
            shared_intermediate_hidden,
            config.block_m, kL2ActsSFGranK,
            1, 0) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ?
        make_tma_2d_desc(
            shared_l2_weights,
            shared_intermediate_hidden, hidden,
            tma_block_k, tma_block_n,
            static_cast<int>(shared_l2_weights.stride(-2)),
            128) : tensor_map_l2_weights;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    // Launch
    constexpr bool bf16_scaled_accum = false;
    auto persistent_config = config;
    // The unified sizing already covers both logical phases. Add only the
    // persistent scheduler mailboxes/barriers here; the physical SF ring is the
    // descriptor stride used after wraparound.
    persistent_config.smem_size +=
        (4 + (num_shared_experts > 0 ? 2 : 0)) *
            static_cast<int>(sizeof(cutlass::arch::ClusterTransactionBarrier)) +
        2 * static_cast<int>(sizeof(sched::TaskInfo<true>));
    persistent_config.sf_pool_stride_tokens = num_sf_ring_tokens;

    const SM90FP8MXFP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts,
        .num_shared_experts = num_shared_experts,
        .num_topk = num_topk,
        .num_ranks = num_ranks,
        .num_ring_tokens = num_ring_tokens,
        .num_sf_ring_tokens = num_sf_ring_tokens,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .bf16_scaled_accum = bf16_scaled_accum,
        .config = persistent_config,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .l1_mxfp4_secondary = l1_mxfp4_secondary.data_ptr<float>(),
        .l1_mxfp4_weights_sf = l1_mxfp4_weights_sf.data_ptr<uint8_t>(),
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .l2_mxfp4_secondary = l2_mxfp4_secondary.data_ptr<float>(),
        .l2_mxfp4_weights_sf = l2_mxfp4_weights_sf.data_ptr<uint8_t>(),
        .tensor_map_shared_l1_acts = tensor_map_shared_l1_acts,
        .tensor_map_shared_l1_acts_sf = tensor_map_shared_l1_acts_sf,
        .tensor_map_shared_l1_weights = tensor_map_shared_l1_weights,
        .shared_l1_weights_sf = num_shared_experts > 0 ?
            shared_l1_weights_sf.data_ptr<float>() : nullptr,
        .tensor_map_shared_l1_output = tensor_map_shared_l1_output,
        .tensor_map_shared_l2_acts = tensor_map_shared_l2_acts,
        .tensor_map_shared_l2_acts_sf = tensor_map_shared_l2_acts_sf,
        .tensor_map_shared_l2_weights = tensor_map_shared_l2_weights,
        .shared_l2_weights_sf = num_shared_experts > 0 ?
            shared_l2_weights_sf.data_ptr<float>() : nullptr,
        .launch_args = LaunchArgs(
            persistent_config.num_sms,
            persistent_config.num_dispatch_threads +
                persistent_config.num_non_epilogue_threads +
                persistent_config.num_epilogue_threads,
            persistent_config.smem_size, 1, false, true)
    };
    // A live-block ring requires L1 production and L2 consumption to overlap
    // in one cooperative grid.
    const auto code = SM90FP8MXFP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build(
        "sm90_fp8_mxfp4_mega_moe_persistent_impl",
        code);
    SM90FP8MXFP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
