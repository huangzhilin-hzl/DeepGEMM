#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <pybind11/functional.h>

#include <deep_gemm/layout/sym_buffer.cuh>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/impls/sm90_fp8_mega_moe.hpp"
#include "../utils/layout.hpp"
#include "../utils/system.hpp"

namespace deep_gemm::mega {

static constexpr int kSM90MegaMoECandidateBlockMs[] = {64, 128};
static constexpr int kSM90MegaMoETokenAlignment = 128;
// The dispatch-side NVLink barrier assigns one signaling thread per rank and
// the compact MXFP4 frontend has 64 dispatch threads.
static constexpr int kSM90MegaMoEMaxRanks = 64;

static bool is_valid_hidden_for_sm90_mega_moe(const int hidden) {
    // The combine epilogue uses four chunks above 8192 elements, with each
    // chunk vectorized in groups of 256 BF16 values.
    return hidden % 512 == 0 and (hidden <= 8192 or hidden % 1024 == 0);
}

static int get_token_alignment_for_sm90_mega_moe() {
    return kSM90MegaMoETokenAlignment;
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    DG_HOST_ASSERT(device_runtime->get_arch_major() == 9);
    DG_HOST_ASSERT(num_ranks > 0 and num_ranks <= kSM90MegaMoEMaxRanks);
    DG_HOST_ASSERT(num_experts > 0);
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank % kSM90MegaMoETokenAlignment == 0);
    DG_HOST_ASSERT(num_topk > 0 and num_topk <= std::min(num_experts, 32));
    DG_HOST_ASSERT(hidden > 0 and intermediate_hidden > 0);
    // Input/L1 K128 FP32 SF rows occupy H/32 bytes and must retain 16-byte
    // TMA alignment. L2 K64 FP32 SF rows occupy I/16 bytes.
    DG_HOST_ASSERT(is_valid_hidden_for_sm90_mega_moe(hidden) and
                   intermediate_hidden % 256 == 0);
    DG_HOST_ASSERT(use_fp8_dispatch);
    DG_HOST_ASSERT(activation == "swiglu");

    // Workspace bytes
    const auto workspace = layout::Workspace(nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);

    // Layouts
    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    const auto fp8_sf_layout = layout::Data(hidden / 32);
    // SM90 L2 activations use one float SF per 64 K elements.
    const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden / 16);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // Buffer configs
    const auto num_max_pool_tokens = static_cast<int>(workspace.num_max_pool_tokens);
    int num_max_padded_sf_pool_tokens = 0;
    for (int block_m: kSM90MegaMoECandidateBlockMs) {
        num_max_padded_sf_pool_tokens = std::max(
            num_max_padded_sf_pool_tokens,
            layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m)
        );
    }

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_max_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_max_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens for cross-rank combine
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());

    // Slice function: creates `(x, x_sf, topk_idx, topk_weights, l1_acts,
    // l1_acts_sf, l2_acts, l2_acts_sf)` views from the raw buffer.
    // NOTES: `x_sf` is K-major, while `l1_acts_sf` and `l2_acts_sf` are M-major
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_max_pool_tokens, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, hidden / 128},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, intermediate_hidden / 64},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr()), slice_input_buffers};
}

// SM90 (Hopper) FP8-activation MegaMoE entry point.
//
// Shared validation and dispatch for packed MXFP4 weights with K32 UE8M0
// scales. Top-level routing is the caller's responsibility (see
// `deep_gemm/mega/__init__.py`).
static void sm90_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math,
    const std::optional<torch::Tensor>& l1_mxfp4_secondary = std::nullopt,
    const std::optional<torch::Tensor>& l2_mxfp4_secondary = std::nullopt
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;
    const bool processed_mxfp4_scales = l1_mxfp4_secondary.has_value();
    DG_HOST_ASSERT(processed_mxfp4_scales == l2_mxfp4_secondary.has_value());

    // Architecture check
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 9);

    // MXFP4 weights use K32 UE8M0 SF. Activations use per-token per-128-K
    // float SF.
    DG_HOST_ASSERT(y.is_cuda());
    DG_HOST_ASSERT(y.dim() == 2);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(y.is_contiguous());
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    DG_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DG_HOST_ASSERT(activation == "swiglu");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks: MXFP4 follows Humming's native packed-E2M1 contract:
    // int8 [E, N, K/2] plus natural-layout uint8 UE8M0 scales [E, N, K/32].
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(l1_weights.scalar_type() == kPackedFP4);
    DG_HOST_ASSERT(l2_weights.scalar_type() == kPackedFP4);
    const auto [num_experts_per_rank, intermediate_hidden_2, l1_stored_k] =
        get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, l2_stored_k] =
        get_shape<3>(l2_weights);
    const int hidden = static_cast<int>(l1_stored_k) * 2;
    const int intermediate_hidden = static_cast<int>(l2_stored_k) * 2;
    DG_HOST_ASSERT(y.size(1) == hidden);
    DG_HOST_ASSERT(num_tokens >= 0);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());
    DG_HOST_ASSERT(l1_weights_sf.is_cuda() and l2_weights_sf.is_cuda());
    DG_HOST_ASSERT(l1_weights.device() == y.device() and
                   l2_weights.device() == y.device() and
                   l1_weights_sf.device() == y.device() and
                   l2_weights_sf.device() == y.device());

    // Keep host validation aligned with the default TMA-aligned Data layouts
    // reconstructed by the generated SM90 kernel.
    DG_HOST_ASSERT(is_valid_hidden_for_sm90_mega_moe(hidden) and
                   intermediate_hidden % 256 == 0);

    // Check weight SF layout. SF is not TMA-loaded, so no TMA-stride alignment
    // is required; the K direction must still be contiguous within each expert.
    DG_HOST_ASSERT(l1_weights_sf.scalar_type() == torch::kUInt8 and
                   l2_weights_sf.scalar_type() == torch::kUInt8);
    DG_HOST_ASSERT(l1_weights_sf.is_contiguous() and
                   l2_weights_sf.is_contiguous());
    DG_HOST_ASSERT(l1_weights_sf.dim() == 3 and
                   l1_weights_sf.size(0) == num_experts_per_rank and
                   l1_weights_sf.size(1) == intermediate_hidden * 2 and
                   l1_weights_sf.size(2) == hidden / 32);
    DG_HOST_ASSERT(l2_weights_sf.dim() == 3 and
                   l2_weights_sf.size(0) == num_experts_per_rank and
                   l2_weights_sf.size(1) == hidden and
                   l2_weights_sf.size(2) == intermediate_hidden / 32);
    if (processed_mxfp4_scales) {
        DG_HOST_ASSERT(l1_mxfp4_secondary->is_cuda() and
                       l2_mxfp4_secondary->is_cuda());
        DG_HOST_ASSERT(l1_mxfp4_secondary->device() == y.device() and
                       l2_mxfp4_secondary->device() == y.device());
        DG_HOST_ASSERT(l1_mxfp4_secondary->scalar_type() == torch::kFloat and
                       l2_mxfp4_secondary->scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(l1_mxfp4_secondary->is_contiguous() and
                       l2_mxfp4_secondary->is_contiguous());
        DG_HOST_ASSERT(l1_mxfp4_secondary->dim() == 1 and
                       l1_mxfp4_secondary->size(0) == num_experts_per_rank);
        DG_HOST_ASSERT(l2_mxfp4_secondary->dim() == 1 and
                       l2_mxfp4_secondary->size(0) == num_experts_per_rank);
    }

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda());
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == y.device());
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() ==
                       num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_ranks > 0 and num_ranks <= kSM90MegaMoEMaxRanks);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(sym_buffer.is_cuda());
    DG_HOST_ASSERT(sym_buffer.device() == y.device());
    DG_HOST_ASSERT(sym_buffer.scalar_type() == torch::kChar);
    DG_HOST_ASSERT(sym_buffer.is_contiguous());
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank % kSM90MegaMoETokenAlignment == 0);
    DG_HOST_ASSERT(num_experts > 0 and num_topk > 0 and
                   num_topk <= std::min(num_experts, 32));
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    DG_HOST_ASSERT(num_experts == num_experts_);
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_sm90_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    sm90_fp8_mega_moe(y,
                     l1_acts, l1_acts_sf,
                     l2_acts, l2_acts_sf,
                     l1_weights, l2_weights,
                     l1_weights_sf, l2_weights_sf,
                     cumulative_local_expert_recv_stats,
                     sym_buffer_ptrs,
                     rank_idx, num_max_tokens_per_rank,
                     num_experts_per_rank,
                     num_tokens, num_topk,
                     hidden, intermediate_hidden,
                     activation_clamp, fast_math, true,
                     processed_mxfp4_scales,
                     l1_mxfp4_secondary, l2_mxfp4_secondary);

    if (get_env<int>("DG_COMM_KERNEL_DEBUG"))
        sym_buffer.zero_();
}

static void fp8_mxfp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math) {
    sm90_mega_moe(
        y, l1_weights_tuple, l2_weights_tuple,
        cumulative_local_expert_recv_stats,
        sym_buffer, sym_buffer_ptrs, rank_idx,
        num_max_tokens_per_rank, num_experts, num_topk,
        recipe, activation, activation_clamp_opt, fast_math);
}

static void fp8_mxfp4_processed_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math) {
    const auto& [l1_weights, l1_relative_sf, l1_secondary] = l1_weights_tuple;
    const auto& [l2_weights, l2_relative_sf, l2_secondary] = l2_weights_tuple;
    sm90_mega_moe(
        y,
        std::make_tuple(l1_weights, l1_relative_sf),
        std::make_tuple(l2_weights, l2_relative_sf),
        cumulative_local_expert_recv_stats,
        sym_buffer, sym_buffer_ptrs, rank_idx,
        num_max_tokens_per_rank, num_experts, num_topk,
        recipe, activation, activation_clamp_opt, fast_math,
        l1_secondary, l2_secondary);
}

static void register_sm90_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_sm90_mega_moe", &get_token_alignment_for_sm90_mega_moe);
    m.def("get_symm_buffer_size_for_sm90_mega_moe", &get_symm_buffer_size_for_sm90_mega_moe);
    m.def("fp8_mxfp4_mega_moe", &fp8_mxfp4_mega_moe);
    m.def("fp8_mxfp4_processed_mega_moe", &fp8_mxfp4_processed_mega_moe);
#endif
}

} // namespace deep_gemm::mega
