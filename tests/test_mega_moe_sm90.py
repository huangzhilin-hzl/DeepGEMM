"""Layered tests for the SM90 (Hopper) MegaMoE kernel.

The split FP8 SM90 MegaMoE kernel is exercised across a hierarchy of
scenarios so that each kernel path / heuristic branch / edge case is
covered with at least one configuration.

Layers
------
  L1  Smoke           : single tiny config; only verifies the kernel runs
                        and produces an output close to a PyTorch reference.
  L2  Heuristic       : covers tokens-per-expert bands of the SM90 selector.
  L3  Shape coverage  : covers divisible-by-128 ``hidden``,
                        ``intermediate_hidden`` and ``num_topk`` values.
  L4  Edge cases      : masking ratio, activation clamp (finite vs inf),
                        ``fast_math`` 0/1, ``num_tokens`` boundaries.
  L5  Stress          : ``--num-correctness-tests`` repeated random configs.

Notes
-----
*   The reference is a pure PyTorch BF16/FP32 simulation of the split path
    (dequantize -> matmul -> SwiGLU + clamp + per-row quantize -> matmul ->
    cross-rank scatter -> BF16 reduce).  It is *not* bitwise-identical to
    the kernel; correctness is checked with ``calc_diff < 0.01`` by default.
*   Because every scenario allocates its own symmetric memory buffer we
    re-`init_dist`/`destroy` once per process at the outer level only,
    and re-create ``SymmBuffer`` per scenario.
*   Skips itself when the device is not SM90.
"""

import argparse
from collections import Counter
import math
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple, List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import calc_diff, get_arch_major


# ----------------------------------------------------------------------------
# Quantization helpers
# ----------------------------------------------------------------------------

def _quantize_grouped_fp8_block_128_128(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block (128, 128) FP8 quantization along (N, K).

    Args
    ----
    w : (G, N, K) bf16, with N % 128 == 0 and K % 128 == 0

    Returns
    -------
    fp8 : (G, N, K) torch.float8_e4m3fn
    sf  : (G, N // 128, K // 128) torch.float32, MN-major in the (N, K)
          plane (i.e. K is the inner contiguous dim, matching the kernel's
          ``stride_k = 1`` expectation and the DeepEP convention).
    """
    g, n, k = w.shape
    assert n % 128 == 0 and k % 128 == 0
    w_view = w.view(g, n // 128, 128, k // 128, 128).float()
    amax = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4)        # (G, N/128, K/128)
    sf = amax / 448.0
    w_fp8 = (w_view / sf.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return w_fp8.view(g, n, k).contiguous(), sf.contiguous()


def _dequant_block_128_128(w_fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of `_quantize_grouped_fp8_block_128_128`. Returns fp32."""
    *prefix, n, k = w_fp8.shape
    assert n % 128 == 0 and k % 128 == 0
    w_view = w_fp8.float().view(*prefix, n // 128, 128, k // 128, 128)
    return (w_view * sf.unsqueeze(-1).unsqueeze(-3)).view(*prefix, n, k)


def _quantize_grouped_mxfp4(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Humming-format packed E2M1 weights plus float UE8M0 powers of two."""
    g, n, k = w.shape
    packed = torch.empty((g, n, k // 2), dtype=torch.int8, device=w.device)
    sf = torch.empty((g, n, k // 32), dtype=torch.float32, device=w.device)
    for group_idx in range(g):
        packed[group_idx], sf[group_idx] = per_token_cast_to_fp4(
            w[group_idx], use_ue8m0=True, gran_k=32)
    return packed, sf


def _dequant_mxfp4(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Dequantize arbitrary-prefix packed E2M1 tensors with K32 scales."""
    *prefix, n, packed_k = packed.shape
    k = packed_k * 2
    codes = torch.empty((*prefix, n, k), dtype=torch.int8, device=packed.device)
    codes[..., 0::2] = packed & 0x0f
    codes[..., 1::2] = (packed >> 4) & 0x0f
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=packed.device)
    value_idx = (codes & 0x07).to(torch.long)
    values = magnitudes[value_idx]
    values = torch.where((codes & 0x08) != 0, -values, values)
    return (values.view(*prefix, n, k // 32, 32) *
            sf.unsqueeze(-1)).view(*prefix, n, k)


def _deinterleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    """Restore natural ``[gate | up]`` rows from the SM90 SwiGLU layout."""
    g, n, *rest = t.shape
    half = n // 2
    interleaved = t.view(g, half // gran, 2, gran, *rest)
    out = torch.empty_like(t)
    out[:, :half].copy_(interleaved[:, :, 0].reshape(g, half, *rest))
    out[:, half:].copy_(interleaved[:, :, 1].reshape(g, half, *rest))
    return out


def _check_mxfp4_processed_decode_mapping() -> None:
    """Prove the processed decoder covers every B128 tile word exactly once.

    Keep this coordinate oracle in lockstep with the half-warp mapping in
    ``sm90_fp8_mega_moe.cuh``.  The end-to-end tolerance check can hide a
    sparse lane error; this contract cannot.
    """
    block_n = 128
    block_k = 128
    packed_words_per_row = 128 // 8
    rows_per_decode_group = 8
    visits = []
    expanded_visits = []
    scale_word_owners = []
    scale_uses = []
    source_bytes = []
    destination_bytes = []
    per_decoder_warp = Counter()
    per_decoder_half = Counter()

    def swizzle(byte_offset: int, bits: int,
                base: int = 4, shift: int = 3) -> int:
        mask = (1 << bits) - 1
        return byte_offset ^ (
            ((byte_offset >> (base + shift)) & mask) << base)

    for decoder_warp_idx in range(2):
        for n_half in range(2):
            decoder_row_base = decoder_warp_idx * 64 + n_half * 32
            for lane_idx in range(32):
                lane_in_half_warp = lane_idx % 16
                row_in_decode_group = lane_in_half_warp // 2
                packed_k_in_k32 = (
                    (lane_idx // 16) * 2 + lane_in_half_warp % 2)
                scale_owner_n = decoder_row_base + lane_idx
                scale_word_owners.append(scale_owner_n)
                for row_group in range(32 // rows_per_decode_group):
                    row_in_warp = (
                        row_group * rows_per_decode_group +
                        row_in_decode_group)
                    decoded_local_n = decoder_row_base + row_in_warp
                    scale_source_lane = row_in_warp
                    assert (
                        scale_owner_n - lane_idx + scale_source_lane ==
                        decoded_local_n)
                    for k32_idx in range(128 // 32):
                        packed_k = k32_idx * 4 + packed_k_in_k32
                        assert packed_k // 4 == k32_idx
                        visits.append((decoded_local_n, packed_k))
                        scale_uses.append((decoded_local_n, k32_idx))
                        per_decoder_warp[decoder_warp_idx] += 1
                        per_decoder_half[(decoder_warp_idx, n_half)] += 1

                        logical_source = (
                            decoded_local_n * (block_k // 2) +
                            packed_k * 4)
                        physical_source = swizzle(logical_source, bits=2)
                        source_bytes.extend(
                            range(physical_source, physical_source + 4))

                        logical_destination = (
                            decoded_local_n * block_k + packed_k * 8)
                        physical_destination = swizzle(
                            logical_destination, bits=3)
                        destination_bytes.extend(range(
                            physical_destination,
                            physical_destination + 8))
                        expanded_visits.extend(
                            (decoded_local_n, packed_k * 8 + element)
                            for element in range(8))

    expected = [
        (row, packed_k)
        for row in range(block_n)
        for packed_k in range(packed_words_per_row)
    ]
    assert len(visits) == block_n * packed_words_per_row
    assert len(set(visits)) == len(visits)
    assert sorted(visits) == expected
    assert len(expanded_visits) == block_n * block_k
    assert len(set(expanded_visits)) == len(expanded_visits)
    assert set(expanded_visits) == {
        (row, k) for row in range(block_n) for k in range(block_k)
    }
    assert Counter(scale_word_owners) == {
        row: 1 for row in range(block_n)
    }
    assert Counter(scale_uses) == {
        (row, k32_idx): 4
        for row in range(block_n)
        for k32_idx in range(block_k // 32)
    }
    assert per_decoder_warp == {0: 1024, 1: 1024}
    assert set(per_decoder_half.values()) == {512}
    assert len(source_bytes) == block_n * block_k // 2
    assert len(set(source_bytes)) == len(source_bytes)
    assert min(source_bytes) == 0
    assert max(source_bytes) == block_n * block_k // 2 - 1
    assert len(destination_bytes) == block_n * block_k
    assert len(set(destination_bytes)) == len(destination_bytes)
    assert min(destination_bytes) == 0
    assert max(destination_bytes) == block_n * block_k - 1


def _check_mxfp4_format_contract(device: torch.device) -> None:
    """Check raw Humming bytes, nibble order, and UE8M0 endpoint semantics."""
    from deep_gemm.mega import _normalize_mxfp4_ue8m0, _process_mxfp4_fused_e8m0

    _check_mxfp4_processed_decode_mapping()

    # Low nibble is the even-K value and high nibble is the odd-K value.
    golden_bytes = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe] * 2,
        dtype=torch.uint8, device=device).view(torch.int8).reshape(1, 1, 16)
    golden_sf = torch.ones((1, 1, 1), dtype=torch.float32, device=device)
    positive = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    negative = [0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    expected = torch.tensor(
        (positive + negative) * 2, dtype=torch.float32, device=device).reshape(1, 1, 32)
    assert torch.equal(_dequant_mxfp4(golden_bytes, golden_sf), expected)

    # Raw checkpoints use uint8 packed weights and uint8/float8_e8m0fnu scales.
    l1_raw = torch.arange(32 * 16, dtype=torch.int32, device=device).to(
        torch.uint8).reshape(1, 32, 16)
    l2_raw = torch.arange(16 * 16, dtype=torch.int32, device=device).to(
        torch.uint8).reshape(1, 16, 16)
    endpoint_codes = torch.tensor(
        [0, 1, 127, 254, 255], dtype=torch.uint8, device=device)
    l1_sf_raw = endpoint_codes[torch.arange(32, device=device) % 5].reshape(1, 32, 1)
    l2_sf_raw = endpoint_codes[torch.arange(16, device=device) % 5].reshape(1, 16, 1)
    e8m0_dtype = getattr(torch, 'float8_e8m0fnu', None)
    l1_sf_input = l1_sf_raw if e8m0_dtype is None else l1_sf_raw.view(e8m0_dtype)
    l2_sf_input = l2_sf_raw if e8m0_dtype is None else l2_sf_raw.view(e8m0_dtype)
    (l1_w, l1_sf), (l2_w, l2_sf) = \
        deep_gemm.transform_weights_for_fp8_mxfp4_mega_moe_sm90(
            (l1_raw, l1_sf_input), (l2_raw, l2_sf_input))
    l1_order = torch.tensor(
        list(range(8)) + list(range(16, 24)) +
        list(range(8, 16)) + list(range(24, 32)),
        dtype=torch.long, device=device)
    assert l1_w.dtype == torch.int8 and l2_w.dtype == torch.int8
    assert torch.equal(l1_w.view(torch.uint8), l1_raw[:, l1_order])
    assert torch.equal(l1_sf, l1_sf_raw[:, l1_order])
    assert torch.equal(l2_w.view(torch.uint8), l2_raw)
    assert torch.equal(l2_sf, l2_sf_raw)

    # The fused-E8M0 preprocessing follows Humming's exact E2M1 requantizer.
    # A 16-code row exercises both signs, negative-zero normalization, and all
    # rounding boundaries for exponent deltas 0..5.
    packed_codes = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe] * 2,
        dtype=torch.uint8, device=device)
    processed_input = packed_codes.repeat(7, 1).reshape(1, 7, 16)
    processed_sf = torch.tensor(
        [109, 108, 107, 106, 105, 104, 120],
        dtype=torch.uint8, device=device).reshape(1, 7, 1)
    processed_w, processed_offsets, processed_secondary = \
        _process_mxfp4_fused_e8m0(processed_input, processed_sf)
    expected_rows = torch.tensor([
        [0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe],
        [0x10, 0x21, 0x32, 0x54, 0x90, 0xa9, 0xba, 0xdc],
        [0x00, 0x11, 0x21, 0x32, 0x80, 0x99, 0xa9, 0xba],
        [0x00, 0x00, 0x11, 0x21, 0x80, 0x88, 0x99, 0xa9],
        [0x00, 0x00, 0x00, 0x11, 0x80, 0x88, 0x88, 0x99],
        [0x00, 0x00, 0x00, 0x00, 0x80, 0x88, 0x88, 0x88],
        [0x10, 0x32, 0x54, 0x76, 0x90, 0xba, 0xdc, 0xfe],
    ], dtype=torch.uint8, device=device).repeat(1, 2)
    assert torch.equal(processed_w.view(torch.uint8)[0], expected_rows)
    assert torch.equal(
        processed_offsets,
        torch.tensor([1, 1, 1, 1, 1, 1, 12], dtype=torch.uint8,
                     device=device).reshape(1, 7, 1))
    assert torch.equal(
        processed_secondary,
        torch.tensor([2.0 ** -19], dtype=torch.float32, device=device))
    try:
        _process_mxfp4_fused_e8m0(
            processed_input[:, :1],
            torch.full((1, 1, 1), 255, dtype=torch.uint8, device=device))
    except AssertionError:
        pass
    else:
        raise AssertionError('fused E8M0 processing accepted NaN scale code 255')
    try:
        _process_mxfp4_fused_e8m0(
            processed_input[:, :2],
            torch.tensor([0, 254], dtype=torch.uint8, device=device).reshape(1, 2, 1))
    except AssertionError:
        pass
    else:
        raise AssertionError('fused E8M0 processing accepted exponent delta >= 128')

    # The explicit fused transform accepts Humming's optional per-expert
    # weight_scale_2 and applies L1 row interleave without a full-size stack.
    fused_l1_input = packed_codes.repeat(16, 1).reshape(1, 16, 16)
    fused_l1_sf = torch.tensor(
        [109] * 15 + [120], dtype=torch.uint8, device=device).reshape(1, 16, 1)
    fused_l1, fused_l2 = \
        deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            (fused_l1_input, fused_l1_sf),
            (processed_input, processed_sf),
            l1_global_scale=torch.tensor([2.0], device=device),
            l2_global_scale=torch.tensor([3.0], device=device))
    assert len(fused_l1) == 3 and len(fused_l2) == 3
    assert torch.equal(
        fused_l1[2], torch.tensor([2.0 ** -18], device=device))
    assert torch.equal(
        fused_l2[2], torch.tensor([3.0 * 2.0 ** -19], device=device))

    # FP32 input supports every finite UE8M0 power of two, including the
    # subnormal 2^-127 endpoint, but rejects zero and infinity.
    fp32_scales = torch.tensor(
        [2.0 ** -127, 2.0 ** -126, 1.0, 2.0 ** 127],
        dtype=torch.float32, device=device)
    assert torch.equal(
        _normalize_mxfp4_ue8m0(fp32_scales),
        torch.tensor([0, 1, 127, 254], dtype=torch.uint8, device=device))
    for invalid in (0.0, float('inf')):
        try:
            _normalize_mxfp4_ue8m0(torch.tensor(
                [invalid], dtype=torch.float32, device=device))
        except AssertionError:
            pass
        else:
            raise AssertionError(f'invalid FP32 UE8M0 scale accepted: {invalid}')


def _dequant_per_token_per_128_k(x_fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """For (M, K) fp8 with (M, K // 128) float SF (per-token, K-major)."""
    m, k = x_fp8.shape
    assert k % 128 == 0
    w_view = x_fp8.float().view(m, k // 128, 128)
    return (w_view * sf.unsqueeze(-1)).view(m, k)


def _stable_name_seed(name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 1000


# ----------------------------------------------------------------------------
# PyTorch reference
# ----------------------------------------------------------------------------

def _swiglu_fp32(gate_up: torch.Tensor, clamp: float) -> torch.Tensor:
    """SwiGLU with one-sided gate clamp and two-sided up clamp.

    Matches the fused kernel: ``silu(min(gate, c)) * clamp(up, -c, c)``.
    """
    n2 = gate_up.size(-1)
    half = n2 // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if math.isfinite(clamp):
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * up


def _reference_fused(
    x_fp8_local: torch.Tensor, x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor, topk_weights_local: torch.Tensor,
    l1_w_fp8: torch.Tensor, l1_w_sf: torch.Tensor,
    l2_w_fp8: torch.Tensor, l2_w_sf: torch.Tensor,
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    num_experts: int, num_topk: int,
    hidden: int, intermediate_hidden: int,
    activation_clamp: float,
    weight_format: str = 'fp8',
) -> torch.Tensor:
    """Reference: returns (num_tokens, hidden) bf16 result for *this* rank.

    All-gathers the global tokens / topk decisions / per-rank weights, then
    for each global token routes through its topk experts, applies the
    L1+SwiGLU+L2 path, and reduces over topk on the source rank.
    """
    num_experts_per_rank = num_experts // num_ranks

    # --- gather global token data --------------------------------------------------
    x_fp8_g = uneven_all_gather(x_fp8_local, group=group)      # (Mg, H)
    x_sf_g = uneven_all_gather(x_sf_local, group=group)        # (Mg, H/128)
    topk_idx_g = uneven_all_gather(topk_idx_local, group=group)         # (Mg, K)
    topk_w_g = uneven_all_gather(topk_weights_local, group=group)       # (Mg, K)
    mg = x_fp8_g.size(0)

    # rank-id lookup for each gathered token (for combine routing)
    rank_offsets = [0]
    sizes = [torch.tensor([0], device='cuda')]                  # placeholder
    # mimic uneven_all_gather to compute per-rank token counts
    local_size = torch.tensor([x_fp8_local.size(0)], device='cuda', dtype=torch.long)
    sizes_t = torch.empty(num_ranks, dtype=torch.long, device='cuda')
    dist.all_gather_into_tensor(sizes_t, local_size, group=group)
    sizes_list = sizes_t.tolist()
    src_rank_of = torch.empty(mg, dtype=torch.long, device='cuda')
    cur = 0
    for r, s in enumerate(sizes_list):
        src_rank_of[cur:cur + s] = r
        cur += s
    assert cur == mg

    # --- gather all-rank weights --------------------------------------------------
    # l1_w_fp8: (E_pr, 2*IH, H), l1_w_sf: (E_pr, 2*IH, H/128)
    l1_w_g = [torch.empty_like(l1_w_fp8) for _ in range(num_ranks)]
    l1_sf_g = [torch.empty_like(l1_w_sf) for _ in range(num_ranks)]
    l2_w_g = [torch.empty_like(l2_w_fp8) for _ in range(num_ranks)]
    l2_sf_g = [torch.empty_like(l2_w_sf) for _ in range(num_ranks)]
    dist.all_gather(l1_w_g, l1_w_fp8, group=group)
    dist.all_gather(l1_sf_g, l1_w_sf, group=group)
    dist.all_gather(l2_w_g, l2_w_fp8, group=group)
    dist.all_gather(l2_sf_g, l2_w_sf, group=group)
    l1_w_all = torch.stack(l1_w_g, dim=0)   # (R, E_pr, 2*IH, H)
    l1_sf_all = torch.stack(l1_sf_g, dim=0)
    l2_w_all = torch.stack(l2_w_g, dim=0)
    l2_sf_all = torch.stack(l2_sf_g, dim=0)

    # --- per-token / per-topk compute --------------------------------------------------
    # The combine slot tensor: (Mg, K, H) bf16 — each src rank will reduce over K.
    combine_buf = torch.zeros(mg, num_topk, hidden, dtype=torch.float32, device='cuda')

    # Precompute dequantized x in fp32
    x_fp32 = _dequant_per_token_per_128_k(x_fp8_g, x_sf_g)         # (Mg, H)

    # Iterate (cheap; reference is for small test configs only)
    # Token-chunked to keep gathered (S, 2*IH, H) dequant tensors below GPU memory.
    # Wide H200 Pro weights make a 256-token gathered dequantization exceed
    # 40 GiB. Keep the reference chunk small enough for the full production
    # shape while preserving exactly the same arithmetic.
    _CHUNK = 32
    for k in range(num_topk):
        # Skip masked
        mask = topk_idx_g[:, k] >= 0
        if not mask.any():
            continue
        sel_idx_full = mask.nonzero(as_tuple=False).squeeze(-1)    # (S,)
        for c0 in range(0, sel_idx_full.numel(), _CHUNK):
            sel_idx = sel_idx_full[c0:c0 + _CHUNK]
            eids = topk_idx_g[sel_idx, k]                          # (S,)
            weights = topk_w_g[sel_idx, k]                         # (S,)
            x_sel = x_fp32[sel_idx]                                # (S, H)

            dst_rank = (eids // num_experts_per_rank).long()
            dst_local = (eids % num_experts_per_rank).long()

            # L1 GEMM (per-token): y = x @ W^T  shape (S, 2*IH)
            if weight_format == 'mxfp4':
                l1_w_sel = _dequant_mxfp4(
                    l1_w_all[dst_rank, dst_local],
                    l1_sf_all[dst_rank, dst_local])
            else:
                l1_w_sel = _dequant_block_128_128(
                    l1_w_all[dst_rank, dst_local],                 # (S, 2*IH, H)
                    l1_sf_all[dst_rank, dst_local])
            l1_y = torch.einsum('sk,snk->sn', x_sel, l1_w_sel)     # (S, 2*IH)
            del l1_w_sel

            # SwiGLU + clamp + multiply by topk weight
            l1_y = _swiglu_fp32(l1_y, activation_clamp) * weights.unsqueeze(-1)   # (S, IH)

            # Per-row, per-64-col FP8 quantize -> dequantize
            s_, ih = l1_y.shape
            assert ih == intermediate_hidden and ih % 64 == 0
            l1_view = l1_y.view(s_, ih // 64, 64)
            amax = l1_view.abs().amax(dim=-1).clamp(1e-4)          # (S, IH/64)
            sf2 = amax / 448.0
            l1_q = (l1_view / sf2.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
            l2_in = (l1_q * sf2.unsqueeze(-1)).view(s_, ih)        # (S, IH) fp32

            # L2 GEMM
            if weight_format == 'mxfp4':
                l2_w_sel = _dequant_mxfp4(
                    l2_w_all[dst_rank, dst_local],
                    l2_sf_all[dst_rank, dst_local])
            else:
                l2_w_sel = _dequant_block_128_128(
                    l2_w_all[dst_rank, dst_local],                 # (S, H, IH)
                    l2_sf_all[dst_rank, dst_local])
            l2_y = torch.einsum('sn,smn->sm', l2_in, l2_w_sel)     # (S, H)
            del l2_w_sel

            # Scatter to combine buffer (cast to bf16 then back to mimic kernel storage)
            combine_buf[sel_idx, k] = l2_y.to(torch.bfloat16).float()

    # Sum over K -> (Mg, H), keep only this rank's slice
    y_full_bf16 = combine_buf.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16)  # (Mg, H)
    start = sum(sizes_list[:rank_idx])
    end = start + sizes_list[rank_idx]
    return y_full_bf16[start:end].contiguous()


# ----------------------------------------------------------------------------
# Single-scenario runner
# ----------------------------------------------------------------------------

def _run_scenario(
    name: str,
    cfg: Dict[str, Any],
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    diff_tol: float,
):
    num_max = cfg['num_max_tokens_per_rank']
    num_tokens = cfg.get('num_tokens', num_max)
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    masked_ratio = cfg.get('masked_ratio', 0.0)
    activation_clamp = cfg.get('activation_clamp', 10.0)
    fast_math = cfg.get('fast_math', True)
    weight_format = cfg.get('weight_format', 'fp8')
    mxfp4_scale_mode = cfg.get('mxfp4_scale_mode', 'processed')
    force_mxfp4_requant = cfg.get('force_mxfp4_requant', False)

    assert num_experts % num_ranks == 0, f'{name}: experts {num_experts} not divisible by ranks {num_ranks}'
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0

    verbose = bool(int(os.environ.get('DG_TEST_VERBOSE', '0')))
    def _trace(stage: str):
        if verbose:
            print(f'[rank{rank_idx}] {name} :: {stage}', flush=True)

    _trace('begin')
    seed = rank_idx * 1000 + _stable_name_seed(name)
    torch.manual_seed(seed)
    random.seed(seed)

    # ---- Inputs (bf16) -------------------------------------------------------
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    # Quantize x to FP8 with per-128 K float SF (SM90 format)
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    if weight_format == 'mxfp4':
        l1_w_fp8, l1_w_sf = _quantize_grouped_mxfp4(l1_bf)
        l2_w_fp8, l2_w_sf = _quantize_grouped_mxfp4(l2_bf)
        if force_mxfp4_requant:
            assert mxfp4_scale_mode == 'processed'
            # Force exponent spread 12 so code 115 is clamped to base 116
            # and its packed payload is actually requantized (delta=1).
            l1_codes = torch.arange(
                l1_w_sf.numel(), dtype=torch.int64, device='cuda').reshape_as(l1_w_sf)
            l2_codes = torch.arange(
                l2_w_sf.numel(), dtype=torch.int64, device='cuda').reshape_as(l2_w_sf)
            l1_w_sf = torch.exp2((l1_codes % 13 + 115).float() - 127.0)
            l2_w_sf = torch.exp2((l2_codes % 13 + 115).float() - 127.0)
    else:
        # Block (128, 128), matching DeepSeekV4FlashFp8 / DeepEP.
        l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
        l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)

    _trace('weight_transform')
    if weight_format == 'mxfp4':
        transform = deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90 \
            if mxfp4_scale_mode == 'processed' \
            else deep_gemm.transform_weights_for_fp8_mxfp4_mega_moe_sm90
        transformed_l1, transformed_l2 = transform(
            (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf))
    else:
        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
            (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf))

    reference_l1_w, reference_l1_sf = l1_w_fp8, l1_w_sf
    reference_l2_w, reference_l2_sf = l2_w_fp8, l2_w_sf
    if weight_format == 'mxfp4' and mxfp4_scale_mode == 'processed':
        l1_processed_w, l1_offsets, l1_secondary = transformed_l1
        l2_processed_w, l2_offsets, l2_secondary = transformed_l2
        reference_l1_w = _deinterleave_weights(l1_processed_w)
        reference_l1_sf = (
            l1_secondary[:, None, None] *
            torch.exp2(_deinterleave_weights(l1_offsets).float()))
        reference_l2_w = l2_processed_w
        reference_l2_sf = (
            l2_secondary[:, None, None] * torch.exp2(l2_offsets.float()))
        if force_mxfp4_requant:
            assert (reference_l1_w != l1_w_fp8).any()
            assert (reference_l2_w != l2_w_fp8).any()

    # ---- Allocate symm buffer -----------------------------------------------
    _trace('alloc_symm_buffer')
    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        group, num_experts,
        num_max, num_topk,
        hidden, intermediate_hidden,
    )
    cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')

    # ---- Run SM90 MegaMoE ----------------------------------------------------
    _trace('copy_inputs')
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_w)

    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    _trace('launch_sm90 (may JIT-compile, can take minutes)')
    mega_moe = deep_gemm.fp8_mxfp4_mega_moe \
        if weight_format == 'mxfp4' else deep_gemm.fp8_mega_moe
    mega_moe(
        y_fused, transformed_l1, transformed_l2, buffer,
        cumulative_local_expert_recv_stats=cum_stats,
        recipe=(1, 1, 32) if weight_format == 'mxfp4' else (128, 128, 128),
        activation='swiglu',
        activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
        fast_math=fast_math,
    )
    _trace('sync_sm90')
    torch.cuda.synchronize()
    _trace('sm90_done')

    # ---- Reference & check ---------------------------------------------------
    # Use the FP8 weights and their block-(128, 128) SF directly — the dequant
    # helper expects this MN/K-block SF layout, and the original (gate||up) row
    # ordering is what `_swiglu_fp32` splits with ``[..., :IH], [..., IH:]``.
    _trace('reference')
    y_ref = _reference_fused(
        x_fp8, x_sf, topk_idx, topk_w,
        reference_l1_w, reference_l1_sf, reference_l2_w, reference_l2_sf,
        rank_idx, num_ranks, group,
        num_experts, num_topk,
        hidden, intermediate_hidden,
        activation_clamp,
        weight_format,
    )

    diff = calc_diff(y_fused, y_ref)
    ok = diff < diff_tol
    failed = torch.tensor([not ok], dtype=torch.int, device='cuda')
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    any_rank_failed = bool(failed.item())
    format_label = f'{weight_format}/{mxfp4_scale_mode}' \
        if weight_format == 'mxfp4' else weight_format
    dist_print(f'  [{name:<32}] format={format_label:<16} diff={diff:.4f} '
               f'(tol={diff_tol:.2f}) {"OK" if not any_rank_failed else "FAIL"}',
               once_in_node=True)

    buffer.destroy()
    dist.barrier()
    assert not any_rank_failed, (
        f'{name}: at least one rank exceeded diff tolerance {diff_tol}; '
        f'local diff={diff}'
    )


# ----------------------------------------------------------------------------
# Scenario tables
# ----------------------------------------------------------------------------

# A single tiny config used as a smoke test.
_SMOKE = dict(
    num_max_tokens_per_rank=64, num_tokens=64,
    hidden=512, intermediate_hidden=512,
    num_experts=8, num_topk=2,
)


def _layer1_smoke() -> List[Tuple[str, Dict[str, Any]]]:
    return [('L1.smoke', dict(_SMOKE))]


def _layer1_mxfp4_requant() -> List[Tuple[str, Dict[str, Any]]]:
    cfg = dict(_SMOKE)
    cfg['force_mxfp4_requant'] = True
    return [('L1.mxfp4_requant', cfg)]


def _layer2_heuristic_branches(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Cover generic heuristic bands across alternate top-k values."""
    base = dict(hidden=1024, intermediate_hidden=1024,
                num_experts=8 * num_ranks, num_topk=2)
    out: List[Tuple[str, Dict[str, Any]]] = []
    for tokens, label in [(64, 'small'), (256, 'midA'), (512, 'midB'), (2048, 'large')]:
        cfg = dict(base)
        cfg.update(num_max_tokens_per_rank=tokens, num_tokens=tokens)
        out.append((f'L2.heur.{label}.t{tokens}', cfg))
    generic_topk8_base = dict(hidden=512, intermediate_hidden=2048,
                              num_experts=32 * num_ranks, num_topk=8)
    for tokens in (16, 64, 260, 1024):
        cfg = dict(generic_topk8_base)
        cfg.update(num_max_tokens_per_rank=tokens, num_tokens=tokens)
        out.append((f'L2.generic_topk8.t{tokens}', cfg))
    generic_topk6 = dict(generic_topk8_base, num_topk=6)
    generic_topk6.update(num_max_tokens_per_rank=512, num_tokens=512)
    out.append(('L2.generic_topk6.t512', generic_topk6))
    return out


def _layer3_shape_cases(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    base_experts = 8 * num_ranks
    for hidden in (512, 2048):
        for ih in (512, 2048):
            for topk in (1, 2, 4):
                if topk > base_experts:
                    continue
                cfg = dict(num_max_tokens_per_rank=128, num_tokens=128,
                           hidden=hidden, intermediate_hidden=ih,
                           num_experts=base_experts, num_topk=topk)
                out.append((f'L3.h{hidden}_ih{ih}_k{topk}', cfg))
    # Production H200 shapes. Pro M128 covers the BN512/BF16 path and M256
    # covers the BK256 split-phase path when this runs on a full H200 node.
    out.extend([
        ('L3.h200_flash_m128', dict(
            num_max_tokens_per_rank=128, num_tokens=128,
            hidden=4096, intermediate_hidden=2048,
            num_experts=32 * num_ranks, num_topk=6)),
        ('L3.h200_pro_m128', dict(
            num_max_tokens_per_rank=128, num_tokens=128,
            hidden=7168, intermediate_hidden=3072,
            num_experts=48 * num_ranks, num_topk=6)),
        ('L3.h200_pro_m256', dict(
            num_max_tokens_per_rank=256, num_tokens=256,
            hidden=7168, intermediate_hidden=3072,
            num_experts=48 * num_ranks, num_topk=6)),
        ('L3.h200_range_alt_compact_load24', dict(
            num_max_tokens_per_rank=72, num_tokens=72,
            hidden=4096, intermediate_hidden=2048,
            num_experts=24 * num_ranks, num_topk=8)),
        ('L3.h200_range_alt_wide_load32', dict(
            num_max_tokens_per_rank=128, num_tokens=128,
            hidden=7168, intermediate_hidden=3072,
            num_experts=32 * num_ranks, num_topk=8)),
    ])
    return out


def _layer4_edges(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base = dict(num_max_tokens_per_rank=128,
                hidden=512, intermediate_hidden=512,
                num_experts=8 * num_ranks, num_topk=2)
    out = []
    # Masked ratios
    for mr in (0.0, 0.3, 0.7):
        cfg = dict(base); cfg.update(num_tokens=128, masked_ratio=mr)
        out.append((f'L4.mask{mr:.1f}', cfg))
    # All masked
    cfg = dict(base); cfg.update(num_tokens=128, masked_ratio=1.0)
    out.append(('L4.mask_all', cfg))
    # Activation clamp variations (finite vs inf)
    for c in (1.0, 10.0, math.inf):
        cfg = dict(base); cfg.update(num_tokens=128, activation_clamp=c)
        out.append((f'L4.clamp{c}', cfg))
    # fast_math toggle
    for fm in (True, False):
        cfg = dict(base); cfg.update(num_tokens=128, fast_math=fm)
        out.append((f'L4.fm{int(fm)}', cfg))
    # num_tokens boundaries
    cfg = dict(base); cfg.update(num_tokens=0)
    out.append(('L4.tokens0', cfg))
    cfg = dict(base); cfg.update(num_tokens=base['num_max_tokens_per_rank'])
    out.append(('L4.tokens_max', cfg))
    return out


def _layer5_stress(num_ranks: int, num_tests: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Random configs under simple constraints."""
    rng = random.Random(0xC0FFEE)
    out = []
    for i in range(num_tests):
        hidden = rng.choice([512, 1024, 2048])
        ih = rng.choice([512, 1024, 2048])
        topk = rng.choice([1, 2, 4])
        tokens = rng.choice([32, 64, 128, 256, 512])
        masked = rng.choice([0.0, 0.0, 0.3, 0.5])
        clamp = rng.choice([1.0, 10.0, math.inf])
        fm = rng.choice([True, False])
        cfg = dict(num_max_tokens_per_rank=tokens, num_tokens=tokens,
                   hidden=hidden, intermediate_hidden=ih,
                   num_experts=8 * num_ranks, num_topk=topk,
                   masked_ratio=masked, activation_clamp=clamp, fast_math=fm)
        out.append((f'L5.rand{i:03d}', cfg))
    return out


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def _test_worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)

    # Skip on non-SM90
    if get_arch_major() != 9:
        dist_print(f'[SKIP] test_mega_moe_sm90 requires SM90; got SM{get_arch_major()}0',
                   once_in_node=True)
        dist.destroy_process_group()
        return

    if args.weight_format == 'mxfp4':
        _check_mxfp4_format_contract(torch.device('cuda'))
        dist_print('MXFP4 format/decode mapping contract: PASS', once_in_node=True)

    diff_tol = args.diff_tol
    layers: List[Tuple[str, Dict[str, Any]]] = []

    if 1 in args.layers:
        layers += _layer1_smoke()
        if args.weight_format == 'mxfp4' and args.mxfp4_scale_mode == 'processed':
            layers += _layer1_mxfp4_requant()
    if 2 in args.layers:
        layers += _layer2_heuristic_branches(num_ranks)
    if 3 in args.layers:
        layers += _layer3_shape_cases(num_ranks)
    if 4 in args.layers:
        layers += _layer4_edges(num_ranks)
    if 5 in args.layers:
        layers += _layer5_stress(num_ranks, args.num_correctness_tests or 8)

    if args.filter:
        layers = [(n, c) for n, c in layers if args.filter in n]
    for _, cfg in layers:
        cfg['weight_format'] = args.weight_format
        cfg['mxfp4_scale_mode'] = args.mxfp4_scale_mode

    dist_print(f'SM90 MegaMoE test plan: {len(layers)} scenarios across '
               f'layers {sorted(args.layers)} on {num_ranks} ranks',
               once_in_node=True)

    failures: List[str] = []
    for name, cfg in layers:
        try:
            _run_scenario(name, cfg, rank_idx, num_ranks, group, diff_tol)
        except AssertionError as ex:
            dist_print(f'  [{name}] FAIL: {ex}', once_in_node=True)
            failures.append(name)
            if args.fail_fast:
                break

    dist_print('', once_in_node=True)
    if failures:
        dist_print(f'FAILED {len(failures)}/{len(layers)} scenarios: {failures}',
                   once_in_node=True)
    else:
        dist_print(f'PASSED all {len(layers)} scenarios', once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layered SM90 MegaMoE tests')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='Number of ranks to spawn (default: 2)')
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Which layers to run (1..5). Default: 1 2 3 4. '
                             'Layer 5 requires --num-correctness-tests.')
    parser.add_argument('--num-correctness-tests', type=int, default=None,
                        help='Layer 5 stress test count')
    parser.add_argument('--filter', type=str, default='',
                        help='Substring filter on scenario names')
    parser.add_argument('--diff-tol', type=float, default=0.01,
                        help='calc_diff tolerance (default: 0.01)')
    parser.add_argument('--weight-format', choices=('fp8', 'mxfp4'), default='fp8',
                        help='SM90 weight path to test (default: fp8)')
    parser.add_argument('--mxfp4-scale-mode', choices=('processed', 'raw'),
                        default='processed',
                        help='MXFP4 scale representation (default: processed)')
    parser.add_argument('--fail-fast', action='store_true',
                        help='Stop on first failing scenario')
    args = parser.parse_args()

    np = args.num_processes
    torch.multiprocessing.spawn(_test_worker, args=(np, args), nprocs=np)
