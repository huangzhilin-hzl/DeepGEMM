import torch
import types
from typing import Tuple, Optional
from ..utils.math import align

# noinspection PyBroadException
try:
    # noinspection PyProtectedMember
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load mega kernels, please check your PyTorch version: {exception}')

from .. import _C


class SymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 use_fp8_dispatch: bool = True,
                 activation: str = 'swiglu',
                 _get_size_fn=_C.get_symm_buffer_size_for_mega_moe):
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        # Allocate a symmetric buffer
        num_bytes, slice_input_buffers = _get_size_fn(
            group.size(), num_experts,
            num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden,
            use_fp8_dispatch, activation
        )
        allocator = torch if group.size() == 1 else symm_mem
        self.buffer = allocator.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = (
            types.SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm_mem.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Create input buffer views
        (self.x, self.x_sf,
         self.topk_idx, self.topk_weights,
         self.l1_acts, self.l1_acts_sf,
         self.l2_acts, self.l2_acts_sf) = slice_input_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        for name in (
            'x', 'x_sf', 'topk_idx', 'topk_weights',
            'l1_acts', 'l1_acts_sf', 'l2_acts', 'l2_acts_sf',
        ):
            setattr(self, name, None)
        self.buffer = None
        self.group = None


# Keep the public name while sharing one implementation.
SM90SymmBuffer = SymmBuffer


def get_symm_buffer_for_mega_moe(group: dist.ProcessGroup,
                                 num_experts: int,
                                 num_max_tokens_per_rank: int, num_topk: int,
                                 hidden: int, intermediate_hidden: int,
                                 use_fp8_dispatch: bool = True,
                                 activation: str = 'swiglu') -> SymmBuffer:
    # Token count must be aligned to block sizes
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_mega_moe())

    return SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation
    )


def get_symm_buffer_for_sm90_mega_moe(group: dist.ProcessGroup,
                                      num_experts: int,
                                      num_max_tokens_per_rank: int, num_topk: int,
                                      hidden: int, intermediate_hidden: int,
                                      use_fp8_dispatch: bool = True,
                                      activation: str = 'swiglu') -> SM90SymmBuffer:
    num_max_tokens_per_rank = align(
        num_max_tokens_per_rank, _C.get_token_alignment_for_sm90_mega_moe())
    return SM90SymmBuffer(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        use_fp8_dispatch, activation,
        _get_size_fn=_C.get_symm_buffer_size_for_sm90_mega_moe,
    )


def _interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    # [gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...] instead of [gate | up]
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    out = torch.empty_like(t)
    out_view = out.view(g, half // gran, 2, gran, *rest)
    out_view[:, :, 0].copy_(gate)
    out_view[:, :, 1].copy_(up)
    return out


def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    num_groups, mn, packed_sf_k = sf.shape
    assert sf.dtype == torch.int and mn % 128 == 0
    result = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                .transpose(2, 3)
                .reshape(num_groups, mn, packed_sf_k))
    return torch.empty_like(sf).copy_(result)


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    # L1: interleave gate/up for weight and SF, then transpose SF for UTCCP.
    l1_w = _interleave_weights(l1_weights[0])
    l1_sf = _transpose_sf_for_utccp(_interleave_weights(l1_weights[1]))
    l1_transformed = (l1_w, l1_sf)
    # L2: only transpose SF for UTCCP.
    l2_transformed = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_transformed, l2_transformed



def transform_weights_for_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """SM90 (Hopper) variant of `transform_weights_for_mega_moe`.

    SM90 has no TMEM / UTCCP path, so the SF tensors are consumed directly by
    WGMMA promote and don't need the 4x32 transpose. With block (128, 128)
    weight quantization, weight SFs are read by the math warpgroup directly
    from global memory in their natural ``(E, N/128, K/128)`` MN-major layout
    and require no transformation. Only L1's gate/up FP8 weight interleave is
    preserved.
    """
    l1_fp8, l1_sf = l1_weights
    return (_interleave_weights(l1_fp8), l1_sf), l2_weights


def _normalize_mxfp4_ue8m0(sf: torch.Tensor) -> torch.Tensor:
    """Return natural-layout UE8M0 exponent bytes.

    Raw MXFP4 checkpoints commonly expose these tensors as ``uint8`` or
    ``float8_e8m0fnu``.  The float path is convenient for DeepGEMM's
    ``per_token_cast_to_fp4`` test utility, which returns the same powers of
    two as FP32 values.
    """
    if sf.dtype == torch.uint8:
        return sf.contiguous()
    e8m0_dtype = getattr(torch, 'float8_e8m0fnu', None)
    if e8m0_dtype is not None and sf.dtype == e8m0_dtype:
        return sf.contiguous().view(torch.uint8)
    assert sf.dtype == torch.float32
    bits = sf.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xff
    mantissa = bits & ((1 << 23) - 1)
    is_min_subnormal = bits == (1 << 22)  # UE8M0 code 0: 2^-127.
    is_normal_power_of_two = (
        ((bits >> 31) == 0) & (mantissa == 0) &
        (exponent >= 1) & (exponent <= 254)
    )
    assert (is_min_subnormal | is_normal_power_of_two).all()
    return torch.where(
        is_min_subnormal, torch.zeros_like(exponent), exponent).to(torch.uint8)


def _normalize_mxfp4_packed_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return raw Humming checkpoint bytes in DeepGEMM's packed-int8 view."""
    assert weight.dtype in (torch.uint8, torch.int8)
    weight = weight.contiguous()
    return weight if weight.dtype == torch.int8 else weight.view(torch.int8)


def _reorder_mxfp4_sign_bits_for_sm90(weight: torch.Tensor) -> torch.Tensor:
    """Reorder sign bits in each packed word for the fused SM90 decoder.

    Magnitude bits stay in their original consecutive-nibble positions.  The
    eight sign bits are changed from ``[s0,s1,s2,s3,s4,s5,s6,s7]`` to
    ``[s0,s4,s1,s5,s2,s6,s3,s7]`` so the decoder can use ``packed << 4`` and
    ``packed`` directly as the signs for output bytes 0..3 and 4..7.  This is
    an internal processed-payload layout; raw Humming pairs remain unchanged.
    """
    assert weight.dtype in (torch.uint8, torch.int8)
    assert weight.size(-1) % 4 == 0
    original_dtype = weight.dtype
    source = weight.contiguous().view(torch.uint8).reshape(*weight.shape[:-1], -1, 4)
    reordered = source & 0x77
    reordered[..., 0] |= (source[..., 0] & 0x08) | ((source[..., 2] & 0x08) << 4)
    reordered[..., 1] |= ((source[..., 0] & 0x80) >> 4) | (source[..., 2] & 0x80)
    reordered[..., 2] |= (source[..., 1] & 0x08) | ((source[..., 3] & 0x08) << 4)
    reordered[..., 3] |= ((source[..., 1] & 0x80) >> 4) | (source[..., 3] & 0x80)
    return reordered.reshape(weight.shape).view(original_dtype)


def _restore_mxfp4_sign_bits_from_sm90(weight: torch.Tensor) -> torch.Tensor:
    """Restore standard consecutive-nibble signs from the SM90 layout."""
    assert weight.dtype in (torch.uint8, torch.int8)
    assert weight.size(-1) % 4 == 0
    original_dtype = weight.dtype
    source = weight.contiguous().view(torch.uint8).reshape(*weight.shape[:-1], -1, 4)
    restored = source & 0x77
    restored[..., 0] |= (source[..., 0] & 0x08) | ((source[..., 1] & 0x08) << 4)
    restored[..., 1] |= (source[..., 2] & 0x08) | ((source[..., 3] & 0x08) << 4)
    restored[..., 2] |= ((source[..., 0] & 0x80) >> 4) | (source[..., 1] & 0x80)
    restored[..., 3] |= ((source[..., 2] & 0x80) >> 4) | (source[..., 3] & 0x80)
    return restored.reshape(weight.shape).view(original_dtype)


def _process_mxfp4_fused_e8m0(
    weight: torch.Tensor,
    sf: torch.Tensor,
    interleave_rows: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw Humming MXFP4 scales to bounded exponent offsets.

    The offset range is limited to 11 per expert, matching Humming's FP8
    fused-E8M0 path. Groups below the retained range have their packed E2M1
    payload requantized after a power-of-two downscale. The returned secondary
    scale restores the common expert exponent after WGMMA.
    """
    weight = _normalize_mxfp4_packed_weight(weight)
    raw_sf = _normalize_mxfp4_ue8m0(sf)
    assert weight.dim() == 3 and raw_sf.dim() == 3
    assert weight.size(0) == raw_sf.size(0)
    assert weight.size(1) == raw_sf.size(1)
    assert weight.size(2) == raw_sf.size(2) * 16
    assert (raw_sf != 255).all(), 'fused E8M0 processing does not accept NaN scales'

    sf_i16 = raw_sf.to(torch.int16)
    max_exp = sf_i16.flatten(1).amax(dim=1)
    min_exp = sf_i16.flatten(1).amin(dim=1)
    base_exp = max_exp - torch.minimum(max_exp - min_exp, torch.full_like(max_exp, 11))

    base_view = base_exp.view(-1, 1, 1)
    clamped = torch.maximum(sf_i16, base_view)
    delta = clamped - sf_i16
    # Humming's current CUDA requantizer uses an unsigned FP32 exponent
    # construction that wraps for delta >= 128. Reject that pathological
    # endpoint instead of silently diverging from its byte representation;
    # callers can retain the raw pair contract for such experts.
    assert (delta < 128).all(), 'fused E8M0 exponent delta must be < 128'
    offsets = (clamped - base_view + 1).to(torch.uint8)
    secondary = torch.exp2(base_exp.float() - 128.0).contiguous()

    # Exact lookup for Humming's process_mxfp4_w4a8_weight over the accepted
    # delta range [0, 127]. Delta >= 5 quantizes every finite magnitude to
    # zero. Input negative zero is normalized before requantization.
    nibble_lut = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 1, 2, 2, 3, 4, 5, 0, 9, 9, 10, 10, 11, 12, 13,
        0, 0, 1, 1, 1, 2, 2, 3, 0, 8, 9, 9, 9, 10, 10, 11,
        0, 0, 0, 0, 1, 1, 1, 2, 0, 8, 8, 8, 9, 9, 9, 10,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 8, 8, 8, 8, 8, 9, 9,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8,
    ], dtype=torch.uint8, device=weight.device)
    nibble_lut = nibble_lut.view(6, 16)
    packed_values = torch.arange(256, dtype=torch.long, device=weight.device)
    packed_lut = (
        nibble_lut[:, packed_values & 0x0f] |
        (nibble_lut[:, packed_values >> 4] << 4)
    ).reshape(-1)
    packed = weight.view(torch.uint8)
    rewritten = torch.empty_like(packed)
    if interleave_rows:
        assert weight.size(1) % 16 == 0
        half_rows = weight.size(1) // 2
        rewritten_view = rewritten.view(
            weight.size(0), half_rows // 8, 2, 8, weight.size(2))
    num_k_groups = raw_sf.size(2)
    num_rows_per_chunk = 1024
    for expert_idx in range(weight.size(0)):
        row_regions = ((0, weight.size(1), -1),) if not interleave_rows else (
            (0, half_rows, 0), (half_rows, weight.size(1), 1))
        for region_start, region_end, gate_up_idx in row_regions:
            for row_start in range(region_start, region_end, num_rows_per_chunk):
                row_end = min(row_start + num_rows_per_chunk, region_end)
                packed_chunk = packed[expert_idx, row_start:row_end].view(
                    row_end - row_start, num_k_groups, 16)
                delta_chunk = delta[
                    expert_idx, row_start:row_end].clamp_max(5).unsqueeze(-1)
                lut_idx = delta_chunk.to(torch.long) * 256 + packed_chunk.to(torch.long)
                rewritten_chunk = packed_lut[lut_idx].reshape(
                    row_end - row_start, -1)
                rewritten_chunk = _reorder_mxfp4_sign_bits_for_sm90(
                    rewritten_chunk)
                if interleave_rows:
                    relative_start = row_start - region_start
                    relative_end = row_end - region_start
                    rewritten_view[
                        expert_idx,
                        relative_start // 8:relative_end // 8,
                        gate_up_idx,
                    ].copy_(rewritten_chunk.view(-1, 8, weight.size(2)))
                else:
                    rewritten[expert_idx, row_start:row_end].copy_(rewritten_chunk)

    return rewritten.view(torch.int8), offsets.contiguous(), secondary


def transform_weights_for_fp8_mxfp4_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Prepare raw Humming-compatible MXFP4 weights for Hopper MegaMoE.

    Raw ``uint8`` or DeepGEMM ``int8`` weight tensors stay packed E2M1 with
    shape ``[E, N, K/2]``.
    Scale tensors are natural-layout UE8M0 bytes with shape ``[E, N, K/32]``.
    L1 gate/up rows are interleaved at granularity 8 for the fused SwiGLU
    epilogue; L2 needs no layout change.
    """
    l1_w, l1_sf = l1_weights
    l2_w, l2_sf = l2_weights
    l1_w = _normalize_mxfp4_packed_weight(l1_w)
    l2_w = _normalize_mxfp4_packed_weight(l2_w)
    l1_transformed = (
        _interleave_weights(l1_w),
        _interleave_weights(_normalize_mxfp4_ue8m0(l1_sf)),
    )
    l2_transformed = (l2_w.contiguous(), _normalize_mxfp4_ue8m0(l2_sf))
    return l1_transformed, l2_transformed


def transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    l1_global_scale: Optional[torch.Tensor] = None,
    l2_global_scale: Optional[torch.Tensor] = None,
):
    """Prepare Humming fused-E8M0 MXFP4 weights for Hopper MegaMoE.

    The returned triples contain rewritten packed E2M1 magnitudes with the
    internal SM90 sign-bit layout, bounded exponent offsets in ``[1, 12]``,
    and one FP32 secondary scale per expert.
    Optional Humming ``weight_scale_2`` tensors must also be per-expert ``[E]``
    scales; per-channel secondary scales are not supported by this kernel.
    """
    assert (l1_global_scale is None) == (l2_global_scale is None)
    l1_w, l1_sf, l1_secondary = _process_mxfp4_fused_e8m0(
        *l1_weights, interleave_rows=True)
    l2_w, l2_sf, l2_secondary = _process_mxfp4_fused_e8m0(*l2_weights)

    if l1_global_scale is not None:
        for scale, secondary in (
            (l1_global_scale, l1_secondary),
            (l2_global_scale, l2_secondary),
        ):
            assert scale.dim() == 1 and scale.numel() == secondary.numel()
            assert scale.device == secondary.device and torch.isfinite(scale).all()
        l1_secondary = (l1_secondary * l1_global_scale.float()).contiguous()
        l2_secondary = (l2_secondary * l2_global_scale.float()).contiguous()

    return (
        l1_w,
        _interleave_weights(l1_sf),
        l1_secondary,
    ), (l2_w.contiguous(), l2_sf, l2_secondary)


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: Tuple[torch.Tensor, torch.Tensor],
                     l2_weights: Tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                     recipe: Tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: Optional[float] = None,
                     fast_math: bool = True):
    _C.fp8_fp4_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )

def fp8_mega_moe(y: torch.Tensor,
                 l1_weights: Tuple[torch.Tensor, torch.Tensor],
                 l2_weights: Tuple[torch.Tensor, torch.Tensor],
                 sym_buffer: SM90SymmBuffer,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 recipe: Tuple[int, int, int] = (128, 128, 128),
                 activation: str = 'swiglu',
                 activation_clamp: Optional[float] = None,
                 fast_math: bool = True):
    """SM90 (Hopper) MegaMoE entry point.

    Expects FP8 e4m3 weights and block-(128, 128) float scale factors. The
    weight SF layout matches the convention used by ``DeepSeekV4FlashFp8`` /
    DeepEP, so the same SF tensors can be physically shared between the
    DeepEP path and this kernel.
    """
    _C.fp8_mega_moe(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )


def fp8_mxfp4_mega_moe(y: torch.Tensor,
                       l1_weights,
                       l2_weights,
                       sym_buffer: SM90SymmBuffer,
                       cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                       recipe: Tuple[int, int, int] = (1, 1, 32),
                       activation: str = 'swiglu',
                       activation_clamp: Optional[float] = None,
                       fast_math: bool = True):
    """SM90 MegaMoE with FP8 activations and packed MXFP4 weights.

    ``l1_weights`` and ``l2_weights`` must be matching raw pairs or processed
    triples from the raw or fused variants of
    ``transform_weights_for_fp8_mxfp4_*_mega_moe_sm90``. Raw Humming
    checkpoint tensors must pass through one of those transforms first.

    For processed triples, ``fast_math=True`` additionally updates scaled
    WGMMA fragments with packed BF16 FMA. Set ``fast_math=False`` to retain
    FP32 multiply-accumulate before the persistent sum is stored as BF16.
    """
    assert len(l1_weights) == len(l2_weights)
    assert len(l1_weights) in (2, 3)
    op = _C.fp8_mxfp4_mega_moe if len(l1_weights) == 2 \
        else _C.fp8_mxfp4_processed_mega_moe
    op(
        y,
        l1_weights, l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts, sym_buffer.num_topk,
        recipe,
        activation, activation_clamp,
        fast_math
    )
