# DeepGEMM

DeepGEMM is a unified, high-performance tensor core kernel library that brings together the key computation primitives of modern large language models — GEMMs (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning indexer, HyperConnection (HC), and more — into a single, cohesive CUDA codebase. All kernels are compiled at runtime via a lightweight Just-In-Time (JIT) module, requiring no CUDA compilation during installation.

DeepGEMM leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), but avoids heavy reliance on their templates or algebras. The library is designed for simplicity, with only a limited number of core kernel functions, making it a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques.

Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes.

## News

- 2026.04.16: Mega MoE, FP8xFP4 GEMM, FP4 Indexer, PDL, faster JIT compilation and more.
    - Please see [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) for more details.
    - For Mega MoE benchmarks, refer to [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316).
- 2025.09.28: DeepGEMM now supports scoring kernels (weighted ReLU MQA logits) for the lightning indexer for DeepSeek v3.2.
    - Please see [#200](https://github.com/deepseek-ai/DeepGEMM/pull/200) for more details.
- 2025.07.20: DeepGEMM now supports both SM90/SM100, and has a full refactor with a low-CPU-overhead JIT CPP module.
    - NVRTC and post-compilation SASS optimization are all disabled.
    - NVRTC will be supported later.
    - As NVCC 12.9 will automatically do the FFMA interleaving, all post optimizations will be no longer supported.
    - Please see [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112) for more details.
- 2025.05.14: DeepGEMM now offers weight gradient kernels for dense and MoE backward! See [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95) for details.
- 2025.05.07: DeepGEMM now supports NVRTC with up to 10x compilation speedup! See [#94](https://github.com/deepseek-ai/DeepGEMM/pull/94) for details. Please use `DG_JIT_USE_NVRTC=1` to enable it (may have performance loss with some cases).
- 2025.04.18: DeepGEMM now achieves up to **1550 TFLOPS** on H800! See [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74), [#78](https://github.com/deepseek-ai/DeepGEMM/pull/78), [#81](https://github.com/deepseek-ai/DeepGEMM/pull/81), [#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) and [340d988](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526) for details.

## Quick start

### Requirements

- NVIDIA SM90 or SM100 architecture GPU
- Python 3.8 or higher
- Compilers with C++20 support
- CUDA Toolkit:
    - CUDA 12.3 or higher for SM90
        - **We highly recommend 12.9 or higher for the best performance**
    - CUDA 12.9 or higher for SM100
- PyTorch 2.1 or higher
- CUTLASS 4.0 or higher (could be cloned by Git submodule)
- `{fmt}` library (could be cloned by Git submodule)

### Development

```bash
# Submodule must be cloned
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Link some essential includes and build the CPP JIT module
cat develop.sh
./develop.sh
```

### Installation

```bash
cat install.sh
./install.sh
```

Then, import `deep_gemm` in your Python project, and enjoy!

## Interfaces

#### Notices

This library provides optimized GEMM kernels for NVIDIA GPUs with a naming convention: `D = C + A @ B`. The input shape layout is NT (non-transposed A, transposed B). While the SM90 implementation supports only the NT memory layout (row-major, col-major), the SM100 implementation supports all memory layouts (NT, TN, NN, TT). For example, `fp8_gemm_nt` will do a `D = C + A @ B.T`

For both architectures, the LHS scaling factor is required to have a TMA-aligned and transposed layout. And the data format for the scaling factor of SM90 and SM100 is different:

- SM90 requires scaling factors in FP32 format.
- SM100 requires scaling factors in packed [UE8M0](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats) format, which packs 4 UE8M0 into a single `torch.int`.

Please note that operations like input transposition or FP8 casting must be handled separately by the user, please implement or fuse them into prior kernels independently. While the library provides some simple PyTorch utility functions, these may result in slower performance, but our primary focus is on optimizing the GEMM kernels themselves.

#### Normal dense GEMMs (non-grouped)

To perform a basic non-grouped FP8 GEMM, call the `fp8_gemm_{nt, nn, tn, tt}` function. For more details, please refer to the function documentation.

#### Grouped GEMMs (contiguous layout)

Unlike traditional grouped GEMMs in CUTLASS, DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for scenarios where experts in an MoE model share the same shape. For training forward passes or inference prefilling, where each expert may process a varying number of tokens, we concatenate these tokens into a single tensor, referred to as the "contiguous" layout. Note that each expert segment must be aligned to the GEMM M block size (`get_mk_alignment_for_contiguous_layout()`).  For more information, please refer to the `m_grouped_fp8_gemm_{nt, nn}_contiguous` function documentation.

We also provide a K-axis-grouped API for MoE weight backward (with M and N must remain fixed), please refer to `k_grouped_fp8_gemm_tn_contiguous` for more information.

#### Grouped GEMMs (masked layout)

During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions.

Use `m_grouped_fp8_gemm_nt_masked` for this purpose and consult the relevant documentation. An example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.

#### V3.2 MQA kernels for the indexer

The kernel family has two versions, non-paged (for prefilling) and paged (for decoding).
Take the non-paged version `fp8_mqa_logits` as an example. It has 6 inputs:

- `q`, E4M3 tensor with shape `[seq_len, num_heads, head_dim]`
- `kv`, E4M3 tensor (shaped as `[seq_len_kv, head_dim]`) with float SF (shaped as `[seq_len_kv]`)
- `weights`, float tensor with shape `[seq_len, num_heads]`
- `cu_seq_len_k_start` and `cu_seq_len_k_end`, int tensor with shape `[seq_len]`
- `clean_logits`, whether to clean the unfilled logits into `-inf`

The output tensor is shaped as `[seq_len, seq_len_kv]`, indicating token-to-token logits.
For each token `i` in `q`, it will iterate all tokens `j` from `[cu_seq_len_k_start[i], cu_seq_len_k_end[i])`,
and calculate the logit `out[i, j]` as:

```python
kv_j = kv[0][j, :] * kv[1][j].unsqueeze(1)  # [head_dim]
out_ij = q[i, :, :] @ kv_j  # [num_heads]
out_ij = out_ij.relu() * weights[i, :]  # [num_heads]
out_ij = out_ij.sum()  # Scalar
```

For more details and the paged version `fp8_paged_mqa_logits`, please refer to `tests/test_attention.py`.

#### Mega MoE

Mega MoE fuses and overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2 (FP8xFP4), and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory. Usage:

```python
# Allocate symmetric memory buffer
# NOTES: requires PyTorch >= 2.9
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
)

# Transform weights (FP4 with UE8M0 SF) into the required layout
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

# Copy inputs into the buffer before each call
# You may fuse these into previous kernels
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# Run the fused mega MoE kernel
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

For the full example with multi-process setup and benchmarking, please refer to `tests/test_mega_moe.py`.

##### SM90 FP8 x MXFP4 Mega MoE

Hopper uses an explicit API because its symmetric-buffer ABI and FP32 scale
layout differ from the SM100 backend. Packed-MXFP4 runs one cooperative
persistent kernel: dispatch, dynamic MegaMoE scheduling, routed linear 1/2,
optional shared experts, and combine overlap over a live-block ring. Logical
source metadata remains full-pool while activation/SF scratch is ring-sized.
Checkpoint weights follow the
[Humming fused-E8M0 contract](https://github.com/inclusionAI/humming/blob/a1e6bd3fec719640c6877462edc5f8dbe5b80061/humming/transform.py):
packed E2M1 bytes are shaped `[E_local, N, K/2]`, and natural-layout UE8M0
scales are shaped `[E_local, N, K/32]`.
`num_experts_global` is the global expert count and must be evenly sharded:
rank `r` supplies weights for the contiguous expert-ID interval
`[r * E_local, (r + 1) * E_local)`. `topk_idx` contains global expert IDs;
`-1` is the only masked-route sentinel.

```python
buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
    group, num_experts_global, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden,
)

# Required model-load transform. Each result is
# (processed_e2m1, relative_ue8m0, weight_scale_2_fp32[E_local]).
l1_weights, l2_weights = (
    deep_gemm.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
        raw_l1_weights, raw_l2_weights,
        l1_weight_scale_2=l1_weight_scale_2,
        l2_weight_scale_2=l2_weight_scale_2,
    )
)

buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf_fp32_k128)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_mxfp4_mega_moe(y, l1_weights, l2_weights, buffer)
```

Shared experts use replicated FP8 weights with natural row-major FP32
block-(128, 128) scales. Allocate the buffer with `num_shared_experts=S`, then
prepare the two weight pairs with
`transform_shared_weights_for_fp8_mxfp4_mega_moe_sm90`. The helper interleaves
only the L1 FP8 gate/up rows; it intentionally does not apply the SM100 UTCCP
scale transform. Before every launch, also copy the K128 input scales into the
separate M-major shared view (the shared activation data itself aliases `x`):

```python
shared_l1, shared_l2 = (
    deep_gemm.transform_shared_weights_for_fp8_mxfp4_mega_moe_sm90(
        shared_l1_fp8_and_fp32_scale,
        shared_l2_fp8_and_fp32_scale,
    )
)
buffer.shared_l1_acts_sf[:num_tokens].copy_(x_sf_fp32_k128)
deep_gemm.fp8_mxfp4_mega_moe(
    y, l1_weights, l2_weights, buffer,
    shared_l1_weights=shared_l1,
    shared_l2_weights=shared_l2,
)
```

The routed-weight runtime accepts only the processed triple above. The
model-load transform performs sign-bit reordering and fused exponent rebasing
once instead of in every decode tile. Optional `weight_scale_2` is per expert
(`[E_local]`); channel-wise secondary scales and unprocessed routed-weight
pairs are rejected.

Current SM90 constraints are `num_max_tokens_per_rank % 128 == 0`,
`hidden % 512 == 0` (and `hidden % 1024 == 0` when `hidden > 8192`),
`intermediate_hidden % 256 == 0`,
`num_topk + int(num_shared_experts > 0) <= 32`,
`num_topk <= num_experts`, `num_experts % num_ranks == 0`, and
`num_ranks <= 64`. FP8 dispatch, SwiGLU, and cooperative launch are required;
the symmetric buffer must be reallocated if its H/I/shared-expert or active-SM
layout contract changes. Eligible shapes remain subject to the exact JIT
kernel's cooperative occupancy check. Run the cross-rank Humming-contract
oracle smoke test with:

```bash
python tests/test_mega_moe_sm90.py --num-processes 8 --suite smoke
```

The SM90 persistent kernel also supports IKET-style in-kernel event tracing.
The profiler records the eight warp roles separately and includes dispatch,
scheduler waits, TMA producers, MXFP4 decode, WGMMA issue/completion waits,
scale promotion, L1/L2 epilogues, NVLink scatter, and combine TMA
issue/wait/reduction. Generate a Chrome trace JSON and open it directly in
[Perfetto](https://ui.perfetto.dev/):

```bash
python tests/bench_mega_moe_sm90.py \
  --num-processes 1 --no-dist \
  --model-config flash --batches 128 \
  --num-experts-override 8 \
  --profile-only \
  --in-kernel-trace /tmp/megamoe.json
```

For multi-rank profiling, put `{rank}` in the output path. Each rank trace is
mapped to the host monotonic clock using the midpoint of its launch/synchronize
bracket, and rank 0 also writes a merged trace with `merged` substituted for
`{rank}`. The `clock_alignment.uncertainty_ns` metadata quantifies the mapping's
host-side uncertainty; use Nsight Systems when tighter cross-GPU alignment is
required. Merge requires a complete retained `kernel` Begin/End interval on
every rank. A truncated interval is marked
`host_monotonic_midpoint_incomplete_kernel` and is never silently merged.
Increase `--in-kernel-trace-capacity` if the exported metadata reports dropped
events.

Trace format v5 stores a 48-bit payload in the unused upper bits of the existing
two-word event record, so richer workload metadata does not increase the
profiler buffer. Every Perfetto slice has a workload-specific title rather than
only an event type. For example, a WGMMA slice is named with its phase, global
expert, M/N tile, exact K interval, pipeline stage, and expanded-B slot. The
stable event type remains available as the trace category and `event_type` arg.
Each CTA/warp track carries its current `task` context forward to nested events,
and `scheduler.wait` is annotated with the Task it ultimately acquired.

Routed GEMM M uses the `expert_packed` coordinate space, not the source rank's
original token index. A `dispatch.select` range records the exact destination
expert/M, and the following `dispatch.pull` inherits that destination while
recording `(src_rank, src_token, src_topk)`. This maps the source route to
`(dst_rank, dst_global_expert, dst_m)` without duplicating fields in the 48-bit
payload. The destination M field retains the full uint32 workspace coordinate,
covering the 64-rank by 8192-token hotspot without imposing that benchmark size
as an API limit; source token IDs are also stored as uint32. `nvlink.scatter` events
are joined against these pull records and expose every output rank/token/top-k
destination for the 16-row stripe owned by that math warp. If any payload does
overflow, `coordinates_valid=false` prevents masked fields from being presented
as exact coordinates.

For high-load traces, select only the event families being investigated. Task
and dispatch correlation dependencies are enabled automatically:

```bash
python tests/bench_mega_moe_sm90.py \
  --num-processes 1 --no-dist \
  --model-config pro --batches 8192 --num-experts-override 8 \
  --profile-only --in-kernel-trace /tmp/megamoe.json \
  --in-kernel-trace-events wgmma.issue wgmma.wait pipeline.wait \
  --in-kernel-trace-ctas 0 --in-kernel-trace-warps 4
```

The CTA/warp selectors disable unselected device tracks before any
`%globaltimer` read. Warp indices `0-3` are producer roles and `4-7` are the
four math/epilogue roles, so selecting one math warp is useful for inspecting a
single WGMMA pipeline without producing hundreds of thousands of duplicate
tile events. Filtered traces always retain `kernel` begin/end records for clock
alignment. A filtered `nvlink.scatter` trace also retains only
`dispatch.select/pull` on producer warps 0/1 across all CTAs so route joins stay
complete; these additions are listed in `track_dependencies` metadata.

When a track reaches capacity, it writes one truncation sentinel and stops
reading `%globaltimer`; `attempted` and `dropped` are lower bounds whenever
`counts_are_lower_bounds=true`.

| Event ranges | Workload details visible in Perfetto |
| --- | --- |
| `kernel`, `setup` | Rank, CTA/warp role, token/rank/expert/top-k counts, H/I dimensions, and the 64x128x128 GEMM tile shape. |
| `dispatch.count`, `dispatch.pack` | Exact local token/route counts plus global-expert and destination-rank distributions for the CTA/warp's persistent partition. |
| `grid_barrier`, `dispatch.put`, `dispatch.nvlink`, `dispatch.cleanup` | Grid participant count, the exact expert/rank subset published by CTA0 or `sync_only`, cross-rank barrier participants, and cleanup generation scope. |
| `dispatch.select` | Destination rank, local/global expert, exact uint32 expert-packed M, coordinate validity, and overflow status. |
| `dispatch.pull` | Exact source rank/original token/top-k route and exact destination rank/local/global expert/expert-packed M. |
| `scheduler.wait` | The exact phase/expert/M/N Task returned by the wait, or `end_of_stream`. |
| `task`, `l1_dependency.wait` | Phase, routed/shared expert identity, valid M rows, M/N tile, N/K matrix shape, and logical coordinate spaces. |
| `tma.activation`, `tma.activation_scale`, `tma.weight`, `weight_scale.load`, `pipeline.wait` | Inherited task M/N/expert context plus pipeline stage, K block, and K128 range. |
| `mxfp4.decode` | Inherited task coordinates, packed K128 block, pipeline stage, and expanded-B slot. |
| `wgmma.issue`, `wgmma.wait` | Inherited task expert/M/N plus exact K32 start/count and resulting global K interval, stage, expanded-B slot, and accumulate mode. |
| `scale.promote` | Task M/N/expert context and the exact activation scale-factor K group being promoted. |
| `epilogue.l1`, `swiglu.quantize`, `tma.store_l1` | L1 task M tile, gate/up-interleaved N tile, and corresponding post-SwiGLU output-N interval. |
| `epilogue.l2`, `nvlink.scatter` | L2 task M/N tile plus each math warp's exact 16-row stripe and resolved per-row output rank/token/top-k destinations. |
| `combine.nvlink`, `combine` | Cross-rank combine barrier plus each CTA/warp's local output-token start/stride and full hidden width. |
| `combine.tma_issue`, `combine.tma_wait`, `combine.reduce` | Exact uint32 original output token, top-k/shared slot, hidden chunk index/count, and exact hidden interval. |
| `combine.tma_store` | Exact uint32 original output token and output hidden chunk interval. |

Instrumentation uses a separate JIT specialization; normal calls that do not
pass a `MegaMoeProfiler` compile out all device-side event operations.

#### Utilities

The library provides some utility functions besides the above kernels:

- `deep_gemm.set_num_sms` / `get_num_sms`: set/get the maximum SM count to use
- `deep_gemm.set_tc_util` / `get_tc_util`: set/get an approximated tensor core utilization ratio
- `deep_gemm.set_pdl` / `get_pdl`: enable/disable Programmatic Dependent Launch (PDL)
- `deep_gemm.set_mk_alignment_for_contiguous_layout` / `get_mk_alignment_for_contiguous_layout`: set/get the group-level M/K alignment for contiguous layout
- `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout`: get the theoretical minimum M/K alignment
- `deep_gemm.set_ignore_compile_dims`: configure dimensions to ignore during JIT compilation
- `deep_gemm.set_block_size_multiple_of`: constrain block sizes to be multiples of a given value
- `deep_gemm.transform_sf_into_required_layout`: transform scaling factors into the required layout
- `deep_gemm.get_tma_aligned_size`: get the required TMA alignment size
- `deep_gemm.get_mn_major_tma_aligned_tensor`: get a MN-major TMA-aligned tensor
- `deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor`: get a MN-major TMA-aligned tensor (with packing FP32 into UE8M0)
- `deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`: K-grouped GEMM packing kernel

The library also provides some environment variables, which may be useful:

- General
    - `DG_JIT_DEBUG`: `0` or `1`, print JIT debugging information, `0` by default
    - `DG_PRINT_CONFIGS`: `0` or `1`, print selected configs for each shape, `0` by default
- JIT cache
    - `DG_JIT_CACHE_DIR`: string, cache directory for compiled kernels, `$HOME/.deep_gemm` by default
- Compiler selection
    - `DG_JIT_USE_NVRTC`: `0` or `1`, use NVRTC instead of NVCC (faster compilation, may have lower performance for some cases), `0` by default
    - `DG_JIT_NVCC_COMPILER`: string, NVCC compiler path; defaults to `torch.utils.cpp_extension.CUDA_HOME`
    - `DG_JIT_CPP_STANDARD`: integer, C++ standard version, `20` by default
- Compiler output
    - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print compilation commands, `0` by default
    - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS output, `0` by default
    - `DG_JIT_PTXAS_CHECK`: `0` or `1`, assert no local memory usage in compiled kernels, `0` by default
    - `DG_JIT_PRINT_LOAD_TIME`: `0` or `1`, print kernel load time, `0` by default
- Debug and profiling
    - `DG_JIT_WITH_LINEINFO`: `0` or `1`, embed source line info for profiling tools, `0` by default
    - `DG_JIT_DUMP_ASM`: `0` or `1`, dump both PTX and SASS, `0` by default
    - `DG_JIT_DUMP_PTX`: `0` or `1`, dump PTX output, `0` by default
    - `DG_JIT_DUMP_SASS`: `0` or `1`, dump SASS output, `0` by default
    - `DG_COMM_KERNEL_DEBUG`: `0` or `1`, zero symmetric buffer before each Mega MoE call for debugging, `0` by default
    - `DG_USE_NVIDIA_TOOLS`: `0` or `1`, skip internal profiling when running under external NVIDIA tools, `0` by default
- Build options
    - `DG_SKIP_CUDA_BUILD`: `0` or `1`, skip CUDA extension build during installation, `0` by default
    - `DG_FORCE_BUILD`: `0` or `1`, force local build instead of downloading pre-built wheels, `0` by default
    - `DG_JIT_USE_RUNTIME_API`: `0` or `1`, use CUDA Runtime API for kernel loading (requires CUDA runtime >= 12.8), `0` by default

For additional examples and details, please refer to [the test code](tests/test_core.py) or review the corresponding Python documentation.

## Acknowledgement

DeepGEMM is inspired by the [CUTLASS](https://github.com/nvidia/cutlass) project. Thanks and respect to the developers!

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM: clean and efficient BLAS kernel library on GPU}, 
      author={Chenggang Zhao and Zhean Xu and Liang Zhao and Jiashi Li and Chenhao Xu and Anyi Xu and Shengyu Liu and Kexing Zhou and Kuai Yu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```
