# SM90 MXFP4 MegaMoE optimization log

This document records independently attributable SM90 MXFP4 MegaMoE
optimizations.  Kernel profiler numbers use a single H20 rank with 32 local
experts, which matches the Flash production case's per-rank expert count and
routed-token load (`E=256 / 8 ranks`, `M * topk = 128 * 6`).  End-to-end
numbers use all eight H20 GPUs and report the median of the maximum rank time.

## Baseline diagnosis

The correctness-first MXFP4 kernel expands packed E2M1 weights to an FP8
shared-memory tile, executes four K32 WGMMAs per BK128 tile, waits after every
WGMMA, and promotes every partial accumulator with activation and UE8M0 weight
scales.  The initial profile showed that this was a latency/instruction issue,
not a DRAM-bandwidth issue:

| Metric (Flash M128, single rank, E32) | MXFP4 L1 | FP8 L1 | MXFP4 L2 | FP8 L2 |
|---|---:|---:|---:|---:|
| NCU duration | 2.54 ms | 256.86 us | 1.37 ms | 161.54 us |
| Executed instructions | 527,785,696 | 47,050,871 | 270,613,797 | 28,951,700 |
| Memory throughput | 31.22% | 54.06% | 29.24% | 43.87% |
| Achieved occupancy | 9.38% | 15.93% | 15.36% | 17.98% |

The MXFP4 L1 kernel executed about 11.2 times as many instructions as FP8,
while using less of the memory subsystem.  NCU attributed about 35% of the L1
warp issue interval to an L1TEX scoreboard dependency.  NSYS independently
measured 2,325,542 ns for L1 and 1,260,387 ns for L2.  Tracing all eight ranks
was rejected as performance evidence because profiler interception perturbed
the distributed barriers; only the non-distributed single-rank trace is used
for kernel attribution.

## Iteration 1: stage UE8M0 weight scales in shared memory

### Hypothesis and implementation

Every output-column UE8M0 scale was loaded from global memory once per WGMMA
accumulator row.  For the fixed `BM64/BN128/BK128/1WG` MXFP4 schedule, a K tile
contains only `128 * (128 / 32) = 512` unique scale bytes.  The math warpgroup
now performs one coalesced 32-bit load per output column and reuses the four
K32 scale bytes from a single CTA-level shared-memory scratch region.

The scratch region is intentionally not replicated per pipeline stage.  It is
filled only after the current stage's full barrier and is consumed before the
single math warpgroup advances to the next stage.  Replicating it would
unnecessarily reduce some configurations from six pipeline stages to five.

### Correctness

Tested on eight H20 ranks with a clean device JIT cache and a rebuilt host
extension so the host heuristic and device shared-memory layout matched:

| Scenario | Weight format | `calc_diff` | Tolerance | Result |
|---|---|---:|---:|---|
| L1 smoke, one rank | MXFP4 | 0.0006 | 0.01 | PASS |
| Flash M128, eight ranks | MXFP4 | 0.0006 | 0.01 | PASS |
| Pro M128, eight ranks | MXFP4 | 0.0006 | 0.01 | PASS |

### NCU and NSYS attribution

| Metric (Flash M128, single rank, E32) | Iteration 0 L1 | Iteration 1 L1 | Change | Iteration 0 L2 | Iteration 1 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 2.54 ms | 2.23 ms | -12.2% | 1.37 ms | 1.14 ms | -16.8% |
| Executed instructions | 527,785,696 | 471,669,591 | -10.6% | 270,613,797 | 236,612,859 | -12.6% |
| Achieved occupancy | 9.38% | 9.38% | 0.0 pp | 15.36% | 15.23% | -0.13 pp |
| Dynamic shared memory | 200.37 KB | 200.88 KB | +0.51 KB | 200.37 KB | 200.88 KB | +0.51 KB |

The L1 long-scoreboard share fell only modestly, from about 35.0% to 34.2%,
but the global scale-load deduplication removed enough instructions and long
dependencies to produce a consistent kernel-time reduction.

| NSYS selected hot path | Iteration 0 | Iteration 1 | Change |
|---|---:|---:|---:|
| L1 kernel | 2,325,542 ns | 2,018,824 ns | -13.2% |
| L1-to-L2 gap | 140,289 ns | 136,928 ns | -2.4% |
| L2 kernel | 1,260,387 ns | 1,038,820 ns | -17.6% |

The nearly unchanged inter-kernel gap confirms that the improvement is inside
the two MegaMoE kernels rather than launch or communication overhead.

### Eight-rank performance

Command contract: `--num-processes 8`, `--num-max-tokens-per-rank 8192`,
`--repeats 3`, `--num-tests 20`, no masked routes, fast math enabled.  The
Iteration 0 column comes from the immediately preceding full baseline campaign
on the same Pod; Iteration 1 and FP8 were measured together after the change.

| Model | M | FP8 (us) | MXFP4 iter. 0 (us) | MXFP4 iter. 1 (us) | Iteration gain | Iter. 1 / FP8 |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 297.5 | 3,138.5 | 2,732.5 | 12.9% | 9.18x |
| Flash | 128 | 456.2 | 3,629.0 | 3,171.0 | 12.6% | 6.95x |
| Flash | 512 | 939.6 | 6,953.0 | 6,128.0 | 11.9% | 6.52x |
| Flash | 8192 | 9,871.0 | 83,579.0 | 71,744.0 | 14.2% | 7.27x |
| Pro | 8 | 698.6 | 8,931.0 | 7,396.0 | 17.2% | 10.59x |
| Pro | 128 | 1,245.5 | 14,192.0 | 11,712.0 | 17.5% | 9.40x |
| Pro | 512 | 2,447.8 | 21,424.0 | 17,866.0 | 16.6% | 7.30x |
| Pro | 8192 | 25,130.0 | 229,397.0 | 189,567.0 | 17.4% | 7.54x |

### Remaining bottleneck and next direction

This iteration is a consistent improvement but is not competitive with FP8.
The FP32 promotion work is unchanged: NCU still reports 33,554,432 fused plus
34,173,952 non-fused FP32 instructions in L1.  The next optimization must
remove work rather than only shorten scale loads.  The intended direction is
to vectorize packed E2M1 decoding and then adopt Humming-style bounded exponent
offsets so that the UE8M0 scale can be folded into the FP8 operand before
WGMMA, eliminating four K32 waits and per-accumulator scale promotion where
the numerical contract permits it.

## Iteration 2: vectorize packed E2M1 expansion

### Hypothesis and implementation

Iteration 1 left the packed-weight expansion scalar: every thread repeatedly
loaded one packed byte, decoded its two nibbles independently, calculated two
swizzled addresses, and issued two byte stores.  Iteration 2 processes four
packed bytes at a time:

- one aligned 32-bit packed-weight load produces eight E2M1 values;
- byte-wise SIMD comparisons/arithmetic decode four nibbles in parallel;
- two PRMT operations restore even/odd nibbles to logical K order; and
- one pair of 32-bit shared-memory stores writes the eight E4M3 bytes.

The eight-byte output begins at a K-aligned offset.  `Swizzle<3,4,3>` preserves
the low three address bits, so the vector store does not cross a B128 swizzle
segment.

### Correctness

The same clean-cache gates used by Iteration 1 passed:

| Scenario | Weight format | `calc_diff` | Tolerance | Result |
|---|---|---:|---:|---|
| L1 smoke, one rank | MXFP4 | 0.0006 | 0.01 | PASS |
| Flash M128, eight ranks | MXFP4 | 0.0006 | 0.01 | PASS |
| Pro M128, eight ranks | MXFP4 | 0.0006 | 0.01 | PASS |

### NCU and NSYS attribution

| Metric (Flash M128, single rank, E32) | Iteration 1 L1 | Iteration 2 L1 | Change | Iteration 1 L2 | Iteration 2 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 2.23 ms | 1.37 ms | -38.6% | 1.14 ms | 698.40 us | -38.7% |
| Executed instructions | 471,669,591 | 271,825,794 | -42.4% | 236,612,859 | 137,326,082 | -42.0% |
| Achieved occupancy | 9.38% | 9.41% | +0.03 pp | 15.23% | 15.25% | +0.02 pp |

| NSYS selected hot path | Iteration 1 | Iteration 2 | Change |
|---|---:|---:|---:|
| L1 kernel | 2,018,824 ns | 1,241,382 ns | -38.5% |
| L1-to-L2 gap | 136,928 ns | 143,424 ns | +4.7% |
| L2 kernel | 1,038,820 ns | 638,307 ns | -38.6% |

The inter-kernel gap changed by only 6.5 us, while both kernels became about
39% faster.  NCU and NSYS therefore agree that the improvement comes from the
vectorized expansion inside the kernel.

### Eight-rank performance

The measurement contract is unchanged from Iteration 1.  FP8 is the value
measured alongside Iteration 1; Iteration 2 changes only the compile-time
MXFP4 branch.

| Model | M | FP8 (us) | MXFP4 iter. 1 (us) | MXFP4 iter. 2 (us) | Iteration gain | Iter. 2 / FP8 |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 297.5 | 2,732.5 | 1,761.0 | 35.6% | 5.92x |
| Flash | 128 | 456.2 | 3,171.0 | 1,943.1 | 38.7% | 4.26x |
| Flash | 512 | 939.6 | 6,128.0 | 3,707.0 | 39.5% | 3.95x |
| Flash | 8192 | 9,871.0 | 71,744.0 | 44,219.0 | 38.4% | 4.48x |
| Pro | 8 | 698.6 | 7,396.0 | 4,448.0 | 39.9% | 6.37x |
| Pro | 128 | 1,245.5 | 11,712.0 | 7,074.0 | 39.6% | 5.68x |
| Pro | 512 | 2,447.8 | 17,866.0 | 10,744.0 | 39.9% | 4.39x |
| Pro | 8192 | 25,130.0 | 189,567.0 | 114,895.0 | 39.4% | 4.57x |

### Remaining bottleneck and next direction

The vectorized decoder removed roughly 42% of the executed instructions, but
the FP32 promotion counts are unchanged.  L1 still executes 33,554,432 fused
and 34,173,952 non-fused FP32 instructions for the representative profile.
The remaining 4-6x end-to-end gap cannot be closed by further decoder cleanup
alone.  The next iteration must fold bounded UE8M0 exponent offsets into the
FP8 B operand and reduce or remove the four K32 `warpgroup_wait<0>` plus
per-accumulator promotion sequences.

## Iteration 3: fuse bounded E8M0 exponents and group WGMMA promotion

### Hypothesis and implementation

Humming's fused-E8M0 preprocessing bounds each expert's residual exponent
range to 11. The offline transform now rewrites packed E2M1 values below the
retained exponent window and returns three tensors per layer:

- packed E2M1 weights, with L1 written directly into the gate/up-interleaved
  layout;
- K32 exponent offsets in `[1, 12]`; and
- one FP32 secondary scale per expert.

The SM90 decoder folds each offset into the expanded E4M3 operand. L1 can then
issue all four K32 WGMMAs in one group and promote once because its activation
scale is K128. L2 uses two groups of two WGMMAs because its activation scale is
K64. The raw UE8M0 pair API remains available as a separately JIT-cached
fallback. The optimized triple API is explicit, so the existing transform's
pair return contract is unchanged.

The implementation also preserves Humming's negative-zero result, applies the
fixed E2M1-to-E4M3 factor before it can underflow at UE8M0 codes 0/1, accepts an
optional per-expert `weight_scale_2`, rejects NaN and exponent deltas at or
above 128, and removes the full-size `torch.stack` temporary from L1 row
interleave.

### Correctness

All runs used a rebuilt host extension and a new device JIT cache. The
processed-weight reference reconstructs effective scales as
`secondary * 2**offset`; the forced-requant case uses exponent spread 12 so it
cannot pass without executing the payload rewrite.

| Scenario | Scale representation | `calc_diff` | Tolerance | Result |
|---|---|---:|---:|---|
| L1 smoke, one rank | processed triple | 0.0006 | 0.01 | PASS |
| L1 forced requant, one rank | processed triple | 0.0005 | 0.01 | PASS |
| L1 smoke, one rank | raw pair fallback | 0.0006 | 0.01 | PASS |
| Flash M128, eight ranks | processed triple | 0.0006 | 0.01 | PASS |
| Pro M128, eight ranks | processed triple | 0.0006 | 0.01 | PASS |

The format golden separately covers all 16 E2M1 nibbles for deltas 0 through
5, negative-zero normalization and regeneration, offsets and secondary scales,
optional per-expert global scales, raw endpoint handling, and explicit
rejection of UE8M0 code 255 and delta 128 or larger.

### NCU and NSYS attribution

The profiler contract remains single-rank Flash M128 with 32 experts. The
Iteration 3 JIT specialization is identified by
`kMXFP4Weights=true, kProcessedMXFP4Scales=true` in the kernel template.

| Metric | Iteration 2 L1 | Iteration 3 L1 | Change | Iteration 2 L2 | Iteration 3 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 1.37 ms | 1.05 ms | -23.4% | 698.40 us | 551.71 us | -21.0% |
| Executed instructions | 271,825,794 | 119,122,618 | -56.2% | 137,326,082 | 65,200,638 | -52.5% |
| Memory throughput | 20.47% | 40.09% | +19.62 pp | 19.55% | 38.61% | +19.06 pp |
| Achieved occupancy | 9.41% | 9.42% | +0.01 pp | 15.25% | 15.26% | +0.01 pp |

| NSYS selected hot path | Iteration 2 | Iteration 3 | Change |
|---|---:|---:|---:|
| L1 kernel | 1,241,382 ns | 970,148 ns | -21.8% |
| L1-to-L2 gap | 143,424 ns | 107,968 ns | -24.7% |
| L2 kernel | 638,307 ns | 509,282 ns | -20.2% |

Instruction removal is substantially larger than latency reduction because the
kernel is now latency- and synchronization-limited: only about 24% of scheduler
cycles have an eligible warp. L1 spends about 34.7% of its issue interval on
long-scoreboard dependencies; L2 spends about 51.3% waiting at CTA barriers.

### Eight-rank performance

The command contract is unchanged: three observations, 20 Kineto tests per
observation, maximum rank time, no masked routes, and transform excluded from
steady-state timing.

| Model | M | FP8 (us) | MXFP4 iter. 2 (us) | MXFP4 iter. 3 (us) | Iteration gain | Iter. 3 / FP8 |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 311.8 | 1,761.0 | 1,297.4 | 26.3% | 4.16x |
| Flash | 128 | 461.7 | 1,943.1 | 1,550.6 | 20.2% | 3.36x |
| Flash | 512 | 947.6 | 3,707.0 | 2,902.7 | 21.7% | 3.06x |
| Flash | 8192 | 9,871.0 | 44,219.0 | 33,580.0 | 24.1% | 3.40x |
| Pro | 8 | 719.4 | 4,448.0 | 3,364.0 | 24.4% | 4.68x |
| Pro | 128 | 1,238.7 | 7,074.0 | 5,334.0 | 24.6% | 4.31x |
| Pro | 512 | 2,433.0 | 10,744.0 | 8,071.0 | 24.9% | 3.32x |
| Pro | 8192 | 25,138.0 | 114,895.0 | 84,934.0 | 26.1% | 3.38x |

### Remaining bottleneck and next direction

Iteration 3 is consistently faster but does not yet beat FP8. SourceCounters
identify the next concrete target: the processed offset tensor still uses
natural `[E,N,K/32]` layout. A warp therefore loads one four-byte offset word
per N row with a 128-byte L1 stride or 224-byte L2 stride. The dominant L1
`LDG` produces 3,670,016 excessive L2 sectors; L2 produces 1,835,008.

The next iteration will preprocess offsets into a kernel-specific
`[E,K_block,N,4]` blocked layout. For each BK128 tile, the same 128 threads can
then issue one contiguous, coalesced 32-bit load each while retaining the
one-row-per-thread decoder and avoiding an additional shared-memory barrier.

## Rejected experiment: block processed exponent offsets by K tile

### Hypothesis and implementation

The processed offsets were temporarily rewritten from natural
`[E,N,K/32]` order to an opaque contiguous `[E,K/128,N,4]` kernel layout. This
made the 128 per-row four-byte loads of each BK128 tile contiguous. The host
API used a distinct four-dimensional contract so an old natural-layout triple
could not be silently interpreted with the new addressing.

All correctness gates passed with a rebuilt extension and a fresh JIT cache:
processed smoke `0.0006`, forced requantization `0.0005`, raw-pair fallback
`0.0006`, and both eight-rank Flash/Pro M128 `0.0006`.

### NCU and NSYS attribution

The SourceCounters hypothesis was locally correct but incomplete. Excessive
global-memory sectors collapsed, while the cache behavior and latency became
worse:

| Metric (Flash M128, single rank, E32) | Iteration 3 L1 | Blocked-offset L1 | Change | Iteration 3 L2 | Blocked-offset L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 1.05 ms | 1.15 ms | +9.5% | 551.71 us | 606.11 us | +9.9% |
| Executed instructions | 119,122,618 | 119,120,697 | ~0% | 65,200,638 | 65,203,520 | ~0% |
| L1/TEX hit rate | 83.34% | 3.82% | -79.52 pp | 83.34% | 5.33% | -78.01 pp |
| Eligible warps/cycle | 23.99% | 21.88% | -2.11 pp | ~24% | 21.44% | about -2.6 pp |
| Long-scoreboard issue share | 34.74% | 41.17% | +6.43 pp | -- | -- | -- |
| Excessive global sectors | 3,692,180 | 22,166 | -99.4% | 1,835,136 | 128 | -99.99% |

| NSYS selected hot path | Iteration 3 | Blocked offsets | Change |
|---|---:|---:|---:|
| L1 kernel | 970,148 ns | 1,058,916 ns | +9.2% |
| L1-to-L2 gap | 107,968 ns | 124,800 ns | +15.6% |
| L2 kernel | 509,282 ns | 557,859 ns | +9.5% |

The natural layout has poor lane-to-lane coalescing, but each lane walks its
row's offset words sequentially across K and reuses the fetched cache line.
The blocked layout converts every K step into a new cache-line dependency.
Actual L2 input traffic therefore did not fall enough to compensate for the
loss of temporal locality. This is a case where SourceCounters' theoretical
sector excess alone predicted the wrong optimization.

### Eight-rank performance and decision

FP8 and the experiment were measured together with the same three-by-20
contract used by Iteration 3.

| Model | M | Co-measured FP8 (us) | MXFP4 iter. 3 (us) | Blocked offsets (us) | Regression |
|---|---:|---:|---:|---:|---:|
| Flash | 8 | 309.2 | 1,297.4 | 1,401.0 | +8.0% |
| Flash | 128 | 438.9 | 1,550.6 | 1,659.8 | +7.0% |
| Flash | 512 | 922.3 | 2,902.7 | 3,043.0 | +4.8% |
| Flash | 8192 | 9,863.0 | 33,580.0 | 33,758.0 | +0.5% |
| Pro | 8 | 693.7 | 3,364.0 | 3,733.0 | +11.0% |
| Pro | 128 | 1,238.8 | 5,334.0 | 5,928.0 | +11.1% |
| Pro | 512 | 2,435.2 | 8,071.0 | 8,864.0 | +9.8% |
| Pro | 8192 | 25,121.0 | 84,934.0 | 91,092.0 | +7.2% |

The code change was rejected and reverted rather than committed. The profiler
reports are retained under the external Iteration 4 artifact directory.

The same SourceCounters run exposed a larger and more actionable bottleneck:
shared-memory packed-weight loads produce 35,722,802 excessive wavefronts in
L1 (74% of all wavefronts) and 18,162,176 in L2 (73%). The repeated 32-bit
loads use a 64-byte row stride and incur 16-way bank conflicts. The next
experiment therefore applies a matching B64 TMA descriptor/copy swizzle and
software address swizzle to the packed MXFP4 tile.

## Iteration 5: B64-swizzle the packed MXFP4 staging tile

### Hypothesis and implementation

The BK128 packed E2M1 row is exactly 64 bytes. Before this iteration, TMA wrote
the `128 x 64B` temporary tile without swizzle and every math-warp lane loaded
the same four-byte K position from a different N row. The 64-byte row stride
mapped a warp onto only two bank pairs, producing 16-way conflicts.

The host tensor maps and device TMA copies now both use B64 swizzle. The
decoder applies the matching `cute::Swizzle<2,4,3>` byte-address transform;
the hot processed path precomputes the row XOR mask, while the raw fallback
uses the full transform. Each packed stage is 8192 bytes and its shared-memory
base is 1024-byte aligned, so every stage satisfies the B64 base requirement.

### Correctness and review

The host extension was rebuilt after reverting the rejected offset-layout
experiment. Device code used a new JIT cache. Two independent source reviews
found no descriptor/copy, address, alignment, processed/raw, or ABI issues.

| Scenario | Scale representation | `calc_diff` | Tolerance | Result |
|---|---|---:|---:|---|
| L1 smoke, one rank | processed triple | 0.0006 | 0.01 | PASS |
| L1 forced requant, one rank | processed triple | 0.0005 | 0.01 | PASS |
| L1 smoke, one rank | raw pair fallback | 0.0006 | 0.01 | PASS |
| Flash M128, eight ranks | processed triple | 0.0006 | 0.01 | PASS |
| Pro M128, eight ranks | processed triple | 0.0006 | 0.01 | PASS |

### NCU and NSYS attribution

SourceCounters confirms that the packed loads moved from 16-way to 4-way
bank conflicts. The remaining vector stores are also four-way. Global-offset
accesses stay in the natural Iteration 3 layout, preserving its high L1 hit
rate.

| Metric (Flash M128, single rank, E32) | Iteration 3 L1 | Iteration 5 L1 | Change | Iteration 3 L2 | Iteration 5 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 1.050 ms | 940.74 us | -10.4% | 551.71 us | 503.78 us | -8.7% |
| Executed instructions | 119,122,618 | 121,869,919 | +2.3% | 65,200,638 | 66,499,004 | +2.0% |
| L1/TEX hit rate | 83.34% | 83.31% | -0.03 pp | 79.13% | 79.21% | +0.08 pp |
| Achieved occupancy | 9.42% | 9.45% | +0.03 pp | 15.26% | 15.26% | ~0 pp |
| Excessive shared wavefronts | 35,722,802 | 10,556,978 | -70.4% | 18,162,176 | 5,579,264 | -69.3% |
| Total shared wavefronts | 48,596,560 | 23,430,736 | -51.8% | 24,879,182 | 12,296,270 | -50.6% |

The address transform adds about 2% more instructions, but removing roughly
70% of excessive shared-memory wavefronts makes both kernels 9-10% faster in
NCU. Memory-throughput utilization falls because fewer bank-conflict replays
are counted as shared-memory activity; L1/L2 cache hit rates and L2 input
traffic remain effectively unchanged.

| NSYS selected hot path | Iteration 3 | Iteration 5 | Change |
|---|---:|---:|---:|
| L1 kernel | 970,148 ns | 866,499 ns | -10.7% |
| L1-to-L2 gap | 107,968 ns | 126,241 ns | +16.9% |
| L2 kernel | 509,282 ns | 451,361 ns | -11.4% |

The kernel gains are larger than the 18.3 us gap increase, so NSYS still
attributes a net hot-path improvement to the B64 change.

### Eight-rank performance

FP8 and MXFP4 were measured together with three observations and 20 Kineto
tests per observation. The table compares MXFP4 with the last accepted
Iteration 3 rather than the rejected blocked-offset experiment.

| Model | M | Co-measured FP8 (us) | MXFP4 iter. 3 (us) | MXFP4 iter. 5 (us) | Iteration gain | Iter. 5 / FP8 |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 329.7 | 1,297.4 | 1,153.9 | 11.1% | 3.50x |
| Flash | 128 | 460.7 | 1,550.6 | 1,397.5 | 9.9% | 3.03x |
| Flash | 512 | 939.7 | 2,902.7 | 2,587.5 | 10.9% | 2.75x |
| Flash | 8192 | 9,879.0 | 33,580.0 | 29,828.0 | 11.2% | 3.02x |
| Pro | 8 | 705.3 | 3,364.0 | 2,997.6 | 10.9% | 4.25x |
| Pro | 128 | 1,244.9 | 5,334.0 | 4,744.0 | 11.1% | 3.81x |
| Pro | 512 | 2,447.8 | 8,071.0 | 7,175.0 | 11.1% | 2.93x |
| Pro | 8192 | 25,140.0 | 84,934.0 | 75,629.0 | 11.0% | 3.01x |

### Remaining bottleneck and next direction

Iteration 5 is a consistent accepted improvement, but it remains 2.75-4.25x
slower than FP8. Each repeated packed-word `LDS` now has four wavefronts for
one ideal wavefront, and each 64-bit expanded-weight store has four wavefronts
for two ideal wavefronts. A warp still assigns one lane to one N row, so lanes
access a single K position across 32 row strides.

The next experiment will remap a warp iteration to eight N rows by four
contiguous packed words. With B64 input and B128 output swizzles, this should
cover all 32 banks per load and reach the two-wavefront minimum for 64-bit
stores. Each row's already-loaded exponent word can be shared from its owner
lane with a warp shuffle, preserving one global scale load per output row.

## Rejected experiment: 8-row by 4-word warp decode mapping

### Hypothesis and implementation

The processed decoder was temporarily remapped so `lane / 4` selected one of
eight rows and `lane % 4` selected four contiguous packed words. Four row
groups and four K32 groups still covered every one of the `128 x 16` packed
words exactly once. Each target row obtained its original exponent word with
`__shfl_sync` from that row's owner lane. Two independent index/address reviews
and an exhaustive host-side coordinate enumeration found no overlap or gap.

Processed smoke, forced requantization, raw fallback, and eight-rank Flash/Pro
M128 correctness all passed with the same `0.0005-0.0006` differences as
Iteration 5.

### NCU and NSYS attribution

The remap eliminated packed-load bank conflicts, but not the expanded 64-bit
stores. NVIDIA services `STS.64` as two half-warp transactions. Within each
half warp, this lane order touched only 16 banks twice, so the stores still
used four wavefronts instead of the two-wavefront ideal. The additional row
address arithmetic and four shuffles per thread increased executed
instructions enough to cancel most of the load-side gain.

| Metric (Flash M128, single rank, E32) | Iteration 5 L1 | Warp-tiled L1 | Change | Iteration 5 L2 | Warp-tiled L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 940.74 us | 940.48 us | -0.03% | 503.78 us | 488.86 us | -3.0% |
| Executed instructions | 121,869,919 | 129,596,259 | +6.3% | 66,499,004 | 69,741,293 | +4.9% |
| Excessive shared wavefronts | 10,556,978 | 4,265,522 | -59.6% | 5,579,264 | 2,433,536 | -56.4% |
| Total shared wavefronts | 23,430,736 | 17,139,280 | -26.9% | 12,296,270 | 9,150,542 | -25.6% |

| NSYS selected hot path | Iteration 5 | Warp tiled | Change |
|---|---:|---:|---:|
| L1 kernel | 866,499 ns | 859,972 ns | -0.8% |
| L1-to-L2 gap | 126,241 ns | 125,344 ns | -0.7% |
| L2 kernel | 451,361 ns | 444,194 ns | -1.6% |

### Eight-rank performance and decision

The full paired campaign used the standard three observations and 20 tests.
Changes are relative to accepted Iteration 5.

| Model | M | Co-measured FP8 (us) | MXFP4 iter. 5 (us) | Warp tiled (us) | Change |
|---|---:|---:|---:|---:|---:|
| Flash | 8 | 305.8 | 1,153.9 | 1,155.7 | +0.2% |
| Flash | 128 | 462.3 | 1,397.5 | 1,371.9 | -1.8% |
| Flash | 512 | 967.8 | 2,587.5 | 2,561.2 | -1.0% |
| Flash | 8192 | 9,880.0 | 29,828.0 | 29,591.0 | -0.8% |
| Pro | 8 | 704.0 | 2,997.6 | 2,977.0 | -0.7% |
| Pro | 128 | 1,247.1 | 4,744.0 | 4,739.0 | -0.1% |
| Pro | 512 | 2,408.5 | 7,175.0 | 7,225.0 | +0.7% |
| Pro | 8192 | 25,218.0 | 75,629.0 | 75,481.0 | -0.2% |

The mixed `-1.8%` to `+0.7%` result is too small and inconsistent for the
extra mapping complexity. The device change was reverted and only this
profiler record is retained.

The SASS counters identify a corrected follow-up: arrange each half warp as
eight rows by two packed words, with the high half selecting the other two
words. In coordinates,
`row = (lane % 16) / 2` and
`word = (lane / 16) * 2 + lane % 2`.
For B64 loads the full warp then touches all 32 banks once; for B128 `STS.64`,
each half warp also touches all 32 banks once. This preserves the exact same
logical coverage and shuffle ownership while targeting both remaining replay
sources rather than only the loads.

## Iteration 7: half-warp-tile the processed MXFP4 decoder

### Hypothesis and implementation

Iteration 7 keeps the accepted B64 packed tile and changes only the processed
decoder's lane coordinates. Within a warp, both half warps cover the same
eight rows; lanes 0-15 select packed words 0-1 and lanes 16-31 select words
2-3. Four row groups and four K32 groups cover the complete B128 output tile:

```text
row_in_group = (lane % 16) / 2
word_in_k32 = (lane / 16) * 2 + lane % 2
row = warp * 32 + row_group * 8 + row_in_group
packed_word = k32 * 4 + word_in_k32
```

Each row's original 32-bit offset word is broadcast from lane
`row_group * 8 + row_in_group`. The raw-scale fallback is unchanged. This
mapping gives each full-warp B64 load one request per bank, and each half-warp
B128 `STS.64` transaction one 64-bit bank word per bank.

### Correctness and review

Two independent reviews found no coordinate, scale-owner, swizzle, aliasing,
out-of-bounds, synchronization, or raw-fallback issue. A new exact mapping
contract enumerates all 2,048 `(N, packed-word)` destinations, requires every
destination exactly once, and checks the K32 scale byte and source-row owner.
This closes the gap left by the aggregate `calc_diff` check, where a sparse
lane error could otherwise be hidden by the tolerance.

| Scenario | Scale representation | `calc_diff` | Tolerance | Result |
|---|---|---:|---:|---|
| Mapping contract, 2,048 destinations | processed triple | exact | exact | PASS |
| L1 smoke, one rank | processed triple | 0.0006 | 0.01 | PASS |
| L1 forced requant, one rank | processed triple | 0.0005 | 0.01 | PASS |
| L1 smoke, one rank | raw pair fallback | 0.0006 | 0.01 | PASS |
| Flash M128, eight ranks | processed triple | 0.0006 | 0.01 | PASS |
| Pro M128, eight ranks | processed triple | 0.0006 | 0.01 | PASS |

### NCU and NSYS attribution

SourceCounters confirms that the targeted processed-decoder `LDS` and
`STS.64` instructions now have zero excessive shared-memory wavefronts. The
small residual totals come from other kernel regions.

| Metric (Flash M128, single rank, E32) | Rejected iter. 6 L1 | Iteration 7 L1 | Change | Rejected iter. 6 L2 | Iteration 7 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 940.48 us | 911.10 us | -3.1% | 488.86 us | 473.22 us | -3.2% |
| Executed instructions | 129,596,259 | 128,956,122 | -0.5% | 69,741,293 | 69,289,384 | -0.6% |
| Excessive shared wavefronts | 4,265,522 | 71,218 | -98.3% | 2,433,536 | 336,384 | -86.2% |
| Total shared wavefronts | 17,139,280 | 12,944,976 | -24.5% | 9,150,542 | 7,053,390 | -22.9% |
| L1/TEX hit rate | 83.30% | 83.30% | ~0 pp | 79.21% | 79.21% | ~0 pp |

Relative to accepted Iteration 5, NCU duration improves by 3.2% for L1 and
6.1% for L2 even though the shuffle-based mapping still executes 5.8% and
4.2% more instructions. Eliminating the shared replay cost therefore more
than pays for the coordinate and shuffle work.

| NSYS selected hot path | Rejected iter. 6 | Iteration 7 | Change |
|---|---:|---:|---:|
| L1 kernel | 859,972 ns | 825,700 ns | -4.0% |
| L1-to-L2 gap | 125,344 ns | 126,881 ns | +1.2% |
| L2 kernel | 444,194 ns | 428,737 ns | -3.5% |

The 1.5 us gap increase is negligible relative to the 49.7 us removed from
the two kernels, so NCU and NSYS agree on the source of the gain.

### Eight-rank performance

FP8 and Iteration 7 were measured together with the standard three
observations and 20 Kineto tests. The gain column uses accepted Iteration 5;
the rejected Iteration 6 is not treated as a baseline.

| Model | M | Co-measured FP8 (us) | MXFP4 iter. 5 (us) | MXFP4 iter. 7 (us) | Iteration gain | Iter. 7 / FP8 |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 8 | 313.1 | 1,153.9 | 1,135.7 | 1.6% | 3.63x |
| Flash | 128 | 435.8 | 1,397.5 | 1,338.3 | 4.2% | 3.07x |
| Flash | 512 | 920.6 | 2,587.5 | 2,460.0 | 4.9% | 2.67x |
| Flash | 8192 | 9,884.0 | 29,828.0 | 28,797.0 | 3.5% | 2.91x |
| Pro | 8 | 702.7 | 2,997.6 | 2,899.6 | 3.3% | 4.13x |
| Pro | 128 | 1,249.4 | 4,744.0 | 4,601.0 | 3.0% | 3.68x |
| Pro | 512 | 2,425.0 | 7,175.0 | 6,971.0 | 2.8% | 2.87x |
| Pro | 8192 | 25,143.0 | 75,629.0 | 73,156.0 | 3.3% | 2.91x |

Iteration 7 is accepted because all eight production points improve, and it
also beats the rejected Iteration 6 by 1.7-4.0% at every point. Across the
four M values, Iteration 0 to Iteration 7 geometric-mean speedup is 2.80x for
Flash and 3.09x for Pro.

### Remaining bottleneck and next direction

The accepted kernel is still 2.67-4.13x slower than FP8 and therefore has not
reached the SOTA goal. With decoder shared conflicts removed, NCU again makes
the natural-layout exponent loads the largest concrete source issue:
3,692,178 excessive global sectors in L1 and 1,835,136 in L2. The rejected
blocked layout showed that fully transposing the tensor destroys useful K-step
cache locality. The next experiment should retain natural `[E,N,K/32]`
storage but cooperatively load each row's four-byte word with fewer active
lanes, then broadcast it within a small row group. This can reduce global
sectors without changing the K traversal order or the checkpoint-facing
layout. A larger structural alternative is to overlap expansion of the next
pipeline stage with the current WGMMA, since H20 has no native FP4 tensor-core
instruction and runtime E2M1-to-E4M3 expansion remains unavoidable.

## Rejected experiment: four-stage `uint4` exponent prefetch

### Hypothesis and implementation

The processed path temporarily replaced four consecutive 32-bit exponent
loads from the natural `[E,N,K/32]` row with one 128-bit `uint4` load. The four
words stayed in registers and were selected by `k_block_idx % 4`. This retained
the accepted row-major layout and attempted to amortize the sector cost across
four BK128 stages without reintroducing the blocked-layout cache regression.
The raw-scale fallback remained on its original scalar load.

Single-rank processed, forced-requantization, and raw-fallback correctness all
passed (`calc_diff=0.0005-0.0006`, tolerance `0.01`). Detailed NCU reported the
same 168 registers per thread and zero local-memory spill.

### NCU and NSYS attribution

SASS emitted the intended `LDG.E.128.CONSTANT`. SourceCounters shows that it
removed most redundant global sectors, but the run-time cache-word selection
added branches and the end-to-end hot path did not improve consistently.

| Metric (Flash M128, single rank, E32) | Iteration 7 L1 | `uint4` L1 | Change | Iteration 7 L2 | `uint4` L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 911.10 us | 901.22 us | -1.1% | 473.22 us | 480.38 us | +1.5% |
| Executed instructions | 128,956,122 | 128,921,259 | ~0% | 69,289,384 | 69,207,124 | -0.1% |
| Excessive global sectors | 3,692,178 | 546,455 | -85.2% | 1,835,136 | 262,272 | -85.7% |
| Total global sectors | 4,270,924 | 1,125,605 | -73.6% | 2,368,300 | 795,806 | -66.4% |
| Registers per thread | 168 | 168 | 0 | 168 | 168 | 0 |
| Local-memory spill requests | 0 | 0 | 0 | 0 | 0 | 0 |

| NSYS selected hot path | Iteration 7 | `uint4` prefetch | Change |
|---|---:|---:|---:|
| L1 kernel | 825,700 ns | 828,738 ns | +0.4% |
| L1-to-L2 gap | 126,881 ns | 140,001 ns | +10.3% |
| L2 kernel | 428,737 ns | 442,337 ns | +3.2% |

### Eight-rank performance and rejection decision

The full paired campaign used three observations and 20 Kineto tests. Changes
are relative to accepted Iteration 7; lower is better.

| Model | M | Co-measured FP8 (us) | Iteration 7 (us) | `uint4` prefetch (us) | Change |
|---|---:|---:|---:|---:|---:|
| Flash | 8 | 334.5 | 1,135.7 | 1,125.4 | -0.9% |
| Flash | 128 | 454.1 | 1,338.3 | 1,326.7 | -0.9% |
| Flash | 512 | 937.5 | 2,460.0 | 2,479.6 | +0.8% |
| Flash | 8192 | 9,877.0 | 28,797.0 | 28,609.0 | -0.7% |
| Pro | 8 | 705.9 | 2,899.6 | 2,956.0 | +1.9% |
| Pro | 128 | 1,228.3 | 4,601.0 | 4,638.0 | +0.8% |
| Pro | 512 | 2,444.2 | 6,971.0 | 7,015.0 | +0.6% |
| Pro | 8192 | 25,148.0 | 73,156.0 | 73,746.0 | +0.8% |

Five of eight production points regress, and the geometric-mean change is a
0.32% slowdown. Independent review also found that an unconditional `uint4`
load requires each scale row to contain a multiple of 16 bytes. That silently
narrows the public `%128` hidden-size contract to `%512`; for example, a valid
640-wide shape would either fail the temporary static assertion or over-read
the final 20-byte scale row. A tail fallback could preserve the shape contract,
but cannot justify the measured regression and added control flow. The device
change was reverted, so accepted Iteration 7 remains the implementation and
only this profiler record is retained.

## Rejected experiment: natural-layout scale TMA

NCU on accepted Iteration 7 attributes virtually all excessive global sectors
to the processed exponent load: one `LDG.E.CONSTANT` contributes 3,670,016 of
3,692,186 excessive sectors in L1 and 1,835,008 of 1,835,136 in L2. A narrow
experiment therefore added a stage-private 512-byte `[N128,K32x4]` shared tile
and attempted to load it through the packed-B producer's existing TMA barrier.
The math warpgroup kept the accepted half-warp decoder mapping and replaced
only the strided global scale word with a conflict-free shared load. Legal
shapes whose K32 row stride was not 16-byte aligned retained the original
scalar fallback, so the public `%128` shape contract was not narrowed.

The CUDA tensor maps encoded and both kernels JIT-compiled, but the first
processed smoke launch failed with `CUDA_ERROR_ILLEGAL_INSTRUCTION`. The
natural layout makes the contiguous TMA box dimension only four bytes; Hopper
cannot execute that 2D TMA load even though the whole `128 x 4` tile is 512
bytes. Compute Sanitizer localized the failure to the L1 kernel around
`+0x7fe0`; disassembly shows the new scale `UTMALDG.2D` immediately beside the
valid packed-weight `UTMALDG.2D` at `+0x8150/+0x8160`.

Expanding every row to the 16-byte minimum would transfer four adjacent BK128
scale groups for every stage. That multiplies useful scale bytes and
stage-private shared storage by four, while Iteration 8 already demonstrated
that reducing scale-sector counts alone does not improve the hot path. A
blocked/transposed scale layout could make N contiguous, but the equivalent
Iteration 4 layout lost natural K-step locality and regressed both kernels.
The implementation was therefore reverted at the correctness gate; NCU,
NSYS, and the eight-rank campaign were intentionally not run on an invalid
kernel. The next structural experiment should target the sampled latency
hotspot instead: overlap expansion of the next packed-B stage with WGMMA on
the current stage using the two currently idle non-epilogue warps.

## Rejected sub-experiment: two-warp decoder overlap with a 64-register frontend

### Hypothesis and implementation

The processed path moved E2M1-to-E4M3 expansion out of the math warpgroup and
onto the two previously idle non-epilogue warps. Each decoder warp owns 64 of
the BN128 rows in two 32-row halves, waits for the packed-B TMA stage, expands
the complete BK128 tile into the existing swizzled shared-memory buffer, and
arrives on a new per-stage decoded-ready barrier. The math warpgroup waits on
that barrier and can therefore consume one stage while the frontend advances
to the next stage. A shared-to-WGMMA async-proxy fence precedes each readiness
arrival.

The launch-side shared-memory calculation adds one eight-byte barrier per
MXFP4 pipeline stage. Five-stage Flash kernels consequently use 200.92 KiB of
dynamic shared memory, only 40 bytes more than accepted Iteration 7. The raw
scale fallback remains in the math warpgroup and gained the corresponding
generic shared-store-to-WGMMA proxy fence. The first overlap variant capped
the four non-epilogue frontend warps at 64 registers each.

The mapping oracle now checks all 2,048 packed words and 16,384 expanded
elements, exact ownership and use of every K32 scale, and exact-once coverage
of the physical B64 source bytes and B128 destination bytes after their CUTE
swizzles.

### Correctness gate

| Scenario | Scale representation | `calc_diff` | Tolerance | Result |
|---|---|---:|---:|---|
| Mapping and physical-byte oracle | processed triple | exact | exact | PASS |
| L1 smoke, one rank | processed triple | 0.0006 | 0.01 | PASS |
| L1 forced requant, one rank | processed triple | 0.0005 | 0.01 | PASS |
| L1 smoke, one rank | raw pair fallback | 0.0006 | 0.01 | PASS |
| Flash M128 L3, one rank | processed triple | 0.0006 | 0.01 | PASS |
| Pro M128 L3, one rank | processed triple | 0.0006 | 0.01 | PASS |
| Flash M128 L3, one rank | raw pair fallback | 0.0006 | 0.01 | PASS |
| Flash M128 L3, eight ranks | processed triple | 0.0006 | 0.01 | PASS |
| Pro M128 L3, eight ranks | processed triple | 0.0006 | 0.01 | PASS |

An additional 640-wide ancillary case stops before kernel launch because the
existing MegaMoE scale-buffer layout requires 16-byte TMA alignment while a
`hidden / 32` row contains 20 bytes. That broader layout-contract mismatch is
not counted as a failure of this kernel candidate, and the temporary test case
was removed instead of expanding the experiment's scope.

### Eight-rank quick gate

The standard three observations and 20 Kineto tests were run at M128. The
candidate was already more than twice as slow for both production shapes, so
the remaining six M points were intentionally not run.

| Model | Accepted Iteration 7 (us) | Decoder overlap reg64 (us) | Candidate range (us) | Candidate / Iter. 7 | Change |
|---|---:|---:|---:|---:|---:|
| Flash M128 | 1,338.252 | 2,712.259 | 2,692.014-2,713.262 | 2.027x | +102.67% |
| Pro M128 | 4,601.000 | 10,181.000 | 10,166-10,210 | 2.213x | +121.28% |

Flash rank-zero L1 takes 1,996-2,017 us and L2 takes 695.259-697.262 us.
Pro rank-zero L1 takes 7,368-7,401 us and L2 takes 2,792-2,807 us. Both
phases therefore regress rather than exposing a single isolated tail.

### NCU and NSYS attribution

Detailed NCU and SourceCounters use one rank, Flash M128, and 32 local
experts, matching the previous iteration's attribution boundary.

| Metric | Iteration 7 L1 | reg64 L1 | Change | Iteration 7 L2 | reg64 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| NCU duration | 911.104 us | 2,058.080 us | +125.89% | 473.216 us | 740.288 us | +56.44% |
| Executed instructions | 128,956,122 | 121,720,099 | -5.61% | 69,289,384 | 66,207,438 | -4.45% |
| Local-memory spill requests | 0 | 2,915,820 | new | 0 | 1,206,072 | new |
| Spill-request overhead | 0% | 100% | +100 pp | 0% | 100% | +100 pp |
| Achieved occupancy | 9.45% | 12.51% | +3.06 pp | 15.27% | 18.10% | +2.83 pp |
| Excessive global sectors | 3,692,178 | 3,692,185 | ~0% | 1,835,136 | 1,835,136 | 0% |
| Excessive shared wavefronts | 71,218 | 71,218 | 0% | 336,384 | 336,384 | 0% |

The launch report still shows 168 registers per thread because that is the
kernel-wide maximum from the math role. SourceCounters SASS nevertheless
contains 96 L1 and 75 L2 local `LDL`/`STL` instructions whose aggregate
executions exactly match the 2,915,820 and 1,206,072 spill requests. The
candidate executes fewer instructions and reports higher achieved occupancy,
yet takes substantially longer; the new local-memory traffic and the two-warp
decoder critical path dominate both apparent improvements.

| NSYS selected hot path | Iteration 7 | reg64 overlap | Change |
|---|---:|---:|---:|
| L1 kernel | 825,700 ns | 1,932,199 ns | +134.01% |
| L1-to-L2 gap | 126,881 ns | 170,049 ns | +34.02% |
| L2 kernel | 428,737 ns | 687,714 ns | +60.40% |
| L1 + gap + L2 | 1,381,318 ns | 2,789,962 ns | +101.98% |

This 64-register variant is rejected. The overlap direction is not yet
discarded because the register cap is a controlled confounder: raising it to
168 registers keeps the CTA role budget at 54,272 registers, below the 64,512
limit, and should remove or sharply reduce spills. The next sub-experiment
therefore changes only that cap and reruns the same correctness, M128 quick
gate, and NCU spill checks before considering a full eight-point campaign.

### Follow-up: 168-register frontend

Changing only the processed frontend cap from 64 to 168 registers preserves
the 54,272-register CTA role budget. Processed L1 smoke and forced requant pass
at `calc_diff=0.0006` and `0.0005`; Flash and Pro eight-rank M128 L3 both pass
at `0.0006`. The higher cap removes every local spill reported by NCU and
recovers most of the reg64 regression, but it remains more than 20% slower than
accepted Iteration 7.

| Model | Iteration 7 (us) | reg64 (us) | reg168 (us) | reg168 range (us) | reg168 / Iter. 7 | reg168 / reg64 |
|---|---:|---:|---:|---:|---:|---:|
| Flash M128 | 1,338.252 | 2,712.259 | 1,628.709 | 1,628.128-1,628.727 | 1.217x | 0.600x |
| Pro M128 | 4,601.000 | 10,181.000 | 5,646.000 | 5,640-5,688 | 1.227x | 0.555x |

The candidate improves 39.95% and 44.54% over reg64, proving that spilling
caused most of that variant's regression. It still regresses 21.70% and
22.71% against Iteration 7, so the full eight-point campaign remains gated.

| NCU metric (Flash M128, one rank, E32) | Iteration 7 L1 | reg168 L1 | Change | Iteration 7 L2 | reg168 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| Duration | 911.104 us | 1,115.904 us | +22.48% | 473.216 us | 566.784 us | +19.77% |
| Executed instructions | 128,956,122 | 129,363,240 | +0.32% | 69,289,384 | 69,516,313 | +0.33% |
| Local-memory spill requests | 0 | 0 | 0 | 0 | 0 | 0 |
| Achieved occupancy | 9.45% | 12.55% | +3.10 pp | 15.27% | 18.07% | +2.80 pp |
| Excessive global sectors | 3,692,178 | 3,692,180 | ~0% | 1,835,136 | 1,835,136 | 0% |
| Excessive shared wavefronts | 71,218 | 71,218 | 0% | 336,384 | 336,384 | 0% |

Instruction volume, cache-access path, and conflict counts are effectively the
same as Iteration 7, while spill is zero. PC sampling instead identifies the
new readiness dependency as the critical path. The hottest L1 and L2 PCs are
the branches immediately following the math warpgroup's
`SYNCS.PHASECHK.TRANS64.TRYWAIT` on the decoded-ready barrier; they collect
21,090 and 6,500 long-scoreboard samples respectively.

| PC-sampling share | Iteration 7 L1 | reg168 L1 | Iteration 7 L2 | reg168 L2 |
|---|---:|---:|---:|---:|
| Long scoreboard | 27.42% | 58.81% | 16.55% | 34.82% |
| Barrier | 21.06% | 10.69% | 55.30% | 44.03% |
| Wait | 16.18% | 10.03% | 8.53% | 6.13% |

| NSYS selected hot path | Iteration 7 | reg168 overlap | Change |
|---|---:|---:|---:|
| L1 kernel | 825,700 ns | 1,032,900 ns | +25.09% |
| L1-to-L2 gap | 126,881 ns | 143,809 ns | +13.34% |
| L2 kernel | 428,737 ns | 511,073 ns | +19.20% |
| L1 + gap + L2 | 1,381,318 ns | 1,687,782 ns | +22.19% |

The 168-register two-warp variant is therefore rejected as well. With spills
removed, two decoder warps still do the work that four math warps performed in
Iteration 7, and the attempted overlap cannot hide that halved decode
parallelism. The next controlled implementation keeps the stage pipeline but
has both TMA producer warps join the two idle warps after issuing A and B,
giving all four frontend warps one 32-row quadrant. The decoded-ready barrier
then expects four arrivals. This restores the accepted decoder width while
retaining the possibility of overlapping the next stage with current WGMMA.

## Rejected experiment: four-warp frontend decoder pipeline

### Hypothesis and implementation

This controlled follow-up keeps the reg168 stage pipeline and restores the
four-warp decoder width of accepted Iteration 7. The A and B TMA producers
issue their loads and then join the two previously idle frontend warps. Each
of the four warps waits for the same packed-B stage and expands one contiguous
32-row quadrant of the BN128/BK128 tile. The decoded-ready barrier therefore
expects four arrivals instead of two.

The mapping oracle exhaustively checks 2,048 packed words, 16,384 expanded
elements, all 128 scale owners, four uses of every `(row, K32)` scale, and
exact-once coverage of all 8,192 B64 source and 16,384 B128 destination bytes.
It also proves that the device's row-XOR B64 address expression equals the
canonical CUTE swizzle for every packed word. A separate review found no
barrier, TMA-completion, phase-parity, dummy-CTA, or async-proxy ordering
blocker. The processed frontend executes its register deallocation once as a
warpgroup-uniform instruction and statically requires the dispatch/frontend
boundary to be four-warp aligned.

This layout does not provide pure producer/consumer overlap: after issuing a
stage, both TMA producer warps wait for it and participate in decode, so they
cannot run ahead to fill the following stage. The experiment tests whether
restored decode width is nevertheless enough to retain any useful overlap.

### Correctness gate

| Scenario | Scale representation | `calc_diff` | Tolerance | Result |
|---|---|---:|---:|---|
| Mapping and physical-byte oracle | processed triple | exact | exact | PASS |
| L1 smoke, one rank | processed triple | 0.0006 | 0.01 | PASS |
| L1 forced requant, one rank | processed triple | 0.0005 | 0.01 | PASS |
| Flash M128 L3, eight ranks | processed triple | 0.0006 | 0.01 | PASS |
| Pro M128 L3, eight ranks | processed triple | 0.0006 | 0.01 | PASS |

The L3 cases traverse more stages than the five-stage pipeline depth, so they
also exercise barrier reuse and phase wrap. The raw-scale and FP8 paths retain
their preceding implementation; no candidate-specific behavior is enabled
for them.

### Eight-rank quick gate

The standard three observations and 20 Kineto tests were run at M128. Iter11
lowers the reg168 median by 16.36% for Flash and 16.24% for Pro, but both
production shapes remain slower than accepted Iteration 7. The remaining six
M points were therefore gated rather than mixing a rejected candidate into the
full campaign.

| Model | Iteration 7 (us) | reg168 (us) | Four-warp candidate (us) | Candidate range (us) | Candidate / Iter. 7 | Change vs Iter. 7 | Change vs reg168 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flash M128 | 1,338.252 | 1,628.709 | 1,362.222 | 1,356.516-1,386.851 | 1.018x | +1.79% | -16.36% |
| Pro M128 | 4,601.000 | 5,646.000 | 4,729.000 | 4,702-4,756 | 1.028x | +2.78% | -16.24% |

### NCU and NSYS attribution

Detailed NCU again uses one rank, Flash M128, and 32 local experts. The
instruction values below compare the same `smsp__inst_executed.sum` metric
exported from each report.

| NCU metric | Iteration 7 L1 | Four-warp L1 | Change | Iteration 7 L2 | Four-warp L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| Duration | 911.104 us | 862.016 us | -5.39% | 473.216 us | 487.616 us | +3.04% |
| Executed instructions | 128,956,122 | 118,239,609 | -8.31% | 69,289,384 | 64,300,370 | -7.20% |
| Local-memory spill requests | 0 | 0 | 0 | 0 | 0 | 0 |
| Achieved occupancy | 9.45% | 12.60% | +3.16 pp | 15.27% | 18.07% | +2.80 pp |
| Excessive global sectors | 3,692,178 | 3,692,183 | ~0% | 1,835,136 | 1,835,136 | 0% |
| Excessive shared wavefronts | 71,218 | 71,218 | 0% | 336,384 | 336,384 | 0% |

The four-warp decoder removes the reg168 serialized-decode penalty: relative
to that two-warp variant, NCU duration improves 22.75% for L1 and 13.97% for
L2. It also executes 7-8% fewer instructions than Iteration 7. The remaining
stall distribution is mixed, however. L1 improves overall despite a larger
decoded-ready long-scoreboard share, while L2 loses enough TMA/decode runahead
to move in the wrong direction.

| PC-sampling share | Iteration 7 L1 | Four-warp L1 | Iteration 7 L2 | Four-warp L2 |
|---|---:|---:|---:|---:|
| Long scoreboard | 27.42% | 44.04% | 16.55% | 26.32% |
| Barrier | 21.06% | 12.62% | 55.30% | 44.60% |
| Wait | 16.18% | 12.07% | 8.53% | 6.76% |

| NSYS selected hot path | Iteration 7 | Four-warp frontend | Change |
|---|---:|---:|---:|
| L1 kernel | 825,700 ns | 797,219 ns | -3.45% |
| L1-to-L2 gap | 126,881 ns | 132,992 ns | +4.82% |
| L2 kernel | 428,737 ns | 451,874 ns | +5.40% |
| L1 + gap + L2 | 1,381,318 ns | 1,382,085 ns | +0.06% |

The single-rank hot path is therefore effectively flat: its L1 gain is fully
cancelled by the L2 and inter-kernel tail. At eight ranks, max-rank latency is
1.79-2.78% worse, so the candidate fails the acceptance gate and is rejected.
The experiment establishes that four-wide decode is necessary within the
current reg168 stage pipeline, but making the TMA producers decode every stage
prevents useful runahead. The next iteration returns to the accepted Iteration
7 scheduling boundary and targets a structural reduction in scheduler tiles
and A-TMA traffic rather than adding another decoded-ready dependency.

## Rejected experiment: L2-only BN256 split-N

### Hypothesis and implementation

Iteration 12 returns processed decoding to the math warpgroup and changes only
L2 from BM64/BN128/BK128 with one math warpgroup to BM64/BN256/BK128 with two
math warpgroups. L1 retains accepted Iteration 7's BN128 schedule. Each L2 work
item now reuses one A stage across two independent M64N128 WGMMA consumers and
halves the number of scheduler N tiles and A TMA loads.

The compact L2 frontend uses two dispatch warps, two TMA warps, and eight math
warps, preserving the 384-thread CTA size. Its dispatch and TMA warps share a
warpgroup and execute one warpgroup-uniform register-deallocation instruction.
The two math warpgroups own disjoint N ranges `[0, 128)` and `[128, 256)` in
packed B, expanded B, scale rows, WGMMA descriptors, and the L2 epilogue.
Generic shared stores from the decoder are published to WGMMA's async proxy by
an all-warpgroup barrier, proxy fence, and second barrier.

The host selector enables L2 BN256 only when `hidden % 256 == 0`; shapes that
are merely N128-aligned preserve the BN128 public fallback. The generated
production JIT instances confirm L1 BN128/one math warpgroup/five stages and
L2 BN256/two math warpgroups/three stages. The BN256 stage contains twice the
expanded and packed B data, so the pipeline depth falls from five to three.

The CPU mapping oracle covers both BN128/one-WG and BN256/two-WG layouts. For
BN256 it proves exact-once ownership of 4,096 packed words, 32,768 expanded
bytes, 256 scale rows, and both complete swizzled physical byte ranges.

### Correctness gate

| Scenario | Scale representation | `calc_diff` | Tolerance | Result |
|---|---|---:|---:|---|
| BN128/BN256 mapping and physical-byte oracle | processed triple | exact | exact | PASS |
| L1 smoke, one rank | processed triple | 0.0006 | 0.01 | PASS |
| L1 forced requant, one rank | processed triple | 0.0005 | 0.01 | PASS |
| L1 smoke, one rank | raw pair fallback | 0.0006 | 0.01 | PASS |
| Flash M128 L3, one rank | processed triple | 0.0006 | 0.01 | PASS |
| Pro M128 L3, one rank | processed triple | 0.0006 | 0.01 | PASS |
| Flash M128 L3, eight ranks | processed triple | 0.0006 | 0.01 | PASS |
| Pro M128 L3, eight ranks | processed triple | 0.0006 | 0.01 | PASS |

### Eight-rank quick gate

The standard three observations and 20 Kineto tests were run at M128. Both
production shapes regress by about 30%, and rank-zero phase timing localizes
the loss to L2. The remaining M points were therefore gated.

| Model | Iteration 7 (us) | L2 BN256 candidate (us) | Candidate range (us) | Candidate / Iter. 7 | Change |
|---|---:|---:|---:|---:|---:|
| Flash M128 | 1,338.252 | 1,727.848 | 1,721.129-1,785.658 | 1.291x | +29.11% |
| Pro M128 | 4,601.000 | 6,065.000 | 6,049-6,091 | 1.318x | +31.82% |

Flash rank-zero L1 is 888.388-962.580 us while L2 is 816.556-820.983 us.
Pro rank-zero L1 is 3,068-3,108 us while L2 is 2,977-2,983 us. The BN128 L1
stays close to its accepted behavior; the new BN256 L2 dominates the regression.

### NCU and NSYS attribution

Detailed NCU uses the standard one-rank Flash M128/E32 boundary. The L1
schedule is unchanged except for the formally required async-proxy fence; its
duration remains within 1.8% of Iteration 7 and has no local spill. L2 executes
fewer scheduler work items but compiles at the 384-thread launch-bound ceiling
of 168 registers per thread and spills heavily.

| NCU metric | Iteration 7 L1 | Candidate L1 | Change | Iteration 7 L2 | Candidate L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| Duration | 911.104 us | 927.104 us | +1.76% | 473.216 us | 867.200 us | +83.26% |
| Executed instructions | 128,956,122 | 129,332,753 | +0.29% | 69,289,384 | 101,091,768 | +45.90% |
| Local-memory spill requests | 0 | 0 | 0 | 0 | 21,939,536 | new |
| Launch registers per thread | 168 | 168 | 0 | 168 | 168 | 0 |
| Achieved occupancy | 9.45% | 9.45% | ~0 pp | 15.27% | 18.36% | +3.09 pp |
| Excessive global sectors | 3,692,178 | 3,692,174 | ~0% | 1,835,136 | 1,835,136 | 0% |
| Excessive shared wavefronts | 71,218 | 71,218 | 0% | 336,384 | 336,384 | 0% |

SourceCounters contains no L1 local-memory instruction but finds 435 distinct
L2 `LDL`/`STL` SASS instructions whose aggregate executions exactly equal the
21,939,536 spill requests. The traffic and the resulting 45.9% instruction
increase explain why more active warps do not improve duration. Global-sector
and shared-conflict counts are unchanged, ruling out the original scale-load
sector issue and the decoder swizzle as the new regression source.

| PC-sampling share | Candidate L1 | Candidate L2 |
|---|---:|---:|
| Long scoreboard | 27.95% | 25.09% |
| Barrier | 21.09% | 28.29% |
| Wait | 16.02% | 8.81% |

| NSYS selected hot path | Iteration 7 | L2 BN256 candidate | Change |
|---|---:|---:|---:|
| L1 kernel | 825,700 ns | 857,444 ns | +3.84% |
| L1-to-L2 gap | 126,881 ns | 125,664 ns | -0.96% |
| L2 kernel | 428,737 ns | 801,027 ns | +86.83% |
| L1 + gap + L2 | 1,381,318 ns | 1,784,135 ns | +29.16% |

The L2-only BN256 candidate is rejected in its FP32 persistent-accumulator
form. It validates the split-N ownership and compact frontend. The L2-only
spill, its 435 local-memory instructions, and the longer-lived two-WG state are
consistent with FP32 persistent-accumulator register pressure at the
168-register compile ceiling; SourceCounters does not attribute every local
instruction to a C++ source variable. The next controlled sub-experiment keeps
the same BN256 schedule and stores the persistent scaled accumulation in packed
BF16, halving its register footprint while converting back to FP32 only for
the existing BF16 output epilogue.

## Rejected experiment: packed-BF16 persistent accumulation

### Hypothesis and implementation

Iteration 12b keeps the L2 BN256/two-math-WG schedule unchanged and modifies
only the processed MXFP4 L2 promotion. WGMMA still produces FP32 fragments,
but each K32/K64 promotion uses packed `__hfma2` to retain the cross-K-block
sum as 32 `nv_bfloat162` values instead of 64 FP32 values. The candidate
converts that packed sum back to FP32 after the K loop so the existing L2
epilogue remains unchanged. L1, raw scales, the decoder, TMA, scheduler,
barriers, descriptors, and scatter are identical to Iteration 12a.

The intended register-lifetime reduction did not materialize in generated
code. The compiler retained enough of the packed accumulator and the later
FP32 epilogue array simultaneously that local-memory traffic increased rather
than disappeared.

### Correctness and eight-rank quick gate

Fresh-JIT processed correctness passed for Flash M128 on one rank and for both
Flash and Pro M128 on eight ranks. Every case reported `calc_diff=0.0006`
against the existing `0.01` tolerance, so the packed-BF16 promotion did not
change the observed BF16-output error boundary.

| Model | Iteration 7 (us) | Iteration 12a FP32 (us) | Packed-BF16 (us) | Candidate range (us) | Change vs Iter. 12a |
|---|---:|---:|---:|---:|---:|
| Flash M128 | 1,338.252 | 1,727.848 | 1,944.157 | 1,905.511-1,947.803 | +12.52% |
| Pro M128 | 4,601.000 | 6,065.000 | 6,844.000 | 6,841-6,983 | +12.84% |

Flash rank-zero L1 is 864.572-927.803 us while L2 is 1,020-1,022 us. Pro
rank-zero L1 is 3,088-3,152 us while L2 is 3,748-3,752 us. The unchanged L1
stays near Iteration 12a; the packed-BF16 L2 is the regression.

### NCU and NSYS attribution

Detailed NCU again uses one-rank Flash M128/E32 and a fresh candidate cache.
The candidate still compiles at 168 registers per thread. L2 spill requests
increase by 23.40%, while executed instructions remain nearly flat; the BF16
conversion and packed FMA sequence therefore add pressure and latency without
removing the original local-memory path.

| NCU metric | Iteration 12a L1 | Packed-BF16 L1 | Change | Iteration 12a L2 | Packed-BF16 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| Duration | 927.104 us | 934.688 us | +0.82% | 867.200 us | 1,067.232 us | +23.07% |
| Executed instructions | 129,332,753 | 129,334,049 | ~0% | 101,091,768 | 100,887,046 | -0.20% |
| Local-memory spill requests | 0 | 0 | 0 | 21,939,536 | 27,073,344 | +23.40% |
| Launch registers per thread | 168 | 168 | 0 | 168 | 168 | 0 |
| Achieved occupancy | 9.45% | 9.45% | ~0 pp | 18.36% | 18.35% | ~0 pp |

NSYS independently records the same phase-local regression. The one-shot
profile contains a large host-side interval between L1 and L2, so only kernel
durations are compared here.

| NSYS kernel | Iteration 12a | Packed-BF16 | Change |
|---|---:|---:|---:|
| L1 | 857,444 ns | 843,235 ns | -1.66% |
| L2 | 801,027 ns | 994,467 ns | +24.15% |

Iteration 12b is rejected. Packed BF16 is numerically acceptable but does not
reduce generated-code spill when it is converted into a separate FP32 array
before the epilogue. The next controlled variant must let the L2 epilogue
consume the packed accumulator directly, eliminating the overlapping FP32
array rather than relying on compiler lifetime reuse.

## Rejected experiment: direct packed-BF16 L2 epilogue

Iteration 12c keeps Iteration 12b's packed-BF16 promotion but removes the
post-loop BF16-to-FP32 array conversion. The regular L2 epilogue instead stores
each `nv_bfloat162` pair directly into the CTA BF16 scratch tile before the
existing NVLink scatter. This makes `final_accum` dead throughout the
processed L2 specialization and tests whether the original 1,064-byte stack
frame came only from overlapping packed and unpacked epilogue lifetimes.

Fresh-JIT one-rank Flash and eight-rank Flash/Pro correctness all pass with
`calc_diff=0.0006`. The generated one-rank L2 cubin nevertheless remains at
168 registers per thread with a 1,064-byte stack frame, identical to Iteration
12b. Directly consuming the packed array therefore does not remove the local
frame.

| Model | Iteration 12a FP32 (us) | Iteration 12b (us) | Direct epilogue (us) | Candidate range (us) | Change vs Iter. 12b |
|---|---:|---:|---:|---:|---:|
| Flash M128 | 1,727.848 | 1,944.157 | 1,963.323 | 1,922.132-1,978.560 | +0.99% |
| Pro M128 | 6,065.000 | 6,844.000 | 6,842.000 | 6,820-6,848 | -0.03% |

The standard NCU boundary shows only noise-level reductions versus Iteration
12b. More than 27 million spill requests remain, so the candidate stays about
13% slower than the original FP32 BN256 experiment.

| NCU metric | Iteration 12b L1 | Direct L1 | Change | Iteration 12b L2 | Direct L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| Duration | 934.688 us | 928.320 us | -0.68% | 1,067.232 us | 1,064.256 us | -0.28% |
| Executed instructions | 129,334,049 | 129,329,332 | ~0% | 100,887,046 | 100,335,536 | -0.55% |
| Local-memory spill requests | 0 | 0 | 0 | 27,073,344 | 27,065,152 | -0.03% |
| Launch registers per thread | 168 | 168 | 0 | 168 | 168 | 0 |
| Achieved occupancy | 9.45% | 9.45% | ~0 pp | 18.35% | 18.34% | ~0 pp |

NSYS records L1 at 854,787 ns and L2 at 990,852 ns. The L2 kernel is only
0.36% shorter than Iteration 12b's 994,467 ns and remains 23.7% slower than
Iteration 12a's 801,027 ns.

Iteration 12c is rejected. The result falsifies the narrower epilogue-overlap
hypothesis: packed-BF16 accumulation itself and its surrounding promotion
state still exceed the 168-register ceiling. The next direction should stop
carrying the persistent accumulator in thread-local arrays, for example by
staging partial scaled sums in the existing CTA scratch between K-block
chunks, or return to the accepted BN128 schedule and target scheduler/decoder
work without adding a second live math warpgroup.

## Accepted experiment: Humming-style PRMT/LOP3 MXFP4 decode

### Hypothesis and implementation

Iteration 13 returns both phases to the accepted Iteration 7 BN128 schedule and
changes only the eight-value E2M1-to-E4M3 decoder. The former CUDA intrinsic
sequence decoded the even and odd nibbles separately and then interleaved them.
The replacement adapts Humming's integer lookup to DeepGEMM's
consecutive-nibble byte layout: two `mad.lo.u32` instructions construct the
offset-dependent E4M3 lookup table, `prmt.b32` gathers sign and magnitude bytes,
and `lop3.b32` merges them. PTX-local temporaries avoid extending the already
tight C++ register lifetime. A `fence.proxy.async.shared::cta` publishes the
resulting generic shared stores to WGMMA's async proxy before the existing
128-thread math-warpgroup barrier.

A direct copy of Humming's selector was not correct for DeepGEMM's byte layout:
two early fresh-JIT checks reported `calc_diff=0.9681`. Rebuilding the sign
gather with selectors `0x5140` and `0x7362` restored logical K order. The final
helper also preserves E2M1 negative zero, matching Humming's requantizer.

### Correctness, review, and generated-code contract

Fresh-JIT processed Flash M128 passed on one rank, processed Flash and Pro M128
passed on eight ranks, and the independent raw L1 smoke passed on one rank.
Every kernel result reported `calc_diff=0.0006` against the existing `0.01`
tolerance. The host oracle exhausts all 16 E2M1 codes, all eight nibble
positions, and exponent offsets 1 through 12, including the raw offset-6
negative-zero case.

Independent review found no decoder correctness blocker. A non-portable
early-output GNU inline-assembly constraint was replaced with PTX-local output
registers and ordinary `=r` outputs. Review also found that a named barrier
alone does not establish generic-to-async proxy ordering. The final candidate
therefore adds the explicit proxy fence and reruns correctness, the formal
matrix, NCU, and NSYS. Its L1 and L2 SASS each contain the expected
`FENCE.VIEW.ASYNC.S`; both cubins use 168 registers per thread with zero stack
frame and zero local memory.

### Formal H20 performance

The full eight-rank benchmark uses the PR383 contract: 50 observations for
small-M shapes, three for large-M shapes, 20 internal tests per observation,
and the median of the maximum rank. All 44 FP8/MXFP4 model-shape cases
completed. Iteration 13 is about 22% faster than Iteration 7 at both representative
M128 shapes, but it remains materially behind FP8.

| Model, M128 | FP8 (us) | Iteration 7 MXFP4 (us) | Iteration 13 MXFP4 (us) | Change vs Iter. 7 | MXFP4 / FP8 |
|---|---:|---:|---:|---:|---:|
| Flash | 430.344 | 1,338.252 | 1,041.838 | -22.15% | 2.42x |
| Pro | 1,222.618 | 4,601.000 | 3,601.000 | -21.73% | 2.95x |

The complete formal MXFP4 series is shown below; the paired FP8 series and raw
per-rank observations are retained in the benchmark artifact.

| M | Flash MXFP4 (us) | Pro MXFP4 (us) |
|---:|---:|---:|
| 8 | 886.188 | 2,262.402 |
| 16 | 948.550 | 3,248.500 |
| 32 | 1,016.747 | 3,536.500 |
| 64 | 1,037.341 | 3,575.000 |
| 128 | 1,041.838 | 3,601.000 |
| 256 | 1,038.029 | 3,641.000 |
| 512 | 1,945.773 | 5,469.000 |
| 1,024 | 3,392.000 | 9,062.000 |
| 2,048 | 6,089.000 | 15,970.000 |
| 4,096 | 11,602.000 | 29,439.000 |
| 8,192 | 22,521.000 | 57,240.000 |

### NCU and NSYS attribution

Detailed one-rank Flash M128/E32 profiling attributes the gain directly to the
decoder. Compared with Iteration 7, executed instructions fall by 38.6% in L1
and 36.0% in L2, without introducing spills. NCU duration falls by 21.5% and
22.5% respectively.

| NCU metric | Iteration 7 L1 | Iteration 13 L1 | Change | Iteration 7 L2 | Iteration 13 L2 | Change |
|---|---:|---:|---:|---:|---:|---:|
| Duration | 911.100 us | 714.980 us | -21.53% | 473.220 us | 366.880 us | -22.47% |
| Executed instructions | 128,956,122 | 79,133,688 | -38.64% | 69,289,384 | 44,354,612 | -35.99% |
| Local-memory spill requests | 0 | 0 | 0 | 0 | 0 | 0 |
| Launch registers per thread | 168 | 168 | 0 | 168 | 168 | 0 |
| Achieved occupancy | 9.49% | 9.48% | -0.01 pp | 15.26% | 15.27% | +0.01 pp |

NSYS independently records L1 at 656,739 ns, the inter-kernel gap at 124,864
ns, and L2 at 337,666 ns, for 1,119,269 ns across the selected hot path. The
corresponding Iteration 7 kernels were 825,700 ns and 428,737 ns.

The required proxy fence costs 1.90% on formal Flash M128 and 1.42% on formal
Pro M128 relative to the otherwise identical pre-fence source. In the isolated
NSYS hot path the total increases by only 0.46%. Those pre-fence measurements
are retained as attribution evidence, but they are not used as the accepted
headline result.

Iteration 13 is accepted as the new MXFP4 baseline because the gain reproduces
in the formal matrix, NCU instruction/duration metrics, and NSYS kernel timing,
with both raw and processed format paths covered. It is not yet the target
state: M128 is still 2.42-2.95x slower than the matching FP8 MegaMoE. The next
controlled directions are to reduce the number of staged expanded-weight tiles
so two CTAs can reside on an H20 SM, and to evaluate a register-source WGMMA
path that avoids materializing expanded E4M3 weights in shared memory.

## Accepted experiment: two resident MXFP4 worker CTAs per H20 SM

### Hypothesis and implementation

Iteration 14 implements the first Iteration 13 follow-up. The MXFP4 schedule
keeps three staged A, packed-B, and SFA tiles, but replaces the three expanded
FP8 B stages with one fixed 16 KiB CTA scratch tile. The single math warpgroup
fully consumes that tile before the next packed weight tile overwrites it. This
reduces dynamic shared memory to 93,312 bytes for Flash and 100,480 bytes for
Pro, allowing two 256-thread CTAs to reside on one H20 SM.

The heuristic now launches `2 * physical_sms` logical persistent workers. That
logical worker count is used consistently by the launch grid, persistent
scheduler, dispatch/combine strides, and grid/NVLink arrival counters. MXFP4
uses `__launch_bounds__(256, 2)`, a half-SM 32,768-register CTA budget, maximum
shared-memory carveout, and an exact compiled-kernel occupancy query as a hard
launch gate. Both CUDA Driver and opt-in CUDA Runtime API paths require at
least two active blocks per SM. PDL is disabled for this residency-sensitive
path so an overlapping predecessor cannot consume resources before a grid
barrier.

The compact frontend uses 64 dispatch threads and 64 TMA threads. Together
they execute one warpgroup-collective `setmaxnreg.dec 48`; the 128-thread math
warpgroup executes one `setmaxnreg.inc 208`. Static assertions pin all shared
memory tile sizes, 128-byte region alignment, barrier alignment, and the exact
register budget. The generated L1/L2 PTX contains `.maxntid 256`,
`.minnctapersm 2`, one register deallocation and one allocation instruction.
The final cubins report 128 registers per thread, 48/64-byte L1/L2 stack frames,
and zero separately declared local memory.

### Correctness and backend coverage

Fresh-JIT processed Flash M128 passed on one rank, processed Flash and Pro M128
passed on eight ranks, and the independent raw L1 smoke passed on one rank.
All four checks reported `calc_diff=0.0006` against the existing `0.01`
tolerance. The default CUDA Driver API produced the full correctness and
performance campaign. A separate force rebuild with
`DG_JIT_USE_RUNTIME_API=1` also passed fresh-JIT processed Flash M128 with
`calc_diff=0.0006`; the pod was then rebuilt back to the default Driver API.

### Formal H20 performance

The full eight-rank PR383 contract completed all 22 MXFP4 cases: 50
observations for M at most 128, three for larger M, 20 internal tests per
observation, and the median of the maximum rank. Every point improves on
Iteration 13. The geometric-mean speedup over all 11 shapes is 1.271x for
Flash and 1.249x for Pro.

| Model, M128 | FP8 (us) | Iteration 13 MXFP4 (us) | Iteration 14 MXFP4 (us) | Change vs Iter. 13 | MXFP4 / FP8 |
|---|---:|---:|---:|---:|---:|
| Flash | 430.344 | 1,041.838 | 817.269 | -21.56% | 1.90x |
| Pro | 1,222.618 | 3,601.000 | 2,890.500 | -19.73% | 2.36x |

| M | Flash Iter. 14 (us) | Change vs Iter. 13 | Pro Iter. 14 (us) | Change vs Iter. 13 |
|---:|---:|---:|---:|---:|
| 8 | 710.410 | -19.84% | 1,861.390 | -17.73% |
| 16 | 781.130 | -17.65% | 2,614.731 | -19.51% |
| 32 | 810.788 | -20.26% | 2,862.996 | -19.04% |
| 64 | 811.207 | -21.80% | 2,876.267 | -19.55% |
| 128 | 817.269 | -21.56% | 2,890.500 | -19.73% |
| 256 | 829.220 | -20.12% | 2,912.000 | -20.02% |
| 512 | 1,535.472 | -21.09% | 4,343.000 | -20.59% |
| 1,024 | 2,622.244 | -22.69% | 7,204.000 | -20.50% |
| 2,048 | 4,720.000 | -22.48% | 12,647.000 | -20.81% |
| 4,096 | 8,893.000 | -23.35% | 23,265.000 | -20.97% |
| 8,192 | 17,180.000 | -23.72% | 45,486.000 | -20.54% |

Iteration 14 narrows the full-matrix geometric-mean FP8 gap to 1.967x for
Flash and 2.127x for Pro. At M128 the gap is 1.90x and 2.36x, so this is a
material accepted step but not FP8 parity.

### NCU and NSYS attribution

Detailed one-rank Flash M128/E32 NCU confirms that two-CTA residency improves
latency and achieved occupancy despite compiler spill traffic. L1/L2 duration
falls by 22.79%/23.72% versus Iteration 13. The launch uses 128 registers per
thread and reaches 18.04%/23.87% achieved occupancy, up by 8.56/8.60
percentage points. The tradeoff is 2,687,344/2,177,584 local-memory spill
requests and short 48/64-byte stack frames. The net result remains decisively
positive in the profiler and every formal shape.

| NCU metric | Iteration 13 L1 | Iteration 14 L1 | Iteration 13 L2 | Iteration 14 L2 |
|---|---:|---:|---:|---:|
| Duration | 714.980 us | 552.060 us | 366.880 us | 279.840 us |
| Executed instructions | 79,133,688 | 81,400,657 | 44,354,612 | 47,451,611 |
| Local-memory spill requests | 0 | 2,687,344 | 0 | 2,177,584 |
| Launch registers per thread | 168 | 128 | 168 | 128 |
| Achieved occupancy | 9.48% | 18.04% | 15.27% | 23.87% |
| Achieved active warps/SM | 6.07 | 11.55 | 9.77 | 15.28 |
| L1/TEX hit rate | 83.28% | 16.74% | 79.33% | 55.43% |
| L2 hit rate | 58.79% | 75.05% | 57.28% | 76.20% |

NSYS independently records L1 at 507,908 ns, the inter-kernel gap at 132,864
ns, and L2 at 257,570 ns, for 898,342 ns across the selected hot path. The
total is 19.74% below Iteration 13; the L1 and L2 kernels are 22.66% and 23.72%
shorter, while the gap increases by 6.41%.

Iteration 14 is accepted as the new MXFP4 baseline because its gain reproduces
across every formal point, NCU, and NSYS, while raw/processed formats,
one/eight-rank execution, and both host launch APIs retain correctness. The
next optimization should reduce the new spill traffic without sacrificing
two-CTA residency, or remove the expanded shared-memory tile with a
register-source WGMMA path. MXFP4 remains 1.90-2.36x behind FP8 at M128.
