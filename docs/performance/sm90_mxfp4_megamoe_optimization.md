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
