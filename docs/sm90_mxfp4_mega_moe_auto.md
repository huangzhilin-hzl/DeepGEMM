# SM90 MXFP4 MegaMoE optimization log

This log tracks the profiler-guided optimization work on
`molou/support_sm90_humming_mxfp4afp8_megamoe_auto`.

## Measurement contract

- GPU: 8 x NVIDIA H20 (SM90), one process per GPU.
- Authoritative benchmark: `tests/bench_mega_moe_sm90.py`.
- Models: DSV4 `flash` and `pro`.
- Batch sizes: `8 16 32 64 128 256 512 1024 2048 4096 8192`.
- Cache policy: cold L2 (`--flush-l2`, the benchmark default).
- Score: `max_rank_median_us`; lower is better.
- Small-M observations: 50; large-M observations: 3; 20 launches per
  observation.
- Baseline candidate: `5c060dc51427b497384d1b42f2b01263813c2d87`.
- Reference: PR383 at `0220ac51939dcaa2b4c9aa6aa60c5a9bf8785aa4`.

Profiler timings are deliberately not substituted for the benchmark score.
NCU replay and multi-process NSYS tracing perturb the cooperative workload;
they are used for topology, resource, and bottleneck attribution only.

## Iteration 00: baseline and profiler plumbing

### Reason

The original profiler scripts did not delimit the single production launch,
and NCU replay of all eight distributed processes deadlocked at collective
barriers. Add an opt-in CUDA profiler range to `--profile-only`, then run one
NCU instance per rank so every participant advances through the same replay
pass.

### Direction

- `DG_NCU_RANGE=1` wraps only the target launch, its device synchronize, and
  the rank barrier with `cudaProfilerStart/Stop`.
- The environment variable defaults to zero, so normal correctness and
  performance runs are unchanged.
- NCU uses eight independent application-replay profilers and PM Sampling.
- NSYS uses a low-perturbation mode: only rank 0 is traced while ranks 1-7 run
  normally.

### Baseline performance

`delta` is `(candidate / PR383 - 1) * 100`; positive values are regressions.

| model | M | PR383 us | candidate us | delta |
| --- | ---: | ---: | ---: | ---: |
| flash | 8 | 301.924 | 449.780 | +48.97% |
| flash | 16 | 312.858 | 494.348 | +58.01% |
| flash | 32 | 328.370 | 503.078 | +53.20% |
| flash | 64 | 361.647 | 487.814 | +34.89% |
| flash | 128 | 433.330 | 489.759 | +13.02% |
| flash | 256 | 518.971 | 533.920 | +2.88% |
| flash | 512 | 917.665 | 907.125 | -1.15% |
| flash | 1024 | 1526.078 | 1562.000 | +2.35% |
| flash | 2048 | 2745.844 | 2826.000 | +2.92% |
| flash | 4096 | 5079.000 | 5311.000 | +4.57% |
| flash | 8192 | 9808.000 | 10293.000 | +4.94% |
| pro | 8 | 693.125 | 1179.500 | +70.17% |
| pro | 16 | 971.170 | 1538.500 | +58.42% |
| pro | 32 | 1064.547 | 1598.000 | +50.11% |
| pro | 64 | 1106.107 | 1634.000 | +47.73% |
| pro | 128 | 1228.166 | 1637.000 | +33.29% |
| pro | 256 | 1636.918 | 1663.000 | +1.59% |
| pro | 512 | 2415.108 | 2582.000 | +6.91% |
| pro | 1024 | 4060.000 | 4015.000 | -1.11% |
| pro | 2048 | 7025.000 | 7130.000 | +1.49% |
| pro | 4096 | 12987.000 | 13352.000 | +2.81% |
| pro | 8192 | 25203.000 | 25761.000 | +2.21% |

Aggregate regression versus PR383:

- all 22 cases: +20.36% geometric mean;
- M <= 128: +45.93% geometric mean;
- M >= 256: +2.51% geometric mean.

### NCU and NSYS findings

| implementation | launch topology | registers/thread | dynamic SMEM/CTA |
| --- | --- | ---: | ---: |
| candidate Flash M8 | one persistent kernel, 156 CTAs | 128 | 111.84 KiB |
| candidate Pro M8 | one persistent kernel, 156 CTAs | 128 | 102.62 KiB |
| PR383 Flash M8 | L1 + L2, 78 CTAs each | 168 | 231.70 KiB |
| PR383 Pro M8 | L1 + L2, 78 CTAs each | 168 | 213.76 KiB |

The candidate explicitly consumes two resident CTAs per H20 SM. PR383 runs
one high-resource CTA per SM for each phase. PM Sampling also reports only 25%
maximum active warps for the candidate and 18.75% for PR383; neither latency
case is occupancy-throughput limited. Absolute replay timings and sampled
utilization vary widely across ranks, so the stable evidence is the launch and
resource topology rather than the replay duration.

Low-perturbation NSYS confirms one fused candidate launch versus two sequential
PR383 launches. Even this mode inflates timings and reverses the uninstrumented
ranking, so it is retained as a dependency/timeline view only.

### Next experiments

1. For M <= 128, test a 78-worker cooperative grid while preserving the
   156-worker path for M >= 256. This directly tests whether the second
   resident CTA amplifies dispatch, grid-barrier, and low-work contention.
2. If 78 workers help, compile a true one-CTA/SM latency variant with a relaxed
   launch bound and a larger per-CTA register budget to eliminate spills.
3. Specialize fixed dispatch/combine barriers only after the worker-count
   experiment quantifies their contribution.
4. Reject any change that regresses either model at M >= 256 beyond run-to-run
   noise.

### Result

No kernel behavior changed in iteration 00. The benchmark values above remain
the branch baseline.

## Rejected experiment R01: one worker CTA per SM

### Reason and direction

NCU showed a 156-CTA candidate grid versus 78 CTAs per PR383 phase. To test
whether the second resident CTA was mostly fixed dispatch/barrier overhead,
M <= 128 was changed temporarily from `2 * physical_sms` workers to
`physical_sms` workers. The kernel remained at 128 registers/thread and kept
the same pipeline; M >= 256 was untouched.

### Performance

| model | M | baseline us | experiment us | change |
| --- | ---: | ---: | ---: | ---: |
| flash | 8 | 449.780 | 656.990 | +46.07% |
| flash | 16 | 494.348 | 723.421 | +46.34% |

Both points used the full 50-observation benchmark contract. The experiment
was stopped after the second completed point because the regression was far
outside measurement noise; the source change was reverted and not committed.

### Conclusion

The second resident CTA provides necessary compute parallelism in the current
128-register kernel. A useful one-CTA path must also reclaim the per-SM
register/shared-memory budget and change the pipeline, rather than only
shrinking the grid. The next iteration therefore targets latency-path register
pressure or fixed synchronization inside the existing two-CTA grid.

## Iteration 02: blockwise Flash swap-AB for M <= 32

### Reason

NCU ruled out local-memory traffic as the baseline bottleneck: Pro M8 reported
zero local loads/stores, while the fused kernel spent 18.73% of sampled cycles
on long-scoreboard stalls versus 4.86-6.08% in PR383's two phase kernels. The
baseline also issues an M64 x N128 WGMMA for experts that commonly own fewer
than eight tokens. Repository history showed that an earlier per-tensor-scale
kernel avoided this waste by swapping the WGMMA operands at small M, but the
path was removed when routed activations changed to per-block scaling.

### Direction

- Keep the 156-CTA, two-resident-CTA launch topology proven necessary by R01.
- Treat each N64 weight half as WGMMA M and bucket routed tokens into WGMMA N8,
  N16, N32, or N64.
- Preserve the current blockwise numerical contract: promote each L1 K128 and
  L2 K64 group with its staged activation scale, and retain cross-promotion
  partials as packed BF16.
- Remap the swapped L1 result through SwiGLU and a cross-warp dynamic-scale
  reduction; remap L2 back into the existing BF16 scatter layout.
- Use epilogue-exclusive C/D shared memory for the scale reduction. Reusing a
  released pipeline stage caused the producer to overwrite the reduction
  scratch and was rejected during correctness development.
- Enable the specialization only for routed-only Flash with M <= 32. The
  crossover sweep below shows that M64 is neutral-to-slower and M128 is a clear
  regression.

### Crossover sweep

This sweep temporarily enabled swap-AB through M128. It uses the same 50
observations, cold-L2 policy, and `max_rank_median_us` score as the baseline.

| M | baseline us | swap-AB us | change |
| ---: | ---: | ---: | ---: |
| 8 | 449.780 | 414.278 | -7.89% |
| 16 | 494.348 | 450.907 | -8.79% |
| 32 | 503.078 | 463.545 | -7.86% |
| 64 | 487.814 | 493.518 | +1.17% |
| 128 | 489.759 | 683.926 | +39.64% |

### Final performance

After fixing the selector at M <= 32, all five points were rerun from the final
source. M64 and M128 compile the regular baseline path.

| M | baseline us | final us | change | PR383 us | gap to PR383 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 449.780 | 413.731 | -8.01% | 301.924 | +37.03% |
| 16 | 494.348 | 445.744 | -9.83% | 312.858 | +42.47% |
| 32 | 503.078 | 456.946 | -9.17% | 328.370 | +39.16% |
| 64 | 487.814 | 483.059 | -0.97% | 361.647 | +33.57% |
| 128 | 489.759 | 508.761 | +3.88% | 433.330 | +17.41% |

The geometric-mean gain over the branch baseline is 9.01% for the three
enabled points and 4.97% over M8-M128. The remaining M8-M32 geometric-mean gap
to PR383 falls from 53.30% to 39.54%. The unchanged M128 path varied by +3.88%
in this run; it has no generated swap-AB code and is retained as run-to-run
noise rather than widening the selector.

### Correctness and resource result

- Eight-rank `production.flash_m32` with forced ring wrap passes at
  `diff=0.000656` (`0.01` tolerance).
- The specialized kernel uses 128 registers and four barriers. PTXAS reports a
  448-byte stack frame, 518 bytes of spill stores, and 604 bytes of spill
  loads, plus serialized WGMMA due to the fixed 128-register launch bound.
- The resource report identifies the next target: shorten the live range of
  the swap epilogue/partial arrays enough to restore WGMMA issue parallelism,
  without giving up two resident CTAs.

## Iteration 03: extend blockwise swap-AB to Pro M <= 64

### Reason and direction

Iteration 02 left the largest small-M gaps in DSV4 Pro: +70.17% at M8 and
+47.73% at M64. Pro uses the same routed MXFP4 numerical contract and fixed
M64 x N128 Humming tile, so the validated blockwise swap-AB path was extended
to H7168/I3072. Pro aliases the second decoded-weight buffer with C/D shared
memory to preserve two-CTA occupancy; the alias is safe because the C/D
epilogue begins only after the routed mainloop has finished.

A temporary selector enabled the Pro path through M256 to locate the actual
crossover before choosing a production cutoff.

| M | baseline us | crossover sweep us | change |
| ---: | ---: | ---: | ---: |
| 8 | 1179.500 | 1060.000 | -10.13% |
| 16 | 1538.500 | 1350.000 | -12.25% |
| 32 | 1598.000 | 1395.500 | -12.67% |
| 64 | 1634.000 | 1461.000 | -10.59% |
| 128 | 1637.000 | 1773.500 | +8.34% |
| 256 | 1663.000 | 3957.000 | +137.94% |

The sweep used ten observations per point only to select the boundary. It
rejects M128/M256 and fixes the final Pro selector at M <= 64. The accepted
points were then rerun with the full 50-observation contract.

### Final performance

| M | baseline us | final us | change | PR383 us | gap to PR383 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1179.500 | 1042.000 | -11.66% | 693.125 | +50.33% |
| 16 | 1538.500 | 1328.000 | -13.68% | 971.170 | +36.74% |
| 32 | 1598.000 | 1390.500 | -12.98% | 1064.547 | +30.62% |
| 64 | 1634.000 | 1448.500 | -11.35% | 1106.107 | +30.95% |
| 128 | 1637.000 | 1646.500 | +0.58% | 1228.166 | +34.06% |

The four enabled points improve by 12.42% geometric mean. Their geometric-
mean gap to PR383 falls from 56.36% to 36.94%. Across Pro M8-M128, including
the unchanged M128 path, the gain is 9.97% geometric mean.

### Correctness and resource result

- Eight-rank, forced-ring-wrap validation passes at `diff=0.005395` for
  `production.pro_m32` and `diff=0.003913` for `production.pro_m64`.
- PTXAS reports 128 registers, four barriers, a 488-byte stack frame, 584
  bytes of spill stores, and 732 bytes of spill loads for Pro M32, with the
  same fixed-register WGMMA serialization warning as Flash.
- The next optimization should split the shared blockwise math from the
  model-specific remap so the N8/N16 specializations do not carry arrays sized
  for N64. That is the most direct route to reduce the reported local frame and
  long-scoreboard pressure while preserving the proven launch topology.

## Iteration 04: post-iteration profiler and final matrix

### Matched large-M validation

The optimized selectors are false above Flash M32 and Pro M64, but a first
three-observation rerun showed enough node variance to make Flash M256 appear
8.84% slower than the earlier branch baseline. A ten-observation recheck gave
526.617 us versus the earlier 533.920 us, confirming that the source-identical
path had not regressed. PR383 was then run immediately after the candidate on
the same H20 node for a matched large-M comparison.

| model | M | final us | matched PR383 us | gap |
| --- | ---: | ---: | ---: | ---: |
| flash | 256 | 526.617 | 520.417 | +1.19% |
| flash | 512 | 951.115 | 965.191 | -1.46% |
| flash | 1024 | 1609.000 | 1563.810 | +2.89% |
| flash | 2048 | 2855.000 | 2743.821 | +4.05% |
| flash | 4096 | 5392.000 | 5099.000 | +5.75% |
| flash | 8192 | 10345.000 | 9899.000 | +4.51% |
| pro | 256 | 1727.000 | 1678.652 | +2.88% |
| pro | 512 | 2585.000 | 2455.305 | +5.28% |
| pro | 1024 | 4031.000 | 4067.000 | -0.89% |
| pro | 2048 | 7108.000 | 7084.000 | +0.34% |
| pro | 4096 | 13332.000 | 13010.000 | +2.48% |
| pro | 8192 | 25884.000 | 24955.000 | +3.72% |

The matched M >= 256 geometric-mean gap is +2.54%, effectively unchanged from
the original +2.51%. Combining it with the 50-observation small-M results gives
a +16.19% geometric-mean gap over all 22 Flash/Pro points, down from +20.36%.
The M <= 128 gap falls more substantially, from +45.93% to +34.99%.

### Post-iteration NCU

The table reports medians across eight application-replay reports for Pro M8.
Rank-local utilization is highly skewed by distributed replay, so these values
are diagnostic trends rather than throughput scores.

| counter | baseline | final swap-AB |
| --- | ---: | ---: |
| local-load sectors | 0 | 675904 |
| local-store sectors | 0 | 562560 |
| issue active | 2.71% | 4.51% |
| tensor-pipe active | 0.085% | 0.130% |
| barrier stall | 62.45% | 62.37% |
| long-scoreboard stall | 18.73% | 75.38% |

Swap-AB increases useful issue/tensor activity and reduces unproductive padded
WGMMA work enough to win despite introducing a large local-memory frame. The
unchanged barrier median says grid synchronization is no longer the first
optimization target. The new local traffic and fourfold long-scoreboard median
agree with PTXAS's 488-byte Pro stack frame and are the clearest remaining
bottleneck.

### Post-iteration NSYS

Low-perturbation rank-0 traces retain the same one-kernel, 156-CTA persistent
topology. Their single-launch durations move from 755.548 to 744.348 us for
Flash M8 (-1.48%) and from 1603.704 to 1399.545 us for Pro M8 (-12.73%). NSYS
still perturbs the collective launch enough that these durations are not used
as the benchmark score; the trace is evidence that no extra phase or launch
was added.

## Rejected experiment R04: stage swap-AB SwiGLU in shared memory

### Reason and direction

Iteration 04 attributed the remaining small-M bottleneck to the swap-AB local
frame. R04 tested whether the epilogue's 32 FP32 post-SwiGLU values and 16
inverse scales per thread were responsible. The experiment staged a row-major
BF16 post-SwiGLU tile in epilogue-exclusive C/D shared memory, retained only
the current eight-token chunk in registers, and compressed the staging tile to
FP8 in place. This added one warpgroup barrier per token chunk while preserving
the 156-CTA, two-resident-CTA topology.

### Resource and correctness result

All forced-ring-wrap cases passed the 0.01 tolerance: Flash M32 at 0.000661,
Pro M32 at 0.002372, and Pro M64 at 0.002741. Resource usage barely changed:

| model | stack before/after | spill stores before/after | spill loads before/after |
| --- | ---: | ---: | ---: |
| Flash | 448 / 440 B | 518 / 514 B | 604 / 596 B |
| Pro | 488 / 480 B | 584 / 572 B | 732 / 720 B |

The persistent WGMMA serialization warning remained. The small resource delta
shows that these epilogue arrays were not the dominant source of the local
frame; the accumulator lifetime across the routed mainloop remains the primary
suspect.

### Screening performance

The experiment used ten cold-L2 observations and 20 launches per observation.
`change` compares it with the accepted 50-observation Iteration 02/03 result.

| model | M | accepted us | R04 us | change |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 413.731 | 453.184 | +9.54% |
| Flash | 16 | 445.744 | 491.462 | +10.26% |
| Flash | 32 | 456.946 | 489.124 | +7.04% |
| Pro | 8 | 1042.000 | 1069.500 | +2.64% |
| Pro | 16 | 1328.000 | 1404.000 | +5.72% |
| Pro | 32 | 1390.500 | 1405.000 | +1.04% |
| Pro | 64 | 1448.500 | 1486.000 | +2.59% |

The extra synchronization costs substantially more than the 8-12 byte spill
reduction saves, so the kernel change was reverted. The full remote evidence is
under `/app/deepgemm-auto-results/iter05-shared-swiglu` on the H20 pod.

## Rejected experiment R05: one-CTA high-register swap-AB

### Reason and direction

R05 tested whether the swap specialization could exchange the proven second
resident CTA for enough registers to eliminate all local traffic. Its launch
bound was relaxed from two blocks to one, and the cooperative worker grid was
reduced from 156 to 78 CTAs to match exact-kernel residency. Regular large-M
kernels retained the original two-block bound and 156-CTA grid.

### Resource and correctness result

Pro M32 forced-ring-wrap correctness passed at 0.000709. PTXAS moved from 128
to 231 registers and eliminated the entire 488-byte stack frame and all
584/732-byte spill stores/loads. The WGMMA serialization warning nevertheless
remained, and exact occupancy allowed only one CTA per H20 SM.

### Screening performance

The ten-observation screen was stopped after two Flash points because the
regression was decisive.

| M | R01 128-reg/1-CTA us | accepted 128-reg/2-CTA us | R05 231-reg/1-CTA us | change vs accepted |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 656.990 | 413.731 | 693.239 | +67.56% |
| 16 | 723.421 | 445.744 | 754.531 | +69.27% |

Despite zero spill traffic, R05 is also 5.52%/4.30% slower than R01. The
current fused pipeline needs the second CTA's compute parallelism more than it
benefits from a larger register file. The source change was reverted; evidence
is under `/app/deepgemm-auto-results/iter06-swap-launchbound` on the H20 pod.

## Rejected experiment R06: consume packed BF16 in the epilogue

### Reason and direction

R06 preserved the 156-CTA, two-resident-CTA topology and added no barriers. It
kept the routed result in the mainloop's 32 packed BF16x2 values and consumed
them directly from both swap-AB epilogues, avoiding the intermediate expansion
to 64 FP32 values per thread.

Pro M32 correctness passed at 0.004380. PTXAS moved only from 488 to 480 bytes
of stack, 584 to 574 bytes of spill stores, and 732 to 720 bytes of spill
loads; WGMMA remained serialized.

The ten-observation cold-L2 screen was mixed and regressed Flash M32
decisively:

| model | M | accepted us | R06 us | change |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 413.731 | 410.351 | -0.82% |
| Flash | 16 | 445.744 | 465.213 | +4.37% |
| Flash | 32 | 456.946 | 504.060 | +10.31% |
| Pro | 8 | 1042.000 | 1072.000 | +2.88% |
| Pro | 16 | 1328.000 | 1322.500 | -0.41% |
| Pro | 32 | 1390.500 | 1377.500 | -0.93% |
| Pro | 64 | 1448.500 | 1465.500 | +1.17% |

The isolated sub-1% wins are within screening noise and do not justify the
Flash regression, so the source change was reverted. Evidence is under
`/app/deepgemm-auto-results/iter07-bf16-direct` on the H20 pod.

## Rejected experiment R07: bucket-local mainloop frames

### Reason and direction

R07 retained all four runtime swap buckets and the N64 skew-routing fallback,
but allocated each branch's WGMMA accumulator and packed BF16 persistent state
at its actual N8/N16/N32/N64 size. It initialized and expanded only live bucket
values, while preserving the 156-CTA, two-resident-CTA topology.

Pro M32 correctness passed at 0.000704. PTXAS removed the WGMMA serialization
warning and reduced the stack from 488 to 472 bytes, but spill stores/loads
rose from 584/732 to 752/896 bytes. A same-node, immediately following matched
control confirmed that the added spill traffic outweighed the issue benefit:

| model | M | matched control us | R07 us | change |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 426.198 | 434.758 | +2.01% |
| Flash | 16 | 478.143 | 495.790 | +3.69% |
| Flash | 32 | 491.390 | 497.128 | +1.17% |
| Pro | 8 | 1041.000 | 1117.000 | +7.30% |
| Pro | 16 | 1340.500 | 1435.500 | +7.09% |
| Pro | 32 | 1400.500 | 1474.500 | +5.28% |
| Pro | 64 | 1466.000 | 1559.000 | +6.34% |

Both sides used ten cold-L2 observations and 20 launches per observation. The
source change was reverted; candidate and matched-control logs are under
`/app/deepgemm-auto-results/iter08-bucket-frames` on the H20 pod.

## Accepted experiment R08: target packed BF16 to Pro M16/M32

### Reason and direction

R06 had a weak but repeatable positive signal at Pro M16/M32 and regressions
at Pro M8/M64 and Flash. R08 turns that observation into a JIT compile-time
selector instead of applying the epilogue rewrite globally. The selector is
true only for routed DSV4 Pro (`hidden=7168`) at exactly M16 or M32. Flash,
shared experts, Pro M8/M64, and every regular-M kernel retain their original
compile-time path.

The selected kernel keeps the MXFP4 result as 32 packed BF16x2 values through
the swap-AB epilogue. It avoids expanding those values into a second 64-FP32
array, then unpacks only the gate/up pair currently consumed by the L1 SwiGLU
or L2 store. The 156-CTA, two-resident-CTA topology and four barriers are
unchanged.

### Correctness and resources

The final synced source passed the eight-rank forced-ring-wrap
`production.pro_m32` test with `calc_diff=0.006622`, below the 0.01 contract.
PTXAS for the authoritative
M8192-capacity benchmark instance reports:

| resource | matched control | R08 | change |
| --- | ---: | ---: | ---: |
| stack frame | 488 B | 480 B | -8 B |
| spill stores | 584 B | 578 B | -6 B |
| spill loads | 732 B | 724 B | -8 B |
| registers/thread | 128 | 128 | unchanged |
| barriers | 4 | 4 | unchanged |

The fixed-register WGMMA serialization warning remains. This is a deliberately
small resource reduction, but unlike R04 it adds no synchronization, and
unlike R05 it preserves the second resident CTA.

### Matched screening and formal A/B

The ten-observation screen was positive at both selected points: -1.75% at
M16 and -1.23% at M32. The result was then repeated with the full contract:
50 cold-L2 observations and 20 launches per observation on both sides. The
control differs only by forcing the new template selector to false.

| model | M | matched control us | R08 us | change |
| --- | ---: | ---: | ---: | ---: |
| Pro | 16 | 1333.500 | 1312.000 | -1.61% |
| Pro | 32 | 1390.500 | 1369.000 | -1.55% |

The two independent run lengths agree in sign and magnitude, so R08 is kept.
The full candidate/control logs are under
`/app/deepgemm-auto-results/iter09-targeted-bf16`.

### Final DSV4 Flash/Pro matrix

This is the required `tests/bench_mega_moe_sm90.py` matrix after R08. Small-M
points use 50 observations; large-M points use three. `gap` compares with the
original PR383 baseline from Iteration 00. Only Pro M16/M32 are source-changed
by R08; movement at every other point is measurement variance and is not
credited to this optimization.

| model | M | PR383 us | R08 final us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 301.924 | 423.155 | +40.15% |
| Flash | 16 | 312.858 | 453.093 | +44.82% |
| Flash | 32 | 328.370 | 453.737 | +38.18% |
| Flash | 64 | 361.647 | 484.287 | +33.91% |
| Flash | 128 | 433.330 | 481.499 | +11.12% |
| Flash | 256 | 518.971 | 514.339 | -0.89% |
| Flash | 512 | 917.665 | 905.505 | -1.33% |
| Flash | 1024 | 1526.078 | 1552.000 | +1.70% |
| Flash | 2048 | 2745.844 | 2844.000 | +3.57% |
| Flash | 4096 | 5079.000 | 5307.000 | +4.49% |
| Flash | 8192 | 9808.000 | 10330.000 | +5.32% |
| Pro | 8 | 693.125 | 1038.500 | +49.83% |
| Pro | 16 | 971.170 | 1289.000 | +32.73% |
| Pro | 32 | 1064.547 | 1375.500 | +29.21% |
| Pro | 64 | 1106.107 | 1442.500 | +30.41% |
| Pro | 128 | 1228.166 | 1657.000 | +34.92% |
| Pro | 256 | 1636.918 | 1647.000 | +0.62% |
| Pro | 512 | 2415.108 | 2572.000 | +6.50% |
| Pro | 1024 | 4060.000 | 3999.000 | -1.50% |
| Pro | 2048 | 7025.000 | 7110.000 | +1.21% |
| Pro | 4096 | 12987.000 | 13358.000 | +2.86% |
| Pro | 8192 | 25203.000 | 25841.000 | +2.53% |

Against the original PR383 matrix, the geometric-mean gaps are now +15.56%
over all 22 points, +34.14% for M <= 128, and +2.06% for M >= 256. The prior
accepted report was +16.19%, +34.99%, and +2.54%, respectively. The matched
R08 A/B above is the attribution result; the aggregate movement also contains
unchanged-path node variance.

### R08 NCU comparison with PR383 at Pro M32

Eight independent application-replay reports were collected for each
implementation. The following launch resources are invariant across ranks:

| implementation/phase | launches | grid | threads/CTA | registers/thread | dynamic SMEM/CTA | barriers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R08 fused | 1 | 156 | 256 | 128 | 101600 B | 4 |
| PR383 L1 | 1 | 78 | 384 | 168 | 212736 B | 3 |
| PR383 L2 | 1 | 78 | 384 | 168 | 212736 B | 16 |

The counter table reports medians across the eight rank-local reports. Replay
is strongly rank-skewed, so these values are diagnostic and are not benchmark
scores.

| NCU counter | R08 fused | PR383 L1 | PR383 L2 |
| --- | ---: | ---: | ---: |
| local-load sectors | 449920 | 0 | 0 |
| local-store sectors | 901248 | 0 | 0 |
| issue active | 3.315% | 1.545% | 1.030% |
| tensor-pipe active | 0.050% | 0.025% | 0.065% |
| barrier stall | 62.410% | 73.380% | 85.675% |
| long-scoreboard stall | 14.480% | 24.680% | 16.615% |

R08 improves the fused kernel without changing its architecture, but the
comparison exposes the remaining structural difference: PR383 crosses a hard
L1/L2 kernel boundary and has no local sectors in either phase, while the
fused implementation still materializes almost 1.35 million local sectors per
rank. The fused path has higher sampled issue activity and fewer barriers, yet
remains 29.21% behind PR383 at the authoritative M32 point. Eliminating the
cross-phase frame is therefore more important than another small epilogue
micro-optimization.

### R08 NSYS topology

Low-perturbation rank-0 traces confirm one fused MegaMoE launch for R08 versus
separate L1 and L2 launches for PR383. The trace also contains one NCCL
all-reduce and one sub-microsecond fill kernel on both sides. R08's traced main
kernel is 1.660 ms; PR383's traced L1/L2 kernels are 0.909/1.326 ms. These
perturbed durations contradict the production benchmark ordering, so they are
used only as launch-topology evidence, consistent with the measurement
contract.

Complete profiler reports remain on the H20 pod under
`/app/deepgemm-auto-results/iter09-targeted-bf16/profiles`. A compact export of
the benchmark logs, rank-0 NCU text, and NSYS reports is also available locally
at
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter09-targeted-bf16`.

### Recommended next iterations

1. Prototype a true two-phase latency path for M <= 64, with separate L1 and
   L2 kernels so the L1 accumulator and packed-BF16 frame dies at a hard kernel
   boundary. R05 proved that merely reducing the fused grid to one CTA per SM
   is not enough; the lifetime split, not only the register allowance, is the
   feature to copy from PR383. Keep the current fused kernel for M >= 128.
2. Preserve R08's exact selector until a matched sweep proves a wider range.
   Pro M8/M64 and every Flash point regressed under the global packed-BF16
   rewrite, so they must retain the existing epilogue.
3. Do not key a maximum N8/N16/N32 bucket only from global M: one expert can
   receive skewed routes aggregated from all ranks, so its runtime `valid_m`
   may require the N64 fallback even when per-rank M is small.
4. Use NCU source counters on the two-phase prototype to require zero local
   sectors before a long benchmark. A resource experiment that removes spills
   but loses the second fused CTA, as R05 did, should be rejected before the
   full matrix.
5. Preserve the measured crossover guards: Flash M <= 32 and Pro M <= 64.
   Every future change should rerun M64/M128 boundaries plus the full DSV4
   Flash/Pro matrix so a latency win cannot leak into the throughput path.

## Rejected experiment R09: restore the historical split-kernel snapshot

### Reason and direction

PR383 wins the latency points with separate L1 and L2 kernels, while the
current branch uses one 156-CTA persistent kernel. Before transplanting that
architecture into the current code, commit `985cbba` was built in an isolated
worktree as a low-cost control. That snapshot already launches distinct
`sm90_fp8_mega_moe_l1_impl` and `sm90_fp8_mega_moe_l2_impl` kernels and overlaps
MXFP4 L2 weight decode. The benchmark was patched only to expose the current
explicit `--flush-l2` switch; kernel source was left unchanged.

The comparison is cold-L2 on the same H20 pod. PR383's original benchmark did
not spell out the flag, but its `bench_kineto` call also defaults to
`flush_l2=True`, so the reference matrix uses the same cache policy.

### Screening result

| model | M | R08 us | historical split us | change |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 423.155 | 459.975 | +8.70% |
| Flash | 16 | 453.093 | 502.088 | +10.81% |
| Flash | 32 | 453.737 | 532.980 | +17.47% |
| Flash | 64 | 484.287 | 523.777 | +8.15% |
| Flash | 128 | 481.499 | 535.222 | +11.16% |
| Pro | 8 | 1038.500 | 1218.391 | +17.32% |
| Pro | 16 | 1289.000 | 1677.314 | +30.12% |
| Pro | 32 | 1375.500 | 1834.994 | +33.40% |
| Pro | 64 | 1442.500 | 1879.606 | +30.30% |
| Pro | 128 | 1657.000 | 1914.422 | +15.54% |

The old snapshot is decisively slower, so restoring it wholesale is rejected.
Its hard L1/L2 lifetime boundary remains useful as a mechanism reference, but
it lacks the subsequent dispatch, decode, swap-AB, and epilogue work on this
branch. A future split path must therefore be derived from current source.

## Accepted experiment R10: reuse the Pro swap-AB fragment by weight half

### Reason

R08's Pro M32 kernel kept both N64 weight-half WGMMA fragments live at once.
PTXAS consequently materialized a 480-byte local frame with 578-byte spill
stores and 724-byte spill loads. NCU measured 449,920 local-load sectors and
901,248 local-store sectors per median rank. The two halves are independent:
after a completed half is promoted into the persistent packed-BF16 partial,
its FP32 fragment does not need to survive the other half.

### Direction

- Add a compile-time `kReuseSwapABFragment` selector for routed DSV4 Pro
  (`hidden=7168`) on the existing M <= 64 swap-AB path.
- Allocate one 32-float half fragment instead of the complete 64-float pair.
- Template WGMMA issue and promotion over a weight-half range. Pro issues,
  waits for, and promotes half 0 before reusing the fragment for half 1.
- Preserve the packed-BF16 state, numerical scaling, N8/N16/N32/N64 runtime
  buckets, 156-CTA cooperative grid, two resident CTAs per H20 SM, and four
  barriers.
- Compile Flash back to the original simultaneous-half path. A preliminary
  global experiment regressed changed Flash points while improving Pro, so the
  final selector is deliberately model-specific.

The L2 phase has two K32 promotion groups. It processes both halves for group
0, then both halves for group 1; the A/SFA producer stage is released only
after the final promotion, preserving the existing stage lifetime contract.

### Correctness and generated resources

Eight-rank forced-ring-wrap validation passes at `diff=0.000656` for Flash M32
and `diff=0.000714` for Pro M32. The Flash smoke compile reproduces its exact
R08 resource report, proving that the selector does not leak into that model.

| Pro M32 resource | R08 | R10 | change |
| --- | ---: | ---: | ---: |
| stack frame | 480 B | 8 B | -472 B |
| spill stores | 578 B | 4 B | -574 B |
| spill loads | 724 B | 4 B | -720 B |
| registers/thread | 128 | 128 | unchanged |
| barriers | 4 | 4 | unchanged |
| WGMMA serialization warning | yes | no | removed |

Flash remains at a 448-byte frame, 518-byte spill stores, 604-byte spill loads,
128 registers, and four barriers.

### Matched 50-observation A/B

The control is fixed commit `31efd2d`; candidate and control use isolated JIT
caches and the full cold-L2 benchmark contract. `change` is the directly
attributable result, independent of the older PR383 run.

| model | M | matched control us | R10 us | change |
| --- | ---: | ---: | ---: | ---: |
| Pro | 8 | 1050.000 | 932.734 | -11.17% |
| Pro | 16 | 1311.500 | 1204.000 | -8.20% |
| Pro | 32 | 1371.000 | 1258.500 | -8.21% |
| Pro | 64 | 1456.000 | 1305.000 | -10.37% |

The four-point geometric-mean improvement is 9.49%. Every point agrees with
the preliminary screen, so the optimization is accepted.

### Final DSV4 Flash/Pro matrix

This is a single authoritative run from the final selector: 50 observations
for M <= 128, three for M >= 256, 20 launches per observation, and explicit
`--flush-l2 1`. `gap` compares against the cold-L2 PR383 matrix.

| model | M | PR383 us | R10 final us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 301.924 | 408.218 | +35.21% |
| Flash | 16 | 312.858 | 464.321 | +48.41% |
| Flash | 32 | 328.370 | 448.784 | +36.67% |
| Flash | 64 | 361.647 | 489.366 | +35.32% |
| Flash | 128 | 433.330 | 498.466 | +15.03% |
| Flash | 256 | 518.971 | 505.672 | -2.56% |
| Flash | 512 | 917.665 | 941.469 | +2.59% |
| Flash | 1024 | 1526.078 | 1593.000 | +4.39% |
| Flash | 2048 | 2745.844 | 2832.000 | +3.14% |
| Flash | 4096 | 5079.000 | 5337.000 | +5.08% |
| Flash | 8192 | 9808.000 | 10304.000 | +5.06% |
| Pro | 8 | 693.125 | 909.827 | +31.26% |
| Pro | 16 | 971.170 | 1192.000 | +22.74% |
| Pro | 32 | 1064.547 | 1245.000 | +16.95% |
| Pro | 64 | 1106.107 | 1304.000 | +17.89% |
| Pro | 128 | 1228.166 | 1641.500 | +33.65% |
| Pro | 256 | 1636.918 | 1647.000 | +0.62% |
| Pro | 512 | 2415.108 | 2583.000 | +6.95% |
| Pro | 1024 | 4060.000 | 3996.000 | -1.58% |
| Pro | 2048 | 7025.000 | 7093.000 | +0.97% |
| Pro | 4096 | 12987.000 | 13326.000 | +2.61% |
| Pro | 8192 | 25203.000 | 25858.000 | +2.60% |

The geometric-mean gaps versus PR383 are +13.73% over all 22 points, +28.91%
for M <= 128, and +2.45% for M >= 256. R08 reported +15.56%, +34.14%, and
+2.06%, respectively. The unchanged large-M movement is run variance; the
matched A/B above is the optimization attribution. Pro M8-M64 alone is now
+22.08% behind PR383 by geometric mean, versus +35.30% in R08's final matrix.

### R10 NCU and NSYS

Eight rank-local application-replay reports were collected for Pro M32. The
counter values below are medians; replay remains strongly rank-skewed and is
not a latency score.

| NCU counter | R08 fused | R10 fused | PR383 L1 | PR383 L2 |
| --- | ---: | ---: | ---: | ---: |
| local-load sectors | 449920 | 36480 | 0 | 0 |
| local-store sectors | 901248 | 2496 | 0 | 0 |
| issue active | 3.315% | 2.914% | 1.545% | 1.030% |
| tensor-pipe active | 0.050% | 0.020% | 0.025% | 0.065% |
| barrier stall | 62.410% | 64.002% | 73.380% | 85.675% |
| long-scoreboard stall | 14.480% | 30.069% | 24.680% | 16.615% |

R10 removes 91.89% of local-load sectors and 99.72% of local-store sectors.
The residual traffic agrees with PTXAS's tiny 4-byte spill accesses rather
than the old cross-half frame. The lower sampled tensor activity and higher
long-scoreboard median expose the new tradeoff: eliminating spills wins the
production benchmark, but `wait<0>` between the two weight halves leaves the
tensor pipeline under-filled.

The low-perturbation NSYS trace still contains exactly one 156-CTA MegaMoE
kernel, one NCCL all-reduce, and one fill kernel. Its traced main-kernel time is
1.558 ms versus 1.660 ms for R08. PR383 remains a two-kernel L1/L2 topology;
as before, trace durations are topology evidence rather than benchmark scores.

Complete evidence remains on the pod under
`/app/deepgemm-auto-results/iter12-pro-sequential-selector`. The compact local
export is under
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter12-pro-sequential-selector`.

### Next iteration

1. Keep the same 32-float Pro fragment allocation, but issue both weight halves
   in one batch for N8/N16/N32. Two bucket-sized fragments require at most 32
   floats there, so this can restore overlap without restoring spills.
2. Retain sequential reuse only for the skew-routing N64 fallback, whose two
   fragments would require 64 floats. The runtime bucket remains authoritative;
   global M cannot safely remove the fallback.
3. Require PTXAS to stay near the 8/4/4-byte resource result and rerun matched
   Pro M8/M16/M32/M64. Reject the hybrid if it restores either local traffic or
   the WGMMA serialization warning.
4. Keep Flash and M >= 128 unchanged. If the hybrid does not close most of the
   remaining 16.95-31.26% small-M gaps, prototype a current-source L1/L2 split
   instead of restoring the obsolete R09 snapshot.

## Rejected experiment R11: issue both Pro weight halves in one commit group

### Reason and direction

R10's spill removal introduced a `wait<0>` between the two weight halves. The
first overlap experiment allocated two compact fragments inside the existing
32-float Pro storage and issued both halves before one wait. N64 retained the
sequential fallback because two full fragments need 64 floats.

### Screening result

Correctness passed for Pro M32 at `diff=0.007567`, and PTXAS still reported
128 registers, an 8-byte stack frame, and no local-memory allocation. However,
the same-commit batch was consistently slower than R10:

| model | M | R10 screen us | same-group us | change |
| --- | ---: | ---: | ---: | ---: |
| Pro | 8 | 932.734 | 1025.500 | +9.95% |
| Pro | 16 | 1204.000 | 1308.500 | +8.68% |
| Pro | 32 | 1258.500 | 1391.500 | +10.57% |
| Pro | 64 | 1305.000 | 1439.000 | +10.27% |

The regression is therefore WGMMA scheduling rather than restored spills.
R11 is rejected and fully reverted.

## Accepted experiment R12: pipeline separate Pro weight-half commit groups

### Reason

R11 showed that sharing one commit group serializes badly, while R10's
per-half `wait<0>` leaves no WGMMA in flight during the first promotion. A
middle ground is possible: give each compact fragment its own commit group,
then use `wait<1>` to expose half 0 while half 1 remains in flight.

### Direction

- Issue the two compact weight halves as separate WGMMA commit groups.
- Promote half 0 after `wait<1>`, then promote half 1 after the final
  `wait<0>`; apply the same schedule to both L2 K32 promotion groups.
- Reuse the existing 32-float allocation and preserve the 156-CTA cooperative
  grid, four barriers, stage-release ordering, and numerical scaling.
- Select the pipeline only for the measured Pro M8 and M64 specializations.
  Packed-BF16 M16/M32 buckets keep R10's sequential schedule, as do Flash,
  N64 fallback work, and M >= 128.

Correctness passes for Pro M64 at `diff=0.003540`. PTXAS remains at 128
registers, an 8-byte stack frame, and zero allocated local bytes.

### Matched 50-observation A/B

Control is fixed commit `3066396`; both sides use isolated JIT caches and the
full cold-L2 benchmark contract.

| model | M | R10 control us | R12 us | change |
| --- | ---: | ---: | ---: | ---: |
| Pro | 8 | 943.654 | 936.180 | -0.79% |
| Pro | 64 | 1318.500 | 1300.000 | -1.40% |

The two-point geometric-mean improvement is 1.10%. Both selected points agree
with the 10-observation screen, so the targeted schedule is accepted.

### Final DSV4 Flash/Pro matrix

The final-selector run uses 50 observations for M <= 128, three for M >= 256,
20 launches per observation, max-rank medians, and explicit `--flush-l2 1`.

| model | M | PR383 us | R12 final us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 301.924 | 409.205 | +35.53% |
| Flash | 16 | 312.858 | 447.273 | +42.96% |
| Flash | 32 | 328.370 | 457.566 | +39.34% |
| Flash | 64 | 361.647 | 492.922 | +36.30% |
| Flash | 128 | 433.330 | 496.938 | +14.68% |
| Flash | 256 | 518.971 | 588.401 | +13.38% |
| Flash | 512 | 917.665 | 924.604 | +0.76% |
| Flash | 1024 | 1526.078 | 1593.000 | +4.39% |
| Flash | 2048 | 2745.844 | 2807.000 | +2.23% |
| Flash | 4096 | 5079.000 | 5298.000 | +4.31% |
| Flash | 8192 | 9808.000 | 10346.000 | +5.49% |
| Pro | 8 | 693.125 | 921.860 | +33.00% |
| Pro | 16 | 971.170 | 1202.000 | +23.77% |
| Pro | 32 | 1064.547 | 1242.000 | +16.67% |
| Pro | 64 | 1106.107 | 1296.000 | +17.17% |
| Pro | 128 | 1228.166 | 1635.500 | +33.17% |
| Pro | 256 | 1636.918 | 1675.000 | +2.33% |
| Pro | 512 | 2415.108 | 2569.000 | +6.37% |
| Pro | 1024 | 4060.000 | 4019.000 | -1.01% |
| Pro | 2048 | 7025.000 | 7095.000 | +1.00% |
| Pro | 4096 | 12987.000 | 13324.000 | +2.59% |
| Pro | 8192 | 25203.000 | 25901.000 | +2.77% |

The geometric-mean gaps versus PR383 are +14.45% over all 22 points, +28.88%
for M <= 128, +3.66% for M >= 256, and +22.48% for Pro M8-M64. The matched
A/B is the change attribution: all Flash points, Pro M16/M32, and M >= 128
compile the previous schedule. In particular, the unchanged Flash M256 point
is unusually slow in this three-observation matrix and accounts for most of
the aggregate movement from R10; it is not caused by R12's Pro-only selector.

### R12 NCU and NSYS

Eight rank-local NCU application-replay reports were collected for Pro M64.
R10's earlier report used Pro M32, so the counter comparison is directional
rather than a latency attribution; the matched benchmark above remains the
acceptance test.

| median NCU counter | R10 Pro M32 | R12 Pro M64 |
| --- | ---: | ---: |
| local-load sectors | 36480 | 36864 |
| local-store sectors | 2496 | 2496 |
| issue active | 2.914% | 2.940% |
| tensor-pipe active | 0.020% | 0.030% |
| barrier stall | 64.002% | 63.353% |
| long-scoreboard stall | 30.069% | 29.770% |

The unchanged local-store count and near-identical local-load count confirm
that the two live compact fragments did not restore R08's spill traffic. The
small tensor-pipe increase and scoreboard decrease agree with the intended
half-1 overlap, while the rank-skewed replay values remain diagnostic only.

Low-perturbation NSYS still contains exactly one 156-CTA fused MegaMoE kernel,
one NCCL all-reduce, and one fill kernel. The traced main kernel is 1.503 ms;
this is topology evidence, not a replacement for the cold-L2 score.

Complete evidence remains on the pod under
`/app/deepgemm-auto-results/iter15-half-group-targeted`. The compact local
export is under
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter15-half-group-targeted`.

### Next iteration

1. Apply the same separate-group schedule to Flash M8/M16/M32 while retaining
   the sequential N64 fallback. Flash still carries R08's 448-byte frame and
   is 35.53-42.96% behind PR383 at those points.
2. Require PTXAS to remove the Flash frame without lowering two-CTA residency,
   then run matched Flash M8/M16/M32 screening before a full matrix.
3. If Flash cannot benefit from compact-fragment pipelining, stop tuning the
   fused accumulator schedule and implement a current-source L1/L2 lifetime
   split for the remaining M <= 128 latency gap.

## Accepted experiment R13: extend fragment reuse to Flash swap-AB

### Reason and direction

Flash M8/M16/M32 still compiled the simultaneous-half accumulator allocation.
The generated kernel retained a 448-byte stack frame with 518-byte spill
stores and 604-byte spill loads even after the Pro path became spill-free.
R12's separate-group schedule provides the missing mechanism: compact runtime
buckets can overlap the two halves without retaining the full N128 fragment,
while N64 can still reuse one N64 fragment sequentially.

Change `kReuseSwapABFragment` from the DSV4 Pro-only selector to every
compile-time small-M swap-AB specialization. No runtime branch, template
argument, grid shape, barrier, SMEM allocation, or host dispatch changes.

### Correctness and generated resources

Eight-rank forced-ring-wrap validation passes for Flash M32 at
`diff=0.000656`. The final Flash cubin retains 128 registers/thread and removes
the entire generated local frame:

| Flash M32 resource | R12 | R13 | change |
| --- | ---: | ---: | ---: |
| stack frame | 448 B | 0 B | -448 B |
| spill stores | 518 B | 0 B | -518 B |
| spill loads | 604 B | 0 B | -604 B |
| registers/thread | 128 | 128 | unchanged |
| stack-backed local frame | yes | no | removed |

### Matched 50-observation A/B

The Flash control is commit `3066396`, which is code-identical to R12 for
Flash. Control and candidate use isolated JIT caches and the full cold-L2
contract.

| model | M | R12 control us | R13 us | change |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 427.321 | 399.766 | -6.45% |
| Flash | 16 | 431.292 | 425.902 | -1.25% |
| Flash | 32 | 449.645 | 449.304 | -0.08% |

The three-point geometric-mean improvement is 2.63%. M32 is effectively
neutral, while M8 and M16 reproduce the screening direction. Because every
point is non-regressing and the common selector eliminates the full frame,
R13 is accepted without another token-count template switch.

### Final DSV4 Flash/Pro matrix

This final-selector run uses the full authoritative contract: 50 observations
for M <= 128, three for M >= 256, 20 launches per observation, max-rank
medians, and explicit `--flush-l2 1`.

| model | M | PR383 us | R13 final us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 301.924 | 427.080 | +41.45% |
| Flash | 16 | 312.858 | 445.934 | +42.54% |
| Flash | 32 | 328.370 | 455.301 | +38.65% |
| Flash | 64 | 361.647 | 500.512 | +38.40% |
| Flash | 128 | 433.330 | 492.109 | +13.56% |
| Flash | 256 | 518.971 | 504.073 | -2.87% |
| Flash | 512 | 917.665 | 916.746 | -0.10% |
| Flash | 1024 | 1526.078 | 1600.000 | +4.84% |
| Flash | 2048 | 2745.844 | 2847.000 | +3.68% |
| Flash | 4096 | 5079.000 | 5332.000 | +4.98% |
| Flash | 8192 | 9808.000 | 10309.000 | +5.11% |
| Pro | 8 | 693.125 | 916.829 | +32.27% |
| Pro | 16 | 971.170 | 1192.000 | +22.74% |
| Pro | 32 | 1064.547 | 1250.500 | +17.47% |
| Pro | 64 | 1106.107 | 1301.500 | +17.66% |
| Pro | 128 | 1228.166 | 1645.500 | +33.98% |
| Pro | 256 | 1636.918 | 1660.000 | +1.41% |
| Pro | 512 | 2415.108 | 2576.000 | +6.66% |
| Pro | 1024 | 4060.000 | 3998.000 | -1.53% |
| Pro | 2048 | 7025.000 | 7071.000 | +0.65% |
| Pro | 4096 | 12987.000 | 13315.000 | +2.53% |
| Pro | 8192 | 25203.000 | 25730.000 | +2.09% |

The geometric-mean gaps versus PR383 are +13.82% over all 22 points, +29.45%
for M <= 128, and +2.25% for M >= 256. The single-sided small-M aggregate is
slightly worse than R12 because Flash M8 moved from 409.205 to 427.080 us in a
different run. Conversely, unchanged Flash M256 moved from the anomalous
588.401 to 504.073 us. These opposite shifts demonstrate why the matched
2.63% Flash A/B, rather than cross-run aggregate movement, is the R13 change
attribution.

### R13 NCU and NSYS

Eight rank-local NCU application-replay reports were collected for Flash M8.
Every rank reports exactly zero local-load and zero local-store sectors,
confirming that the generated-frame removal survives production dispatch and
is not merely a PTXAS allocation annotation.

| median NCU counter | R13 Flash M8 |
| --- | ---: |
| local-load sectors | 0 |
| local-store sectors | 0 |
| issue active | 2.737% |
| tensor-pipe active | 0.010% |
| barrier stall | 53.528% |
| long-scoreboard stall | 25.228% |
| wait stall | 4.433% |

Utilization and stall samples remain strongly rank-skewed under replay and are
not latency scores. Zero local sectors is invariant across all eight reports
and is therefore the stable profiler result.

Low-perturbation NSYS still shows exactly one 156-CTA fused MegaMoE kernel,
one NCCL all-reduce, and one fill kernel. The traced main kernel is 0.735 ms;
the trace is retained only as launch-topology evidence.

Complete evidence remains on the pod under
`/app/deepgemm-auto-results/iter16-flash-fragment-pipeline`. The compact local
export is under
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter16-flash-fragment-pipeline`.

### Next iteration

1. Stop widening accumulator micro-schedules: routed Flash and Pro swap-AB
   paths now have zero or near-zero generated local traffic, but the small-M
   geometric-mean gap remains +29.45% versus PR383.
2. Implement the hard current-source L1/L2 lifetime boundary for M <= 128.
   Preserve the fused kernel for M >= 256 and preserve R12/R13 fragment reuse
   inside whichever phase kernel consumes the routed MXFP4 weights.
3. Start with a Pro M32 prototype and require separate L1/L2 launches, zero
   local sectors per phase, correctness, and a matched win before extending
   the selector to Flash or the full latency range.

## Rejected experiment R14: current-source hard L1/L2 split for Pro M32

### Reason and implementation

R13 left Pro M32 17.47% behind PR383, so this experiment tested whether the
remaining gap was caused mainly by retaining both logical phases in one
compiled kernel. The prototype was derived from the current persistent source,
not the obsolete historical split snapshot:

- two compile-time `Linear1` and `Linear2` entry points;
- a phase-only routed scheduler that never instantiated the other task type;
- L1 retained dispatch metadata and task counters, while L2 performed combine
  and cleanup;
- a host safety proof required the complete worst-case routed pool for the
  invocation to fit in the physical ring before enabling the split;
- the production benchmark summed the two Kineto kernel durations.

The selector was limited to routed-only DSV4 Pro M32. Every other shape kept
the R13 fused kernel.

### Generated resources

Both phase cubins compiled and completed an eight-rank smoke launch while
preserving the two-CTA occupancy contract:

| Pro M32 phase | registers/thread | stack | local | shared |
| --- | ---: | ---: | ---: | ---: |
| Linear1 | 125 | 0 B | 0 B | 1024 B |
| Linear2 | 128 | 0 B | 0 B | 1024 B |

This proves that a hard phase boundary can reduce L1 below the 128-register
launch cap, but the resource reduction alone is not sufficient.

### Cold-L2 screening result

The screen used ten observations, 20 launches per observation, explicit
`--flush-l2 1`, and the maximum rank-local sum of L1 plus L2 time.

| model | M | R13 us | split us | change | PR383 us | split gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pro | 32 | 1250.500 | 1360.462 | +8.79% | 1064.547 | +27.79% |

The split regresses decisively, so it was stopped before a 50-observation
formal run and was not submitted as production code.

### NSYS diagnosis

NSYS confirms the intended two-kernel topology. Profiler-induced cross-rank
skew makes the medians unsuitable as latency scores, but the least-perturbed
rank records 0.824 ms for Linear1 and 0.445 ms for Linear2, or 1.269 ms
combined, consistent with the Kineto result. The trace also contains one NCCL
all-reduce and one fill kernel. The regression therefore comes from removing
the useful L1/L2 overlap and adding a full phase boundary, not from spills or
an accidental extra compute launch.

The prototype source was reverted completely. Evidence remains on the pod at
`/app/deepgemm-auto-results/iter17-current-split` and its isolated cubins at
`/app/deepgemm-auto-results/jit-auto-r14b`. The compact local export is under
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter17-current-split`.

### Next iteration

1. Keep one cooperative launch and the interleaved ring scheduler; the hard
   phase boundary gives up more overlap than its 125-register L1 saves.
2. Use the split cubins only as a compiler oracle: identify state that is live
   solely because both phase epilogues are present, then shorten or alias that
   state inside the fused kernel without changing launch topology.
3. Start with Pro M32/M64 and require a matched cold-L2 win. The first target
   is phase-specific descriptor/epilogue selection around the task lambda,
   followed by scheduler issue/barrier reductions rather than another
   accumulator representation change.

## PR383 phase diagnosis after R14

The matched PR383 implementation uses FP8 weights, two split kernels, one
78-CTA block per H20 SM, two math warpgroups per CTA, N64 per warpgroup for
the small-M swap-AB schedule, and seven producer stages. Its authoritative
Pro M32 rank-0 medians are about 0.675 ms for L1 and 0.382 ms for L2. A new
low-perturbation NSYS capture reproduced 0.674/0.364 ms on the least-perturbed
rank.

R14's current-source MXFP4 phase kernels use two 256-thread CTAs per SM and one
math warpgroup per CTA. Their least-perturbed NSYS times are 0.824/0.445 ms.
The per-phase gaps are therefore already present before the fused scheduler;
the production fused kernel merely recovers roughly 0.110 ms by overlapping
L1 and L2.

Static SASS structure identifies the source of the phase cost:

| phase | implementation | IMAD | LOP3 | registers | local |
| --- | --- | ---: | ---: | ---: | ---: |
| L1 | PR383 FP8 | 345 | 66 | 168 | 0 B |
| L1 | R14 MXFP4 | 1223 | 1129 | 125 | 0 B |
| L2 | PR383 FP8 | 420 | 65 | 168 | 0 B |
| L2 | R14 MXFP4 | 1194 | 996 | 128 | 0 B |

The approximately fourfold integer-address/decode body in MXFP4, rather than
register spill, is now the primary small-M target. PR383 evidence is retained
under `/app/deepgemm-auto-results/iter18-pr383-phase-compare` and the local
export under
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter18-pr383-phase-compare`.

## Rejected experiment R15: replace Pro M32 packed BF16 with half-group pipeline

R08's packed-BF16 epilogue is selected for Pro M16/M32, while R12's accepted
separate-commit-group weight-half pipeline is selected only for non-packed
Pro M8/M64. R15 removed M32 from the packed selector to test the previously
unmeasured combination through the already-compiled R12 path.

The adjacent ten-observation cold-L2 screen used 20 launches per observation
and maximum-rank medians:

| model | M | R13 control us | R15 us | change |
| --- | ---: | ---: | ---: | ---: |
| Pro | 32 | 1282.500 | 1287.000 | +0.35% |

The result is neutral-to-negative and does not justify a 50-observation run.
The one-line selector change was reverted. Logs and cubins remain under
`/app/deepgemm-auto-results/iter19-pro-m32-half-pipeline`, with the compact
local export under
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter19-pro-m32-half-pipeline`.

The next structural prototype keeps one fused launch but moves to one CTA per
SM with two math warpgroups. Each warpgroup owns one N64 weight half, matching
PR383's successful small-M parallel decomposition while retaining packed
MXFP4 storage and the interleaved L1/L2 scheduler.

## Rejected experiment R16: one-CTA dual-math-warpgroup Pro M32

### Reason and direction

R13 processes the two N64 weight halves with one math warpgroup per CTA and
keeps two CTAs resident per H20 SM. PR383 instead uses one CTA per SM and two
math warpgroups, each owning one N64 half. R16 ported that decomposition to the
fused MXFP4 kernel for routed DSV4 Pro M32:

- launch 78 cooperative CTAs instead of 156;
- use two math warpgroups per CTA and decode disjoint N64 packed-weight halves;
- combine the L1 per-token amax across all eight math warps before FP8
  quantization;
- keep one N128 L2 scratch tile and let one warpgroup perform the final remote
  scatter after both warpgroups finish their disjoint stores.

The first 384-thread/three-stage specialization compiled and launched at
`REG=128, STACK=0, LOCAL=0`. It measured 1777 us in the first cold-L2 sample.
A seven-stage version used the extra one-CTA shared-memory budget and improved
the sample to 1684 us. A final 512-thread version restored 128 dispatch threads
per SM (with two spare frontend warps so every `setmaxnreg` warpgroup stayed
collective); it measured 1674 us and retained `REG=128, STACK=0, LOCAL=0`.

### Adjacent performance screen

The final topology and the untouched R13 control were measured adjacently with
20 observations, 20 launches per observation, cold L2, and maximum-rank
medians:

| model | M | R13 control us | R16 us | change | PR383 us | R16 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pro | 32 | 1262.500 | 1665.000 | +31.88% | 1064.547 | +56.40% |

The benchmark's warmup non-finite guard passed, but the dedicated numerical
correctness suite was intentionally not run after the performance screen had
already rejected the topology by a wide margin.

### Conclusion

Two math warpgroups preserve the same two tensor-core warpgroups per physical
SM as R13, but binding them to one CTA also binds them to one scheduler task
and halves the number of independent producer/scheduler streams. Increasing
the producer pipeline from three to seven stages recovered only about 5%, and
doubling dispatch width was neutral. The missing task-level concurrency is
therefore more important than sequential weight-half WGMMA for this fused
kernel. The full prototype was reverted.

R16 rules out a direct PR383 launch-topology transplant. The next iteration
returns to the accepted two-CTA fused schedule and targets the measured MXFP4
decode/address body itself: reduce repeated B128 swizzle/address formation in
`prepare_stage_weights`, then require a static SASS reduction and a matched
cold-L2 improvement before retention.

Artifacts are under
`/app/deepgemm-auto-results/iter20-dual-math-wg`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter20-dual-math-wg`.

## Rejected experiment R17: hoist expanded-B row swizzle

### Reason and direction

R14's phase cubins showed roughly four times as many integer/address
instructions as PR383. In `prepare_stage_weights`, every decoded K32 word
formed `logical_n * BLOCK_K + logical_k` and applied the B128 swizzle inside
the unrolled loop. R17 computed the fixed expanded-row base and its swizzle XOR
once per decoded row, then applied that XOR to each word offset.

The transformation was algebraically valid because `Swizzle<3,4,3>` derives
its XOR mask from row bits 7--9 while the within-row K offset occupies bits
0--6. It did reduce two instruction classes, but ptxas changed address
materialization in the opposite direction:

| Pro M32 static SASS count | R13 | R17 | change |
| --- | ---: | ---: | ---: |
| LOP3 | 1820 | 1661 | -159 |
| SHF | 734 | 552 | -182 |
| IMAD | 1495 | 1675 | +180 |
| IADD3 | 24 | 24 | 0 |

Both cubins used `REG=128, STACK=8, LOCAL=0`.

### Adjacent performance screen

Twenty cold-L2 observations with 20 launches per observation produced:

| model | M | R13 control us | R17 us | change |
| --- | ---: | ---: | ---: | ---: |
| Pro | 32 | 1291.000 | 1294.000 | +0.23% |

The compiler-level trade was neutral-to-negative in the authoritative metric,
so the source change was reverted. This also shows that aggregate static
instruction reduction is insufficient when it lengthens dependency chains or
keeps row bases live across the fully unrolled decoder.

The next decode experiment should reduce work without adding live address
state. A promising direction is to change the packed representation so the
four K32 relative exponents for a row are broadcast/decoded with fewer
per-word extracts, or to fuse address update into inline PTX with immediate
increments rather than relying on C++ swizzle expressions.

Artifacts are under
`/app/deepgemm-auto-results/iter21-row-swizzle-hoist`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter21-row-swizzle-hoist`.

## Iteration R18: PRMT exponent extraction for Pro M32

### Reason and direction

R14 showed that integer decode/address work, rather than local-memory spill,
is the primary remaining Pro M32 cost. The Flash path already had a validated
`PRMT` implementation that extracts four UE8M0 scale bytes from one packed
word. Pro M32 still instantiated the generic shift-and-mask implementation.
R18 enables the existing `PRMT` specialization only for routed DSV4 Pro M32;
all other model/batch combinations keep their previous specialization.

The generated Pro M32 cubin keeps `REG=128, STACK=8, LOCAL=0`. Static SASS
counts confirm a direct integer-instruction substitution with no address or
resource side effect:

| Pro M32 static SASS opcode | R13 control | R18 | change |
| --- | ---: | ---: | ---: |
| LOP3 | 1820 | 1628 | -192 |
| SHF | 734 | 542 | -192 |
| PRMT | 825 | 1081 | +256 |
| IMAD | 1495 | 1495 | 0 |
| IADD3 | 24 | 24 | 0 |

The `-192/-192/+256` shape is consistent with replacing four independent
byte shift/mask extracts per packed exponent word by byte permutations.

### Correctness and authoritative cold-L2 result

The full eight-rank `production.pro_m32` test passes with `fast_math=1` and
`diff=0.000712`, including a physical 32-block ring wrap. The formal matched
run used 50 observations, ten warmups, 20 launches per observation, explicit
`--flush-l2 1`, and maximum-rank medians:

| model | M | R13 control us | R18 us | change | PR383 us | R18 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pro | 32 | 1284.000 | 1258.500 | -1.99% | 1064.547 | +18.22% |

An earlier adjacent 20-observation screen measured 1302.0 versus 1274.5 us
(-2.11%), so the benefit reproduced under both run orders despite node-level
variance. The 50-observation matched result is the acceptance score.

### NCU and NSYS diagnosis

Eight simultaneous application-replay NCU reports were collected for R18.
The table compares cross-rank medians with the prior R10/R13-equivalent Pro
M32 control report. Replay is rank-skewed, so sampled utilization and stalls
are diagnostic trends, not latency scores.

| median NCU counter | control | R18 | change |
| --- | ---: | ---: | ---: |
| executed instructions | 547.671 M | 445.730 M | -18.62% |
| local-load sectors | 36480 | 36480 | 0 |
| local-store sectors | 2496 | 2496 | 0 |
| issue active | 2.915% | 3.000% | +0.085 pp |
| tensor-pipe active | 0.020% | 0.025% | +0.005 pp |
| barrier stall | 64.005% | 46.660% | -17.345 pp |
| long-scoreboard stall | 30.070% | 21.945% | -8.125 pp |
| wait stall | 2.415% | 4.475% | +2.060 pp |

The stable findings are unchanged local traffic/resources and fewer dynamic
instructions. The stall movement agrees directionally with a shorter decode
dependency body, but is not used alone to attribute the 1.99% benchmark win.

Low-perturbation rank-0 NSYS still contains exactly one 156-CTA fused MegaMoE
launch. Its traced main kernel is 1.524 ms; as in prior iterations, this is
topology evidence only and not a replacement for the cold-L2 score.

Complete NCU reports remain on the H20 pod under
`/app/deepgemm-auto-results/iter22-pro-prmt-exponent`. The compact local export,
including raw counter text, NSYS, cubins, SASS, and resource usage, is under
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter22-pro-prmt-exponent`.

### Next iteration

1. Screen the same `PRMT` specialization independently at Pro M8, M16, and
   M64. Accept per shape only when an adjacent cold-L2 A/B win reproduces;
   do not infer benefit from M32 or from static instruction count alone.
2. If the full Pro latency selector is retained, rerun all Pro M <= 128
   points plus the final Flash/Pro matrix so selector boundaries and the
   aggregate PR383 gap are measured on one node epoch.
3. Continue reducing the MXFP4 decoder body after this low-risk extraction
   win; the remaining +18.22% Pro M32 gap is too large to close with launch
   topology or accumulator changes already rejected by R14-R17.

## Iteration R19: extend PRMT extraction to Pro M16 and M64

### Reason and direction

R18's Pro M32 result established that the `PRMT` UE8M0 extractor reduces the
MXFP4 integer body without changing resources. R19 temporarily enabled the
same specialization for every routed DSV4 Pro M <= 64 point, then treated M8,
M16, and M64 as independent acceptance decisions. M32 remained the accepted
R18 path throughout.

Two 20-observation adjacent screens were run in opposite orders. Their
candidate changes for M8/M16/M64 were respectively
`+0.08%/-0.04%/-1.22%` and `-2.50%/-0.33%/-1.79%`. M64 reproduced clearly;
M16 remained weakly positive; M8 changed sign. The final decision therefore
used the required 50-observation run rather than the screen average.

### Formal cold-L2 selection

The formal run used ten warmups, 50 observations, 20 launches per observation,
explicit `--flush-l2 1`, and maximum-rank medians:

| Pro point | generic control us | temporary PRMT us | change | decision | PR383 us | retained gap |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| M8 | 927.100 | 929.900 | +0.30% | reject | 693.125 | generic path |
| M16 | 1215.500 | 1188.000 | -2.26% | accept | 971.170 | +22.33% |
| M64 | 1309.000 | 1284.000 | -1.91% | accept | 1106.107 | +16.08% |

The production selector is explicit: Pro M16, M32, and M64 use `PRMT`; Pro M8
and unmeasured token counts retain the generic extractor. This avoids turning
M8's formal regression into a committed change and avoids extrapolating from
the standard benchmark points.

Both retained cubin specializations remain at `REG=128, STACK=8, LOCAL=0`.
The full eight-rank correctness suite passes `production.pro_m16` with
`diff=0.001685` and `production.pro_m64` with `diff=0.001142`, including the
required physical-ring wrap. Because the repository previously lacked an M16
production correctness case, R19 adds one with the same DSV4 Pro configuration
and ring-wrap contract as M32/M64.

Artifacts are under `/app/deepgemm-auto-results/iter23-pro-prmt-sweep`; the
local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter23-pro-prmt-sweep`.

### Next iteration

Run the final accepted selector across the complete authoritative Flash/Pro
matrix and rerun PR383 on the same H20 node epoch. Use that matrix to update
the aggregate gap, then return to the integer/address body identified by R14:
the retained PRMT changes recover about 2% per affected Pro point, but do not
yet remove the double-digit small-M gap.

## Final matrix after R19

The accepted R19 branch and PR383 were measured consecutively on the same H20
pod. Both used the authoritative DSV4 Flash/Pro shapes, M
`8,16,32,64,128,256,512,1024,2048,4096,8192`, 50 observations for M <= 128,
three observations for M >= 256, 20 launches per observation, cold L2, and
maximum-rank medians. PR383 reports the sum of its L1 and L2 phase durations;
the candidate reports its one fused persistent kernel.

| model | M | PR383 us | R19 us | R19 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 295.816 | 406.674 | +37.48% |
| Flash | 16 | 307.067 | 436.333 | +42.10% |
| Flash | 32 | 327.385 | 431.733 | +31.87% |
| Flash | 64 | 366.641 | 480.363 | +31.02% |
| Flash | 128 | 430.369 | 489.968 | +13.85% |
| Flash | 256 | 498.118 | 538.485 | +8.10% |
| Flash | 512 | 918.462 | 927.685 | +1.00% |
| Flash | 1024 | 1572.457 | 1559.000 | -0.86% |
| Flash | 2048 | 2729.000 | 2847.000 | +4.32% |
| Flash | 4096 | 5094.000 | 5300.000 | +4.04% |
| Flash | 8192 | 9840.000 | 10290.000 | +4.57% |
| Pro | 8 | 708.989 | 915.649 | +29.15% |
| Pro | 16 | 985.679 | 1194.500 | +21.19% |
| Pro | 32 | 1065.082 | 1237.500 | +16.19% |
| Pro | 64 | 1119.668 | 1284.500 | +14.72% |
| Pro | 128 | 1227.411 | 1639.000 | +33.53% |
| Pro | 256 | 1631.865 | 1668.000 | +2.21% |
| Pro | 512 | 2441.122 | 2594.000 | +6.26% |
| Pro | 1024 | 4052.000 | 4003.000 | -1.21% |
| Pro | 2048 | 6990.000 | 7095.000 | +1.50% |
| Pro | 4096 | 13011.000 | 13402.000 | +3.01% |
| Pro | 8192 | 25143.000 | 25751.000 | +2.42% |

The same-node geometric gaps are:

- all 22 points: `+13.14%`;
- M <= 128: `+26.75%`;
- M >= 256: `+2.92%`;
- Flash small-M: `+30.90%`;
- Pro small-M: `+22.74%`.

R13's recorded gaps were `+13.82%`, `+29.45%`, and `+2.25%` for all, small,
and large points. R18/R19 reduce the small-M aggregate by another 2.70
percentage points and the all-point aggregate by 0.68 points. The large-M
code is unchanged; its movement is three-observation node variance.

The remaining ranking is now unambiguous. Flash M8-M64 is 31-42% behind,
Pro M8 and M128 are about 29% and 34% behind, while most M >= 256 points are
within 1-8%. Flash already used the PRMT exponent extractor before R18, so the
next small-M improvement must remove packed-nibble expansion/address work or
hide it behind WGMMA rather than repeat the exponent-only optimization.

Raw logs and the computed table are under
`/app/deepgemm-auto-results/iter24-final-r19-matrix`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter24-final-r19-matrix`.

## Iteration R20: paired packed-word decode for Pro

### Reason and direction

The prior decoder assigned four lanes to the four packed E2M1 words of one
row/K32 group. Every lane independently extracted the same exponent and built
the same two-word E4M3 lookup. R20 remaps the DSV4 Pro decoder so each half
warp owns the same 16 rows and one adjacent packed-word pair:

- one LDS.64 fetches two packed words;
- one exponent lookup is reused by both word decodes;
- one STS.128 publishes the resulting 16 E4M3 bytes;
- two row groups replace the previous four row groups.

All 32 lanes remain active and perform the same total nibble conversion, but
the duplicated lookup, address, LDS, and STS warp instructions are reduced.
The non-Pro path retains the original 8-row/4-word mapping.

The first global prototype improved Pro M32 by 5.59% and 5.40% in opposite
run orders, but regressed Flash M32 by 1.45% and 1.43%. A Pro-only guard alone
did not fully isolate Flash because splitting the old monolithic inline PTX
changed ptxas scheduling. The final implementation therefore also preserves
the exact monolithic `(packed, exponent)` decoder for Flash; only the Pro
paired path uses the separated lookup overload. A final adjacent Flash M32
check measured 466.1 versus 477.8 us, so the earlier regression was removed.

### Generated-code effect

The table compares the R19 and R20 Pro M32 cubins. Opcode counts use the same
static SASS method as R17-R19.

| Pro M32 SASS/resource | R19 | R20 | change |
| --- | ---: | ---: | ---: |
| LOP3 | 1628 | 1435 | -193 |
| SHF | 542 | 455 | -87 |
| SHFL | 150 | 118 | -32 |
| PRMT | 1081 | 958 | -123 |
| IMAD | 1495 | 1469 | -26 |
| LDS | 510 | 382 | -128 |
| STS | 408 | 280 | -128 |
| registers/thread | 128 | 128 | 0 |
| stack | 8 B | 0 B | -8 B |
| local | 0 B | 0 B | 0 |

The paired path emits LDS.64 and STS.128 for the decode body. It removes the
short compiler frame without increasing registers or shared memory.

### Correctness and formal performance

The full eight-rank production cases pass with `fast_math=1`:

| case | diff |
| --- | ---: |
| Pro M8 | 0.000715 |
| Pro M16 | 0.000710 |
| Pro M32 | 0.000715 |
| Pro M64 | 0.000712 |
| Pro M128 | 0.001838 |
| Pro M256 | 0.000713 |

R20 adds the missing Pro M8 and M128 production scenarios. M8 does not require
every rank to wrap because its route population cannot guarantee that; the
M16-M128 cases retain the physical-ring wrap contract.

The formal matched run used 50 observations, ten warmups, 20 launches per
observation, cold L2, and maximum-rank medians:

| Pro point | R19 control us | R20 us | change | same-node PR383 us | R20 gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 955.362 | 881.407 | -7.74% | 708.989 | +24.32% |
| M16 | 1214.500 | 1119.500 | -7.82% | 985.679 | +13.58% |
| M32 | 1261.000 | 1171.500 | -7.10% | 1065.082 | +9.99% |
| M64 | 1321.000 | 1207.500 | -8.59% | 1119.668 | +7.84% |
| M128 | 1660.000 | 1631.000 | -1.75% | 1227.411 | +32.88% |

The matched Pro small-M geometric improvement is `-6.63%`. Its geometric gap
to the same-node PR383 matrix is `+17.35%`, down from R19's `+22.74%`.

A ten-observation large-M screen showed no throughput regression:

| Pro point | R19 control us | R20 us | change |
| --- | ---: | ---: | ---: |
| M256 | 1672.0 | 1668.5 | -0.21% |
| M512 | 2619.5 | 2586.5 | -1.26% |
| M1024 | 4020.5 | 3971.0 | -1.23% |
| M8192 | 25836.0 | 25487.0 | -1.35% |

### NCU and NSYS diagnosis

Eight simultaneous NCU application-replay reports were collected for Pro
M32. Because work distribution is rank-skewed, total executed instructions
across all eight reports are more meaningful than the rank median: R18/R19
uses 12.238 billion instructions versus 9.379 billion for R20, a 23.36%
reduction. Aggregate local-load/store sectors fall from 291072/19968 to zero.

Replay utilization remains noisy: issue-active median is 3.020%, tensor-pipe
active 0.035%, barrier stall 63.875%, long-scoreboard stall 30.020%, and wait
stall 4.795%. Those samples do not replace the matched benchmark; zero local
traffic, lower distributed instruction count, and the static SASS reduction
are the stable profiler evidence.

Low-perturbation NSYS retains exactly one 156-CTA fused MegaMoE launch. Its
rank-0 traced main kernel is 1.435 ms versus 1.524 ms for R18; the trace is
topology evidence, not the benchmark score.

Complete reports remain under
`/app/deepgemm-auto-results/iter25-paired-word-decode`; the compact local export
is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter25-paired-word-decode`.

### Next iteration

Commit the Pro paired decoder, then rerun the complete Flash/Pro matrix. The
remaining Pro M128 gap is schedule/epilogue dominated, while Pro M8-M64 is now
within 8-24% of PR383. Flash remains unchanged and is still the largest small-M
gap, so its next optimization must preserve the monolithic decode schedule or
use a different bank-conflict-free pairing.

## Final matrix after R20

R20 and PR383 were measured consecutively on the same H20 pod after the final
source was synchronized and rebuilt through a fresh JIT cache. The matrix uses
the authoritative DSV4 Flash/Pro shapes and M
`8,16,32,64,128,256,512,1024,2048,4096,8192`. M <= 128 uses 50 observations,
M >= 256 uses three observations, and every observation contains 20 launches
with cold L2. The reported time is the maximum-rank median. PR383 reports the
sum of its L1 and L2 phase durations; R20 reports its one fused persistent
kernel.

| model | M | PR383 us | R20 us | R20 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 302.339 | 410.129 | +35.65% |
| Flash | 16 | 307.546 | 431.620 | +40.34% |
| Flash | 32 | 329.364 | 440.291 | +33.68% |
| Flash | 64 | 364.272 | 504.705 | +38.55% |
| Flash | 128 | 431.221 | 500.753 | +16.12% |
| Flash | 256 | 510.901 | 523.112 | +2.39% |
| Flash | 512 | 930.339 | 950.516 | +2.17% |
| Flash | 1024 | 1548.364 | 1617.000 | +4.43% |
| Flash | 2048 | 2739.407 | 2810.000 | +2.58% |
| Flash | 4096 | 5076.000 | 5325.000 | +4.91% |
| Flash | 8192 | 9845.000 | 10282.000 | +4.44% |
| Pro | 8 | 701.251 | 867.432 | +23.70% |
| Pro | 16 | 975.986 | 1118.500 | +14.60% |
| Pro | 32 | 1064.510 | 1163.000 | +9.25% |
| Pro | 64 | 1110.956 | 1214.000 | +9.28% |
| Pro | 128 | 1219.869 | 1615.500 | +32.43% |
| Pro | 256 | 1666.925 | 1635.000 | -1.92% |
| Pro | 512 | 2417.349 | 2542.000 | +5.16% |
| Pro | 1024 | 4057.000 | 3929.000 | -3.16% |
| Pro | 2048 | 7054.000 | 6962.000 | -1.30% |
| Pro | 4096 | 12997.000 | 13095.000 | +0.75% |
| Pro | 8192 | 24986.000 | 25334.000 | +1.39% |

The same-epoch geometric gaps are:

- all 22 points: `+11.67%`;
- M <= 128: `+24.82%`;
- M >= 256: `+1.79%`;
- Flash small-M: `+32.57%`;
- Pro small-M: `+17.52%`;
- all Flash points: `+15.81%`;
- all Pro points: `+7.68%`.

Compared with the R19 final matrix, the all-point gap falls from `+13.14%` to
`+11.67%`, and Pro small-M falls from `+22.74%` to `+17.52%`. The matched
R19/R20 control already measured the direct Pro small-M code improvement as
`-6.63%`; the remaining difference between final matrices is node-epoch
variation. Flash code is unchanged, so its final-matrix movement is not
credited to R20.

R20 is retained because its improvement is reproduced by the matched control,
the complete matrix, the static SASS reduction, and NCU's 23.36% aggregate
instruction reduction. It still does not meet the terminal goal: Flash M8-M64
is 34-40% behind PR383, Pro M8 is 24% behind, and Pro M128 is 32% behind. R21
therefore targets fixed small-M decode/synchronization cost, starting with a
Flash-safe packed-word pairing and a separate M128 schedule diagnosis.

Raw same-epoch logs are under
`/app/deepgemm-auto-results/iter26-r20-final-matrix`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter26-r20-final-matrix`.

## R21: monolithic paired-word decode for Flash

### Reason and direction

R20 reduced Pro's fixed MXFP4 expansion cost by loading and expanding two
packed words together, but the same source-level construction regressed Flash:
the compiler duplicated lookup construction and extended temporary live
ranges. R21 makes the Flash pairing explicit in one inline PTX block. It builds
the E2M1-to-E4M3 lookup once, decodes both 32-bit packed words serially through
the same four scratch registers, and emits one 128-bit shared-memory store.
The Pro path is deliberately kept byte-for-byte equivalent to R20 in this
iteration so that Flash can be isolated.

Relative to the exact R20 control, Flash M32 static SASS changes are:

| opcode/resource | R20 | R21 | change |
| --- | ---: | ---: | ---: |
| IMAD | 2367 | 1815 | -552 |
| LDS | 514 | 386 | -128 |
| LOP3 | 1707 | 1460 | -247 |
| PRMT | 1030 | 925 | -105 |
| SHF | 587 | 457 | -130 |
| SHFL | 132 | 100 | -32 |
| STS | 420 | 292 | -128 |
| registers/thread | 128 | 128 | unchanged |
| stack/local bytes | 0/0 | 0/0 | unchanged |

The paired decoder passes the production eight-rank physical-ring-wrap cases:
Flash M32 has maximum difference `0.000656`, and Flash M128 has maximum
difference `0.000658`.

### Matched cold-L2 benchmark

The formal matched runs use 50 observations, ten warmups, 20 launches per
observation, cold L2, and maximum-rank medians. Each candidate run was bracketed
by an exact R20 control where practical:

| Flash point | R20 first us | R21 us | change | R20 second us | reverse-order change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 436.040 | 395.743 | -9.24% | 411.962 | -3.94% |
| M16 | 453.419 | 439.959 | -2.97% | 452.526 | -2.78% |
| M32 | 456.822 | 427.592 | -6.40% | - | - |
| M64 | 504.903 | 487.142 | -3.52% | 502.745 | -3.10% |
| M128 | 513.803 | 490.561 | -4.52% | 539.134 | -9.01% |

The first-control geometric improvement across Flash M8-M128 is `-5.36%`.
The four reverse-order points independently reproduce a `-4.74%` geometric
improvement. Two-order ten-observation screens also retain the pairing at
M512, M1024, and M8192; M256 is neutral. Pro M32 produces identical static
opcode and resource counts to the R20 control, confirming that this iteration
does not alter the Pro kernel.

### NCU and NSYS diagnosis

Eight simultaneous NCU application-replay reports were collected for Flash
M32. Aggregate metrics across all ranks are used because the distributed work
is rank-skewed:

| NCU metric, eight-rank aggregate unless noted | R20 | R21 | change |
| --- | ---: | ---: | ---: |
| executed instructions | 1,593,199,640 | 1,085,946,078 | -31.84% |
| executed thread instructions | 37,308,122,624 | 27,247,325,090 | -26.97% |
| theoretical global L2 sectors | 450,190,210 | 241,314,531 | -46.40% |
| theoretical local L2 sectors | 0 | 0 | unchanged |
| registers/thread, rank mean | 128 | 128 | unchanged |
| replay duration, rank mean | 229.447 ms | 169.798 ms | -26.00% |
| issue-active, rank mean | 4.045% | 4.818% | +19.12% |
| tensor-pipe active, rank mean | 0.157% | 0.264% | +67.93% |

Replay duration and PM sampling are profiler-perturbed and are supporting
evidence only. The stable conclusions are the large dynamic-instruction and
sector reductions, unchanged register footprint, zero local traffic, and the
matched benchmark improvement.

Low-perturbation NSYS preserves one 156-CTA fused MegaMoE launch. The rank-0
traced main kernel falls from `779.357 us` to `753.437 us`; NCCL timing varies
between captures, so the trace is used only to confirm launch topology and the
direction of the kernel-local change.

Complete logs and profiler reports are under
`/app/deepgemm-auto-results/iter27-flash-paired-monolithic`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter27-flash-paired-monolithic`.

### Next iteration

R21 is retained because the improvement reproduces across launch order, full
correctness passes, static SASS, NCU, and NSYS. The next step is a fresh full
Flash/Pro matrix against PR383. R22 will then test the same monolithic paired
decoder for Pro, where R20 still uses split helper calls and leaves avoidable
lookup scheduling and live-range overhead.

## Final matrix after R21

R21 and PR383 were measured consecutively on the same H20 pod with fresh JIT
caches. The candidate uses the authoritative fused MXFP4 driver; PR383 uses its
native compatible two-phase FP8 driver, with reported time equal to L1 plus L2.
Both use the DSV4 Flash/Pro shapes, 50 observations for M <= 128, three for
M >= 256, 20 launches per observation, cold L2, and maximum-rank medians.

| model | M | PR383 us | R21 us | R21 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 300.630 | 426.831 | +41.98% |
| Flash | 16 | 305.645 | 434.094 | +42.03% |
| Flash | 32 | 328.809 | 434.549 | +32.16% |
| Flash | 64 | 364.264 | 481.075 | +32.07% |
| Flash | 128 | 433.449 | 484.672 | +11.82% |
| Flash | 256 | 506.221 | 504.392 | -0.36% |
| Flash | 512 | 947.005 | 887.077 | -6.33% |
| Flash | 1024 | 1526.217 | 1577.000 | +3.33% |
| Flash | 2048 | 2749.035 | 2801.000 | +1.89% |
| Flash | 4096 | 5085.000 | 5177.000 | +1.81% |
| Flash | 8192 | 9880.000 | 10067.000 | +1.89% |
| Pro | 8 | 701.005 | 857.922 | +22.39% |
| Pro | 16 | 987.907 | 1113.000 | +12.66% |
| Pro | 32 | 1079.527 | 1163.500 | +7.78% |
| Pro | 64 | 1114.844 | 1209.500 | +8.49% |
| Pro | 128 | 1220.630 | 1626.500 | +33.25% |
| Pro | 256 | 1639.223 | 1628.000 | -0.68% |
| Pro | 512 | 2415.025 | 2559.000 | +5.96% |
| Pro | 1024 | 4050.000 | 3937.000 | -2.79% |
| Pro | 2048 | 7050.000 | 6962.000 | -1.25% |
| Pro | 4096 | 12978.000 | 13120.000 | +1.09% |
| Pro | 8192 | 24961.000 | 25358.000 | +1.59% |

The same-epoch geometric gaps are:

- all 22 points: `+10.47%`;
- M <= 128: `+23.80%`;
- M >= 256: `+0.47%`;
- Flash small-M: `+31.52%`;
- Pro small-M: `+16.53%`;
- all Flash points: `+13.46%`;
- all Pro points: `+7.56%`.

The all-point gap improves from R20's `+11.67%` to `+10.47%`, all-Flash from
`+15.81%` to `+13.46%`, Flash small-M from `+32.57%` to `+31.52%`, and large-M
from `+1.79%` to `+0.47%`. The absolute Flash M8/M16 medians moved with node
epoch, so the R21 code change is credited from the bracketed exact-R20 controls
(`-5.36%` geometric), not from cross-epoch absolute medians alone.

The failed `bench_mega_moe_sm90_r20_standard.py` startup in the artifact is
retained for audit: it passed the newer `num_shared_experts` keyword into the
old PR383 API and failed before timing. The successful matrix uses PR383's
native `tests/bench_mega_moe_sm90.py`, matching all preceding PR383 matrices.

Raw logs are under
`/app/deepgemm-auto-results/iter28-r21-final-matrix`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter28-r21-final-matrix`.

R21 does not meet the terminal goal. Large-M is effectively closed, while the
remaining optimization budget is dominated by Flash M8-M64 and Pro M128. R22
first applies the proven monolithic paired decoder to Pro; if M128 remains
unchanged, the next diagnosis must separate its schedule/dispatch fixed cost
from MXFP4 expansion cost.

## Rejected experiment R22: monolithic paired decoder for Pro

### Reason and direction

R21's Flash decoder placed two packed words and lookup construction in one PTX
block. R22 temporarily selected that helper for Pro as well, replacing R20's
two inlined eight-value decode calls. The intent was to shorten lookup and
temporary live ranges without changing task scheduling, shared-memory layout,
or the Flash kernel.

Eight-rank physical-ring-wrap correctness passed for Pro M32 and M128 at
`0.000715` and `0.000709`. The representative Pro kernel retained
`REG=128, STACK=0, LOCAL=0`, but static SASS fell by only eight IMADs and one
PRMT. A 20-observation R21/R22/R21 screen produced:

| Pro point | R21 first us | R22 us | change | R21 second us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 911.311 | 876.437 | -3.83% | 881.001 | -0.52% |
| M16 | 1132.000 | 1134.000 | +0.18% | 1135.000 | -0.09% |
| M32 | 1186.500 | 1175.000 | -0.97% | 1174.000 | +0.09% |
| M64 | 1227.000 | 1215.000 | -0.98% | 1228.500 | -1.10% |
| M128 | 1665.000 | 1619.500 | -2.73% | 1638.000 | -1.13% |

The screen's geometric improvement was `-1.68%` against the first control but
only `-0.55%` against the second. Formal 50-observation checks then showed M8
regressing by `+0.50%/+1.59%`, while M128 improved by `-1.49%/-0.61%`.
Across the five small-M points the reverse-order screen was effectively neutral
at `-0.03%`. The source experiment was reverted and not committed.

Artifacts are under
`/app/deepgemm-auto-results/iter29-pro-monolithic-pair`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter29-pro-monolithic-pair`.

## R23: extend the current Pro swap-AB path to M128

### Reason and direction

R22 showed that decode-call structure was not the source of Pro M128's 33%
gap. The sharp runtime step instead coincided exactly with the host selector:
M64 used the sparse-expert swap-AB kernel, while M128 switched back to regular
M64xN128 WGMMA despite only 128 routed tokens across 384 experts. An early
branch version had rejected M128 swap-AB by 8.34%, but the swap epilogue,
fragment lifetime, promotion schedule, and MXFP4 decode have since changed
substantially. R23 therefore retests that boundary on the current source by
extending only Pro's routed-only selector from M <= 64 to M <= 128. M8-M64
already select the same path, and M >= 256 remains untouched.

### Correctness and resources

The eight-rank `production.pro_m128` physical-ring-wrap scenario passes at
`diff=0.000700`. The selected kernel remains at `REG=128`, `STACK=0`,
`LOCAL=0`, and 1024 bytes static shared memory. The swap kernel has a larger
static instruction body because it contains N8/N16/N32/N64 buckets; its win is
from executing a smaller WGMMA-N bucket for sparse expert ownership, not from
shrinking total static SASS.

### Matched cold-L2 performance

A first ten-observation R21/R23/R21 screen measured `1664.5/1428.5/1617.0 us`,
or `-14.18%/-11.66%`. The formal run used ten warmups, 50 observations, 20
launches per observation, cold L2, and maximum-rank medians:

| Pro M128 | median us | R23 change |
| --- | ---: | ---: |
| R21 first control | 1653.500 | - |
| R23 | 1426.000 | -13.76% |
| R21 second control | 1631.500 | -12.60% |

Using the immediately preceding same-node PR383 M128 value of `1220.630 us`,
R23's provisional gap is about `+16.82%`, down from R21's `+33.25%`. A fresh
full matrix will establish the final same-epoch gap.

R23 is retained because the improvement is large, reproduces in both launch
orders at both 10 and 50 observations, passes production correctness, and
does not add spilling or alter adjacent selectors. Raw artifacts are under
`/app/deepgemm-auto-results/iter30-pro-m128-swap-retest`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter30-pro-m128-swap-retest`.

## Final matrix after R23

R23 and PR383 were measured consecutively on the same H20 pod with fresh JIT
caches. The candidate uses the authoritative fused MXFP4 benchmark, while
PR383 uses its native compatible two-phase driver and reports L1 plus L2.
Small-M points use 50 observations, large-M points use three, and every
observation contains 20 launches with cold L2 and maximum-rank medians.

| model | M | PR383 us | R23 us | R23 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 301.405 | 410.895 | +36.33% |
| Flash | 16 | 314.704 | 444.815 | +41.34% |
| Flash | 32 | 323.060 | 436.988 | +35.27% |
| Flash | 64 | 363.907 | 479.564 | +31.78% |
| Flash | 128 | 441.379 | 496.806 | +12.56% |
| Flash | 256 | 545.820 | 494.466 | -9.41% |
| Flash | 512 | 926.856 | 1008.000 | +8.76% |
| Flash | 1024 | 1510.966 | 1544.000 | +2.19% |
| Flash | 2048 | 2716.829 | 2768.000 | +1.88% |
| Flash | 4096 | 5045.000 | 5173.000 | +2.54% |
| Flash | 8192 | 9803.000 | 10045.000 | +2.47% |
| Pro | 8 | 690.600 | 865.957 | +25.39% |
| Pro | 16 | 981.139 | 1116.000 | +13.74% |
| Pro | 32 | 1078.618 | 1168.000 | +8.29% |
| Pro | 64 | 1132.643 | 1207.500 | +6.61% |
| Pro | 128 | 1233.846 | 1423.000 | +15.33% |
| Pro | 256 | 1653.253 | 1613.000 | -2.43% |
| Pro | 512 | 2447.575 | 2535.000 | +3.57% |
| Pro | 1024 | 4035.000 | 3925.000 | -2.73% |
| Pro | 2048 | 7007.000 | 6984.000 | -0.33% |
| Pro | 4096 | 13016.000 | 13079.000 | +0.48% |
| Pro | 8192 | 24997.000 | 25365.000 | +1.47% |

The same-epoch geometric gaps are:

- all 22 points: `+9.85%`;
- M <= 128: `+22.06%`;
- M >= 256: `+0.62%`;
- Flash small-M: `+31.06%`;
- Pro small-M: `+13.69%`;
- all Flash points: `+13.85%`;
- all Pro points: `+5.99%`;
- Pro large-M: `-0.02%`.

R23 reduces the all-point gap from R21's `+10.47%` to `+9.85%`, Pro all-point
from `+7.56%` to `+5.99%`, Pro small-M from `+16.53%` to `+13.69%`, and Pro
M128 from `+33.25%` to `+15.33%`. Flash and every point outside Pro M128 are
source-identical to R21; their movement is node-epoch variance and is not
credited to R23.

Raw logs are under `/app/deepgemm-auto-results/iter31-r23-final-matrix`; the
local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter31-r23-final-matrix`.

The terminal goal is still unmet. Pro large-M is closed and Pro M16-M128 is
within 7-15%, but Flash M8-M64 remains 32-41% behind PR383 and now dominates
the geometric gap. The next iteration returns to the Flash swap-AB latency
kernel and targets its fixed epilogue/synchronization cost rather than the
already-reduced MXFP4 decode body.

## R24: extend the current Flash swap-AB path to M64

### Reason and direction

The remaining Flash latency gap had the same selector discontinuity that R23
resolved for Pro: M32 used sparse-expert swap-AB, while M64 switched to the
regular M64xN128 WGMMA kernel. The original R02 source rejected Flash M64
swap-AB by only `+1.17%`; since then fragment reuse removed the swap frame and
paired decoding reduced its MXFP4 body. R24 therefore retests the boundary on
the current source by extending routed-only Flash swap-AB from M <= 32 to
M <= 64. Flash M8-M32 already select this path, M >= 128 remains regular, and
all Pro selectors are unchanged.

The correctness suite previously had no production Flash M64 case, so R24
adds `production.flash_m64` with the same DSV4 shape and forced physical-ring
wrap contract as M32/M128. The new eight-rank case passes at `diff=0.000654`.
Both the R23 regular control and R24 swap kernel use `REG=128`, `STACK=0`,
`LOCAL=0`, and 1024 bytes static shared memory.

A ten-observation R23/R24/R23 screen measured
`515.841/476.958/486.771 us`, or `-7.54%/-2.02%`. The formal run used ten
warmups, 50 observations, 20 launches per observation, cold L2, and
maximum-rank medians:

| Flash M64 | median us | R24 change |
| --- | ---: | ---: |
| R23 first control | 490.817 | - |
| R24 | 464.221 | -5.42% |
| R23 second control | 494.298 | -6.08% |

R24 is retained because the formal improvement is consistent in both orders,
the added production scenario proves the new selector under physical ring
wrap, and resources remain spill-free. Artifacts are under
`/app/deepgemm-auto-results/iter32-flash-m64-swap-retest`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter32-flash-m64-swap-retest`.

## Rejected experiment R25: extend Flash swap-AB to M128

R25 temporarily extended the newly accepted Flash selector one point farther
to M128. The existing eight-rank physical-ring-wrap scenario passed at
`diff=0.000651`, but the ten-observation R24-control/candidate/control sequence
measured `498.147/518.301/507.805 us`. The candidate regressed by
`+4.05%/+2.07%` in both orders, confirming that M128 remains above the current
swap-AB crossover. The source was restored to R24's M64 cutoff without a
commit. Artifacts are under
`/app/deepgemm-auto-results/iter33-flash-m128-swap-retest`; the local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter33-flash-m128-swap-retest`.

## Final matrix after R24

The R24 candidate was run across the complete authoritative matrix with a
fresh JIT cache. PR383 is the immediately preceding iter31 native two-phase
matrix from the same H20 pod; neither PR383 source nor the pod changed between
runs, so its log is copied into the R24 artifact rather than spending another
identical full run. Both use 50 observations for M <= 128, three for M >= 256,
20 launches per observation, cold L2, and maximum-rank medians. Direct
R23/R24/R23 controls remain the change-attribution evidence for Flash M64.

| model | M | PR383 us | R24 us | R24 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 301.405 | 407.397 | +35.17% |
| Flash | 16 | 314.704 | 433.674 | +37.80% |
| Flash | 32 | 323.060 | 429.279 | +32.88% |
| Flash | 64 | 363.907 | 449.459 | +23.51% |
| Flash | 128 | 441.379 | 483.269 | +9.49% |
| Flash | 256 | 545.820 | 516.773 | -5.32% |
| Flash | 512 | 926.856 | 921.571 | -0.57% |
| Flash | 1024 | 1510.966 | 1531.000 | +1.33% |
| Flash | 2048 | 2716.829 | 2759.000 | +1.55% |
| Flash | 4096 | 5045.000 | 5156.000 | +2.20% |
| Flash | 8192 | 9803.000 | 10102.000 | +3.05% |
| Pro | 8 | 690.600 | 854.297 | +23.70% |
| Pro | 16 | 981.139 | 1126.000 | +14.76% |
| Pro | 32 | 1078.618 | 1154.000 | +6.99% |
| Pro | 64 | 1132.643 | 1208.500 | +6.70% |
| Pro | 128 | 1233.846 | 1402.500 | +13.67% |
| Pro | 256 | 1653.253 | 1656.000 | +0.17% |
| Pro | 512 | 2447.575 | 2554.000 | +4.35% |
| Pro | 1024 | 4035.000 | 3941.000 | -2.33% |
| Pro | 2048 | 7007.000 | 6985.000 | -0.31% |
| Pro | 4096 | 13016.000 | 13126.000 | +0.85% |
| Pro | 8192 | 24997.000 | 25473.000 | +1.90% |

The resulting geometric gaps are:

- all 22 points: `+8.94%`;
- M <= 128: `+19.95%`;
- M >= 256: `+0.54%`;
- Flash small-M: `+27.33%`;
- Pro small-M: `+13.00%`;
- all Flash points: `+11.81%`;
- all Pro points: `+6.14%`.

Relative to R23, the all-point gap falls from `+9.85%` to `+8.94%`, Flash
small-M from `+31.06%` to `+27.33%`, and all-Flash from `+13.85%` to `+11.81%`.
Only Flash M64 changed source, and its directly matched `-5.42%/-6.08%` result
is the credited improvement; other point movement is node variation.

Raw logs are under `/app/deepgemm-auto-results/iter34-r24-final-matrix`; the
local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter34-r24-final-matrix`.

The goal is still not complete. Large-M remains within geometric noise, but
Flash M8-M32 is 33-38% behind PR383 and Pro M8 is 24% behind. Future work must
reduce fixed swap-AB latency inside the already-correct selectors; widening
the Flash selector beyond M64 is now formally ruled out.

## R26: compile-time bounds for the swap-AB epilogue

### Reason and direction

The swap-AB math mainloop already dispatches N8/N16/N32/N64 WGMMA from each
expert's actual `valid_m`, but both L1 and L2 epilogues still instantiated all
eight token chunks of the M64 tile. Runtime predicates suppressed invalid
loads and stores, but the unused chunks still enlarged the live fragment and
executed control/address instructions. This was especially wasteful at global
M8, M16, and M32, where no expert can own more than one, two, or four chunks.

R26 passes the rounded global token upper bound (8/16/32/64) from the host JIT
into the persistent-kernel template. `kSwapABTokenChunks` now uses that bound,
so the compiler removes impossible epilogue chunks while the math mainloop
continues to choose a still-smaller bucket from each expert's actual token
count. M64 and larger retain the original eight-chunk bound. The full suite
also gains explicit `production.flash_m8` and `production.flash_m16` physical
ring-wrap scenarios.

### Correctness and resources

All changed eight-rank production cases pass:

| model | M | calc diff |
| --- | ---: | ---: |
| Flash | 8 | 0.000649 |
| Flash | 16 | 0.000645 |
| Flash | 32 | 0.000656 |
| Flash | 64 boundary | 0.000654 |
| Pro | 8 | 0.000725 |
| Pro | 16 | 0.000718 |
| Pro | 32 | 0.000715 |

Official-benchmark JIT cubins remain spill-free (`STACK=0`, `LOCAL=0`) and
use 1024 bytes of static shared memory. Flash M8/M16/M32 fall from the R24
control's 128 registers to 114/114/122; Pro M8/M16/M32 fall to 107/110/122.
M64 retains 128 registers, as expected from its unchanged eight-chunk bound.

### Matched cold-L2 performance

The formal runs use the authoritative benchmark with ten warmups, 50
observations, 20 launches per observation, cold L2, and maximum-rank medians.
R23 is an exact code control for M8-M32 because R24 changed only Flash M64.

| model | M | first control us | R26 us | change | second control us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash | 8 | 407.552 | 398.751 | -2.16% | 394.552 | +1.06% |
| Flash | 16 | 426.754 | 404.858 | -5.13% | 422.286 | -4.13% |
| Flash | 32 | 420.843 | 415.338 | -1.31% | 440.503 | -5.71% |
| Pro | 8 | 884.382 | 856.493 | -3.15% | 888.608 | -3.61% |
| Pro | 16 | 1114.500 | 1076.000 | -3.45% | 1111.500 | -3.19% |
| Pro | 32 | 1167.500 | 1134.000 | -2.87% | 1172.000 | -3.24% |

Flash M8-M32 improve geometrically by `-2.88%/-2.97%` against the two
controls. Pro M8-M32 improve by `-3.16%/-3.35%`. Flash M8 alone straddles
noise, but the other five comparisons and both model-level geometric means
reproduce in both launch orders.

### NCU and NSYS attribution

An isolated M16 Flash NCU run uses one rank and 32 experts to remove
distributed spin-wait perturbation while preserving the changed epilogue
shape. R24 control versus R26 measures:

| NCU metric | control | R26 | change |
| --- | ---: | ---: | ---: |
| `smsp__inst_executed.sum` | 78,936,825 | 73,213,868 | -7.25% |
| `smsp__thread_inst_executed.sum` | 2,471,623,315 | 2,288,589,254 | -7.41% |
| global-load sectors | 844,907 | 844,159 | -0.09% |
| local-load sectors | 0 | 0 | unchanged |
| local-store sectors | 0 | 0 | unchanged |

The instruction reduction with unchanged traffic and no local memory directly
matches the intended removal of impossible epilogue chunks. An eight-rank
NSYS capture of the same point measures the main kernel at 804.764 us for the
control and 797.468 us for R26 (`-0.91%`) in a single profiled launch; the
formal repeated benchmark above remains the performance authority.

Eight-rank NCU application-replay reports are retained, but their aggregate
spin-wait instruction/sector counts are explicitly not used for comparison:
the profiler independently relaunches distributed ranks, so small schedule
changes alter how long peers spin and overwhelm this epilogue-sized delta.

Screen artifacts are under
`/app/deepgemm-auto-results/iter35-swap-epilogue-bounds-screen`, formal Flash
and Pro artifacts under `iter36-swap-epilogue-bounds-formal` and
`iter37-pro-swap-epilogue-bounds-formal`, and profiler reports under
`iter38-swap-epilogue-profiles`. Local exports use the matching directory
names below
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts`.

R26 is retained. It reduces the fixed swap-AB epilogue cost across both DSV4
models without changing M64+ execution or adding spills. The terminal goal is
still unmet; the next full candidate/PR383 matrix will quantify the remaining
gap before the next optimization iteration.

## Final matrix after R26

R26 and PR383 were run consecutively on the same H20 pod. The candidate uses
the authoritative fused MXFP4 benchmark and PR383 uses its native compatible
two-phase driver, reporting L1 plus L2. Candidate small-M points use ten
warmups and both drivers use 50 observations for M <= 128, three for M >= 256,
20 launches per observation, cold L2, and maximum-rank medians. PR383 has one
explicit driver warmup, matching all preceding PR383 matrices; cold-L2 timing
makes the different untimed warmup counts immaterial.

| model | M | PR383 us | R26 us | R26 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 299.531 | 410.841 | +37.16% |
| Flash | 16 | 308.502 | 428.164 | +38.79% |
| Flash | 32 | 333.619 | 433.800 | +30.03% |
| Flash | 64 | 367.438 | 456.904 | +24.35% |
| Flash | 128 | 445.626 | 502.369 | +12.73% |
| Flash | 256 | 495.505 | 504.141 | +1.74% |
| Flash | 512 | 905.925 | 924.273 | +2.03% |
| Flash | 1024 | 1534.908 | 1521.000 | -0.91% |
| Flash | 2048 | 2734.939 | 2760.000 | +0.92% |
| Flash | 4096 | 5099.000 | 5195.000 | +1.88% |
| Flash | 8192 | 9853.000 | 10045.000 | +1.95% |
| Pro | 8 | 704.874 | 848.015 | +20.31% |
| Pro | 16 | 979.115 | 1080.500 | +10.35% |
| Pro | 32 | 1066.244 | 1131.000 | +6.07% |
| Pro | 64 | 1117.183 | 1199.000 | +7.32% |
| Pro | 128 | 1232.130 | 1406.500 | +14.15% |
| Pro | 256 | 1667.199 | 1632.000 | -2.11% |
| Pro | 512 | 2450.788 | 2525.000 | +3.03% |
| Pro | 1024 | 4042.000 | 3924.000 | -2.92% |
| Pro | 2048 | 7044.000 | 6972.000 | -1.02% |
| Pro | 4096 | 13004.000 | 13110.000 | +0.82% |
| Pro | 8192 | 24987.000 | 25488.000 | +2.01% |

The same-epoch geometric gaps are:

- all 22 points: `+8.83%`;
- M <= 128: `+19.60%`;
- M >= 256: `+0.60%`;
- all Flash points: `+12.74%`;
- Flash small-M: `+28.25%`;
- Flash large-M: `+1.26%`;
- all Pro points: `+5.05%`;
- Pro small-M: `+11.53%`;
- Pro large-M: `-0.06%`.

R26's exact R24-equivalent A/B controls, rather than cross-epoch R24 matrix
movement, remain the source attribution: Flash M8-M32 improve geometrically by
2.88-2.97% and Pro M8-M32 by 3.16-3.35%. The fresh matrix shows the resulting
state against PR383. Pro large-M is closed, and both models' large-M aggregate
is within 0.60%; the remaining budget is fixed latency at Flash M8-M64 and Pro
M8/M128.

Raw logs are under `/app/deepgemm-auto-results/iter39-r26-final-matrix`; the
local export is
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts/iter39-r26-final-matrix`.

The terminal goal is still not complete. R27 should target work scheduling,
dispatch/combine synchronization, or the number of persistent tasks reached
at sparse expert occupancy. Further instruction-only trimming of the now
bounded epilogue is unlikely to close Flash's remaining 24-39% small-M gap.

## R27: sparse dispatch completion for Flash M32

### Reason and direction

Dense dispatch contributes one completion atomic for every expert from every
CTA, including zero local counts. With 156 logical CTAs and 256 Flash experts,
that is 39,936 expert completion atomics even when M8-M32 contains only
48-192 routes per rank. The kernel already has a production sparse-completion
path for Flash M1024: CTAs issue atomics only for nonzero local counts, the
existing grid/NVLink rendezvous proves completion, and SM0 aggregates the
eight rank-local expert counts. It uses the same number of grid barriers as
the dense path.

R27 first tested this existing path at Flash M8, M16, and M32. All three
eight-rank physical-ring-wrap scenarios remained bitwise-equivalent within
their previous tolerances (`0.000649`, `0.000645`, and `0.000656`). Formal
performance rejected M8 and M16, however, so the retained host selector is
deliberately exact: Flash M32 and the pre-existing Flash M1024 point only.

### Rejected broad selector and retained M32 result

The formal run uses ten warmups, 50 observations, 20 launches per
observation, cold L2, and maximum-rank medians against the exact R26 commit
`6a20cec`:

| Flash point | first R26 us | sparse us | change | second R26 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 380.732 | 388.059 | +1.92% | 386.782 | +0.33% |
| M16 | 414.444 | 429.259 | +3.57% | 421.474 | +1.85% |
| M32 | 431.728 | 415.297 | -3.81% | 417.442 | -0.51% |

The M8/M16 selectors were removed after both launch orders regressed. M32 was
then independently repeated in candidate/control order at `442.923/445.956 us`,
another `-0.68%`. Three 50-observation comparisons therefore agree on
the sign for M32 (`-3.81%`, `-0.51%`, `-0.68%`) while adjacent points remain
on R26's dense path.

### NCU and NSYS attribution

An isolated one-rank, 32-expert NCU profile removes distributed spin-wait
noise while preserving the M32 dispatch choice:

| NCU metric | R26 dense | R27 sparse | change |
| --- | ---: | ---: | ---: |
| L1 global atomic sectors | 4,396 | 3,171 | -27.87% |
| L2 atomic sectors | 6,421 | 4,622 | -28.02% |
| global-load sectors | 899,720 | 916,889 | +1.91% |
| `smsp__inst_executed.sum` | 80,210,834 | 80,218,488 | +0.01% |
| `smsp__thread_inst_executed.sum` | 2,511,739,800 | 2,512,160,440 | +0.02% |
| local load/store sectors | 0/0 | 0/0 | unchanged |

Sparse completion trades the SM0 rank-count loads for substantially fewer
global atomics rather than reducing the math instruction body. An eight-rank
NSYS single-launch capture measures the main kernel at 746.429 us for R26 and
663.293 us for R27 (`-11.14%`); the repeated cold-L2 measurements above remain
the acceptance authority.

Screen, formal, confirmation, and profiler artifacts are respectively under
`/app/deepgemm-auto-results/iter40-flash-sparse-dispatch-screen`,
`iter41-flash-sparse-dispatch-formal`,
`iter42-flash-m32-sparse-confirm`, and
`iter43-flash-m32-sparse-profiles`. Local exports use matching directory names
below `/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts`.

R27 is retained only for Flash M32. The terminal goal remains unmet; the dense
path is faster at M8/M16, so their remaining gap requires a different way to
reduce frontend/barrier latency rather than simply skipping zero-count
completion atomics.

## Final matrix after R27

The R27 candidate was run across the complete authoritative matrix. PR383 is
the unchanged iter39 native two-phase log from the same pod; the source,
hardware, and benchmark settings are unchanged. Both logs use 50 observations
for M <= 128, three for M >= 256, 20 launches per observation, cold L2, and
maximum-rank medians.

| model | M | PR383 us | R27 us | R27 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 299.531 | 394.786 | +31.80% |
| Flash | 16 | 308.502 | 419.919 | +36.12% |
| Flash | 32 | 333.619 | 416.520 | +24.85% |
| Flash | 64 | 367.438 | 460.693 | +25.38% |
| Flash | 128 | 445.626 | 486.371 | +9.14% |
| Flash | 256 | 495.505 | 596.176 | +20.32% |
| Flash | 512 | 905.925 | 934.160 | +3.12% |
| Flash | 1024 | 1534.908 | 1569.000 | +2.22% |
| Flash | 2048 | 2734.939 | 2780.000 | +1.65% |
| Flash | 4096 | 5099.000 | 5185.000 | +1.69% |
| Flash | 8192 | 9853.000 | 10061.000 | +2.11% |
| Pro | 8 | 704.874 | 839.829 | +19.15% |
| Pro | 16 | 979.115 | 1065.000 | +8.77% |
| Pro | 32 | 1066.244 | 1140.000 | +6.92% |
| Pro | 64 | 1117.183 | 1207.000 | +8.04% |
| Pro | 128 | 1232.130 | 1416.000 | +14.92% |
| Pro | 256 | 1667.199 | 1647.000 | -1.21% |
| Pro | 512 | 2450.788 | 2566.000 | +4.70% |
| Pro | 1024 | 4042.000 | 3944.000 | -2.42% |
| Pro | 2048 | 7044.000 | 6981.000 | -0.89% |
| Pro | 4096 | 13004.000 | 13107.000 | +0.79% |
| Pro | 8192 | 24987.000 | 25319.000 | +1.33% |

The raw same-node geometric gaps are:

- all 22 points: `+9.40%`;
- M <= 128: `+18.09%`;
- M >= 256: `+2.64%`;
- all Flash points: `+13.69%`;
- Flash small-M: `+25.11%`;
- Flash large-M: `+4.98%`;
- all Pro points: `+5.26%`;
- Pro small-M: `+11.46%`;
- Pro large-M: `+0.36%`.

Flash M256 is an explicit epoch anomaly, not an R27 source regression: R27
only changes generated code at Flash M32, while M256 is byte-identical to R26.
A subsequent PR383/R27/PR383 three-observation check measured
`602.613/563.397/528.811 us`; PR383 itself moved 14% across the two controls.
The raw iter44 value remains in the table for audit, but it is not attributed
to R27 and should not drive the next optimization. R27's source attribution
remains the three exact M32 comparisons in the preceding section.

Raw matrix logs are under
`/app/deepgemm-auto-results/iter44-r27-final-matrix`, and the epoch check is
under `iter45-flash-m256-epoch-check`. Local exports use matching names below
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts`.

The goal is still not complete. Excluding the anomalous unchanged M256 point,
large-M remains near PR383 while the dominant verified gaps are Flash M8/M16/
M64 and Pro M8/M128. R28 must change the dense frontend or task schedule rather
than extending sparse completion to points where it formally regressed.

## Rejected experiment R28: skip the second sparse-completion grid sync

### Reason and direction

The Flash M32 sparse-completion path first synchronizes all 156 CTAs after
publishing per-rank counts, then has SM0 aggregate the 32 rank-local expert
totals and performs a second grid synchronization. Every scheduler already
polls the packed ready high word in `fetch_expert_recv_count()`, so R28 tested
whether nonzero CTAs could proceed directly to that acquire loop after SM0's
publication. Flash M1024 retained the established two-sync sequence.

Eight-rank `production.flash_m32` correctness passed with `diff=0.000656`, but
the formal R27/R28/R27 run did not reproduce a win. It used ten warmups, 50
observations, 20 launches per observation, cold L2, and maximum-rank medians:

| Flash point | first R27 us | R28 us | change | second R27 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M32 | 425.158 | 421.790 | -0.79% | 420.533 | +0.30% |

The sign changes across the two controls, so the change was fully reverted.
The post-aggregation grid rendezvous is not a stable latency bottleneck at
M32. Raw evidence is under
`/app/deepgemm-auto-results/iter47-flash-m32-no-grid-formal`; the local export
uses the matching directory below
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts`.

## R29: single-block expert lookup for Flash M8 and M64

### Reason and direction

DSV4 Flash has 32 local experts per rank, exactly one expert per scheduler
lane. For every claimed L1 or L2 N tile, the general scheduler reconstructed
the same M-block-to-expert mapping with a warp prefix sum. At small M, each
nonempty expert normally owns only one M64 block, so its pool-block index is
simply its ordinal in the nonempty-lane mask.

R29 adds a compile-time latency selector and a runtime-safe fast path:

- one ballot detects any expert with more than 64 received tokens;
- when none exists, a second ballot plus `__fns` selects the owner directly;
- any skewed multi-block distribution falls back to the unchanged prefix-sum
  implementation;
- the final selector is exact for Flash M8 and M64. Flash M16/M32 and every
  non-swap/large-M specialization compile the old scheduler path. DSV4 Pro has
  48 local experts, so its two-experts-per-lane scheduler is unchanged.

### Correctness and formal shape selection

Eight-rank production validation passes Flash M32 at `diff=0.000656` during
the broad screen and Flash M64 at `diff=0.000654` after formal selection, both
with physical ring wrap. The formal R27/R29/R27 run used the authoritative
ten warmups, 50 observations, 20 launches, cold L2, and max-rank medians:

| Flash point | first R27 us | broad R29 us | change | second R27 us | reverse change | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M8 | 396.201 | 393.090 | -0.79% | 399.895 | -1.70% | accept |
| M16 | 412.611 | 415.151 | +0.62% | 421.115 | -1.42% | reject |
| M32 | 412.735 | 423.995 | +2.73% | 420.938 | +0.73% | reject |
| M64 | 448.635 | 437.068 | -2.58% | 447.945 | -2.43% | accept |

The retained M8/M64 pair improves by `-1.69%` geometric mean versus the first
control and `-2.07%` versus the second. M16 changed sign and M32 regressed in
both orders, so their compile-time selectors were removed before commit.

### NCU and NSYS analysis

An isolated one-rank, 32-expert M64 NCU run keeps the exact Flash scheduling
shape while avoiding distributed replay skew:

| NCU metric | R27 | R29 | change |
| --- | ---: | ---: | ---: |
| executed warp instructions | 94,252,780 | 94,252,607 | -0.0002% |
| executed thread instructions | 2,960,902,839 | 2,960,868,408 | -0.0012% |
| global-load sectors | 956,108 | 955,131 | -0.10% |
| global atomic sectors | 4,396 | 4,396 | unchanged |
| local load/store sectors | 0/0 | 0/0 | unchanged |

The aggregate instruction delta is deliberately small because only the
task-owner lookup changes; the formal win comes from shortening that uniform
dependency chain on the scheduler critical path. Both cubins retain 128
registers, zero stack, and zero local allocation.

Eight-rank NSYS single-launch durations were 696.221 us for R27 and 706.813 us
for R29 (`+1.52%`), opposite to both formal 50-observation controls. As with
earlier distributed traces, that one perturbed launch is retained as topology
evidence only: both variants remain one 156-CTA fused kernel. The cold-L2
benchmark above is the acceptance authority.

Screen, formal, and profiler artifacts are respectively under
`/app/deepgemm-auto-results/iter48-single-block-scheduler-screen`,
`iter49-flash-single-block-scheduler-formal`, and
`iter50-flash-m64-single-block-profiles`; local exports use matching names
below `/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts`.

The terminal goal remains unmet. R29 removes about 2% from two Flash latency
points but does not close their remaining PR383 gap. The next scheduler
experiment should generalize the direct lookup to Pro's two experts per lane,
then independently select M8 and M128; further Flash work should target the
dispatch/token-pull critical path rather than another grid barrier.

## Rejected experiment R30: two-group direct scheduler lookup for Pro

DSV4 Pro has 48 local experts, so R30 concatenated two nonempty-lane masks and
used a direct `__fns` owner lookup when every expert fit one M64 block. A
multi-block expert still fell back to the general scheduler. Eight-rank
correctness passed Pro M8/M64/M128 at `0.000726/0.001234/0.001379`, including
the M64/M128 physical ring wraps.

The 20-observation R29/R30/R29 screen rejected all three points because none
improved against both controls:

| Pro point | first R29 us | R30 us | change | second R29 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 874.742 | 874.512 | -0.03% | 855.096 | +2.27% |
| M64 | 1243.500 | 1227.000 | -1.33% | 1223.500 | +0.29% |
| M128 | 1438.500 | 1429.500 | -0.63% | 1418.500 | +0.78% |

The extra second-group ballots and selection offset the avoided prefix sums.
R30 was fully reverted. Evidence is under
`/app/deepgemm-auto-results/iter51-pro-two-group-scheduler-screen`, with a
matching local export below the profile-artifact root.

## Rejected experiment R31: packed-BF16 epilogue for Pro M128

R23 made Pro M128's swap-AB path viable after the original packed-BF16 sweep,
so R31 retested the current mature packed epilogue at exactly M128. Correctness
passed at `diff=0.000700`, but a 20-observation R29/R31/R29 screen measured
`1455.5/1476.0/1444.0 us`. The candidate regressed by `+1.41%/+2.22%` and was
reverted. The current M128 cost is not the temporary FP32 expansion targeted
by this specialization. Evidence is under
`/app/deepgemm-auto-results/iter52-pro-m128-packed-bf16-screen`.

## R32: retest PRMT exponent extraction for paired Pro M8 decode

### Reason and direction

R19 rejected Pro M8 PRMT extraction by 0.30% while the decoder still assigned
one lane to each packed word. R20 later replaced that body with paired-word
decode and shared exponent lookup. R32 therefore retests the selector on the
current dependency structure and changes only routed DSV4 Pro M8. Existing
Pro M16/M32/M64 and Flash selectors are unchanged.

### Correctness and formal result

Eight-rank `production.pro_m8` passes at `diff=0.000725`. The formal
R29/R32/R29 run uses ten warmups, 50 observations, 20 launches per observation,
cold L2, and maximum-rank medians:

| Pro point | first R29 us | R32 us | change | second R29 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 855.248 | 853.010 | -0.26% | 856.727 | -0.43% |

The gain is small but agrees in both orders and reverses the old R19 result on
the pre-paired decoder, so the exact M8 selector is retained.

### NCU and NSYS analysis

One-rank NCU uses 48 experts to preserve Pro's two-experts-per-lane layout:

| NCU metric | R29 | R32 | change |
| --- | ---: | ---: | ---: |
| executed warp instructions | 205,054,873 | 202,922,321 | -1.04% |
| executed thread instructions | 6,435,852,130 | 6,367,716,576 | -1.06% |
| global-load sectors | 17,103,543 | 17,103,214 | -0.002% |
| global atomic sectors | 6,408 | 6,408 | unchanged |
| local load/store sectors | 0/0 | 0/0 | unchanged |
| NCU kernel duration us | 809.568 | 801.952 | -0.94% |

Both cubins retain 107 registers, zero stack, and zero local allocation. The
instruction reduction with unchanged memory traffic matches the intended
shift/mask-to-PRMT substitution.

The eight-rank NSYS single launch was 1257.436 us for R29 and 1261.627 us for
R32 (`+0.33%`), opposite to both 50-observation controls and isolated NCU. It
is retained as one-kernel/156-CTA topology evidence, not as the acceptance
score.

Formal and profiler artifacts are under
`/app/deepgemm-auto-results/iter53-pro-m8-prmt-retest-formal` and
`iter54-pro-m8-prmt-retest-profiles`; local exports use matching names below
`/Users/huangzhilin/security_inference/DeepGEMM-profile-artifacts`.

The terminal goal is still not complete. R32 recovers only about 0.3% at Pro
M8; the next authoritative full matrix must measure the combined R29/R32
branch against a fresh same-epoch PR383 run before another structural
dispatch or mainloop change.

## Final matrix after R32

R32 and PR383 were run consecutively on the same H20 pod. Both use the
authoritative DSV4 Flash/Pro shapes and M
`8,16,32,64,128,256,512,1024,2048,4096,8192`: M <= 128 has 50 observations,
M >= 256 has three, every observation has 20 launches with cold L2, and the
score is the maximum-rank median. PR383 reports the sum of its native L1/L2
phase kernels; R32 reports its fused persistent kernel.

| model | M | PR383 us | R32 us | R32 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 307.925 | 385.280 | +25.12% |
| Flash | 16 | 308.482 | 421.392 | +36.60% |
| Flash | 32 | 329.268 | 413.592 | +25.61% |
| Flash | 64 | 366.392 | 464.909 | +26.89% |
| Flash | 128 | 433.574 | 485.777 | +12.04% |
| Flash | 256 | 511.177 | 523.341 | +2.38% |
| Flash | 512 | 922.909 | 897.317 | -2.77% |
| Flash | 1024 | 1521.499 | 1538.000 | +1.08% |
| Flash | 2048 | 2723.100 | 2787.000 | +2.35% |
| Flash | 4096 | 5059.000 | 5164.000 | +2.08% |
| Flash | 8192 | 9829.000 | 10055.000 | +2.30% |
| Pro | 8 | 706.418 | 836.358 | +18.39% |
| Pro | 16 | 979.714 | 1070.500 | +9.27% |
| Pro | 32 | 1064.703 | 1135.500 | +6.65% |
| Pro | 64 | 1106.698 | 1215.500 | +9.83% |
| Pro | 128 | 1219.255 | 1419.000 | +16.38% |
| Pro | 256 | 1637.733 | 1658.000 | +1.24% |
| Pro | 512 | 2396.359 | 2526.000 | +5.41% |
| Pro | 1024 | 4074.000 | 3926.000 | -3.63% |
| Pro | 2048 | 7014.000 | 6976.000 | -0.54% |
| Pro | 4096 | 12989.000 | 13239.000 | +1.92% |
| Pro | 8192 | 24966.000 | 25446.000 | +1.92% |

The fresh same-epoch geometric gaps are:

- all 22 points: `+8.61%`;
- M <= 128: `+18.33%`;
- M >= 256: `+1.12%`;
- all Flash points: `+11.41%`;
- Flash small-M: `+25.00%`;
- Flash large-M: `+1.22%`;
- all Pro points: `+5.88%`;
- Pro small-M: `+12.02%`;
- Pro large-M: `+1.02%`.

This is the lowest fresh all-point gap recorded on the branch, but it remains
far from the terminal goal. Large-M is near PR383 and Flash M512 plus Pro
M1024/M2048 already win; the next work must prioritize the verified Flash
latency gaps, especially M16 (`+36.60%`) and M64/M32/M8 (`+25-27%`). Raw logs
are under `/app/deepgemm-auto-results/iter55-r32-final-matrix`, with a matching
local export below the profile-artifact root.

## Rejected experiment R33: rank-major Flash dispatch pull

### Reason and direction

The Flash dispatch pull loop rebuilt the SM100 round-robin rank schedule for
every routed token. R33 tested a rank-major prefix lookup for the one-wave
small-M Flash path: one warp prefix sum and ballot selected the source rank,
while the existing source metadata preserved combine identity.

Eight-rank correctness passed Flash M8/M16/M32/M64 at
`0.000649/0.000645/0.000656/0.000654`. A broad 20-observation screen gave:

| Flash point | first R32-equivalent us | R33 us | change | second control us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 384.439 | 399.817 | +4.00% | 415.007 | -3.66% |
| M16 | 426.691 | 453.448 | +6.27% | 442.527 | +2.47% |
| M32 | 432.685 | 423.155 | -2.20% | 425.428 | -0.53% |
| M64 | 463.276 | 456.064 | -1.56% | 452.086 | +0.88% |

Only M32 won in that screen, so the selector was narrowed to exactly M32 and
rerun with the full 50-observation A/B/A contract. The result was
`430.171/425.395/420.352 us`: `-1.11%` versus the first control but `+1.20%`
versus the second. The sign reversal identifies node drift rather than a
stable win, so R33 was fully reverted. Evidence is under
`iter56-flash-rank-major-pull-screen` and
`iter57-flash-m32-rank-major-formal` on the pod and local artifact root.

## Rejected experiments R34/R35: row-owned Flash MXFP4 decode

### Reason and direction

The accepted paired Flash decoder assigned two lanes to each logical row.
Consequently each K32 row scale was shuffled and converted to an E4M3 lookup
twice. R34 assigned one complete row to each lane, shared one lookup across
all four packed K32 words, and loaded the four words with LDS.128. The exact
M16 prototype passed eight-rank correctness at `0.000645` and an initial
20-observation screen at `430.990/414.820/439.349 us` (`-3.75%/-5.58%`).

Isolated M16 NCU confirmed that the intended work disappeared:

| NCU metric | R32 | R34 | change |
| --- | ---: | ---: | ---: |
| executed warp instructions | 73,216,037 | 68,002,559 | -7.12% |
| executed thread instructions | 2,288,537,091 | 2,121,797,338 | -7.29% |
| global-load sectors | 844,012 | 846,724 | +0.32% |
| local load/store sectors | 0/0 | 0/0 | unchanged |

The formal 50-observation M16 run did not reproduce the screen:
`417.055/428.269/434.773 us`, or `+2.69%/-1.50%`. A broad shape screen also
failed to find a stable selector:

| Flash point | first control us | R34 us | change | second control us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 392.785 | 405.258 | +3.18% | 406.164 | -0.22% |
| M32 | 454.404 | 448.362 | -1.33% | 434.601 | +3.17% |
| M64 | 460.600 | 465.494 | +1.06% | 456.441 | +1.98% |

R35 retained row ownership but split LDS.128 into two bank-validated LDS.64
loads. M16 remained correct at `0.000645`, but measured
`443.533/444.948/443.519 us`, a `+0.32%/+0.32%` regression. Fewer aggregate
instructions did not shorten the critical shared-memory/decode schedule, so
both row-owned variants were reverted. Evidence is under
`iter58-flash-m16-row-owned-decode` through
`iter61-flash-m16-row-owned-lds64`.

## R36: packed BF16 HFMA2 promotion for Flash M8/M64

### Reason and direction

The swap-AB mainloop already retained its cross-K persistent result as packed
BF16x2, but each promotion unpacked that result to FP32, issued four scalar
FMAs per pair, and packed it again. The regular-orientation fast-math path had
already validated BF16x2 fragment/scale promotion. R36 applies the same
operation to swap-AB: the two token scales and two WGMMA accumulator values
are packed, then accumulated directly with `__hfma2`.

The broad implementation passed eight-rank Flash M8/M16/M32/M64 correctness
at `0.000671/0.000654/0.000666/0.000660`. M16 was mixed at
`429.567/436.640/445.452 us` (`+1.65%/-1.98%`). Independent shape screening
selected only M8 and M64:

| Flash point | first control us | broad R36 us | change | second control us | reverse change | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M8 | 397.829 | 380.210 | -4.43% | 411.467 | -7.60% | accept |
| M32 | 445.930 | 438.096 | -1.76% | 424.845 | +3.12% | reject |
| M64 | 465.409 | 463.827 | -0.34% | 466.647 | -0.60% | accept |

The final compile-time selector is exact for routed Flash M8/M64. Flash
M16/M32, all Pro points, regular-orientation kernels, and strict math retain
their old scalar promotion.

### Formal performance

The formal R32-equivalent/R36/R32-equivalent run used ten warmups, 50
observations, 20 launches per observation, cold L2, and maximum-rank medians:

| Flash point | first control us | R36 us | change | second control us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 396.172 | 376.645 | -4.93% | 389.321 | -3.26% |
| M64 | 460.989 | 449.242 | -2.55% | 465.193 | -3.43% |

The retained two-point geometric mean improves by `-3.75%` versus the first
control and `-3.32%` versus the second. Raw formal logs are under
`iter64-flash-hfma2-formal`.

### NCU and NSYS analysis

An isolated one-rank, 32-expert Flash M64 profile preserves the production
expert/scheduler layout while removing distributed replay skew:

| NCU metric | R32-equivalent | R36 | change |
| --- | ---: | ---: | ---: |
| executed warp instructions | 94,250,307 | 87,158,677 | -7.52% |
| executed thread instructions | 2,961,050,641 | 2,733,535,692 | -7.68% |
| global-load sectors | 956,227 | 960,389 | +0.44% |
| global atomic sectors | 4,396 | 4,396 | unchanged |
| local load sectors | 0 | 180,224 | new spill traffic |
| local store sectors | 0 | 73,024 | new spill traffic |

Both cubins remain at 128 registers/thread. R36 wins the formal benchmark
despite the compiler-generated local traffic because the scalar
conversion/FMA body shrinks substantially; eliminating that spill is now the
immediate optimization target.

Low-perturbation eight-rank NSYS reports one 156-CTA fused kernel on both
sides and a single-launch duration of `713.052 us` for the control versus
`697.436 us` for R36 (`-2.19%`). Unlike several earlier one-launch traces,
this direction agrees with both formal controls. Complete profiler evidence
is under `iter65-flash-m64-hfma2-profiles` on the pod and local artifact root.

The terminal goal remains unmet. R36 removes 2.5-5% from two Flash latency
points but does not close the remaining PR383 gap. The next iteration should
retain the exact M8/M64 selector and shorten the HFMA2 temporary live ranges
or pair it with an epilogue layout that removes the newly measured local
traffic before expanding to another shape.

## Rejected experiment R37: packed BF16 Flash M64 swap epilogue

### Reason and direction

R36 introduced 16 bytes of compiler stack and measurable local traffic while
the final swap-AB epilogue still unpacked and repacked adjacent BF16 values.
R37 enabled the existing packed-BF16 swap epilogue for the exact Flash M64
specialization, with the hypothesis that a shorter packed output path would
reduce register pressure enough to remove the spill.

Eight-rank Flash M64 correctness passed at `0.000660`. The isolated NCU target
did not move: local load/store sectors remained `180,224/73,024`, the cubin
remained at `REG=128, STACK=16`, and executed instructions regressed by about
`1.2%`. A 20-observation screen measured
`488.911/462.501/476.673 us`, apparently winning both controls, but the formal
50-observation R36/R37/R36 run measured
`459.662/459.667/463.465 us`: `+0.001%` versus the first control and `-0.82%`
versus the second. It therefore failed the strict double-control criterion and
was fully reverted. Evidence is under
`iter66-flash-m64-packed-epilogue` and
`iter67-flash-m64-packed-epilogue-formal`.

## Rejected experiment R38: shorten HFMA2 scale live ranges

### Reason and direction

R38 moved the compile-time HFMA2 branch outside the weight-half loop and
constructed the packed scale directly from the two scale expressions. This
removed the named FP32 combined-scale temporaries from the HFMA2 source path
and duplicated only compile-time-eliminated loop bodies. The intended result
was to lower peak register pressure without changing arithmetic.

Eight-rank Flash M64 ring-wrap correctness passed at `0.000660`, but both
static and dynamic profiler evidence rejected the mechanism:

| isolated Flash M64 metric | R36 | R38 | change |
| --- | ---: | ---: | ---: |
| registers/thread | 128 | 128 | unchanged |
| stack bytes | 16 | 16 | unchanged |
| SASS `LDL` / `STL` instructions | 6 / 5 | 6 / 5 | unchanged |
| local load sectors | 180,224 | 180,224 | unchanged |
| local store sectors | 73,024 | 73,024 | unchanged |
| executed warp instructions | 87,158,677 | 87,143,231 | -0.018% |
| executed thread instructions | 2,733,535,692 | 2,733,161,967 | -0.014% |
| global-load sectors | 960,389 | 959,792 | -0.062% |

Because the resource target failed and the instruction difference was
negligible, R38 was rejected before a noisy distributed A/B/A run and fully
reverted. Evidence is under `iter68-flash-hfma2-live-range`. The next spill
experiment must be driven by the actual spilled values in SASS rather than by
CUDA source-level variable names.

## Rejected experiment R39: compact persistent pipeline state

### Reason and direction

R39 targeted the 16-byte Flash M8/M64 HFMA2 spill by shortening state that
crosses the persistent mainloop. The first prototype packed the circular
`stage_idx` and `phase` variables into one three-bit value and decoded them at
their use sites. Flash M64 ring-wrap correctness passed at `0.000660`, but the
isolated cubin remained at `REG=128, STACK=16` with six `LDL` and five `STL`
instructions. The packed state therefore did not contain the spilled values.

A second prototype also rematerialized the epilogue warp index from `%tid.x`
at its phase-local use sites. Correctness again passed at `0.000660`, but the
cubin worsened to `STACK=24` with eleven `LDL` and nine `STL` instructions.
Both variants were fully reverted without a distributed timing run. Static
evidence is under `iter69-packed-pipeline-state`.

## Rejected experiment R40: skip inactive swap-epilogue token chunks

### Reason and direction

The L1 swap-AB epilogue executed clamp, exponent, SwiGLU, and reduction work
for all compile-time token chunks, while inactive chunks merely loaded zero
top-k weights and suppressed the scratch store. R40 made the whole inactive
body warp-uniformly conditional. Eight-rank correctness passed six Flash
points from M8 through M1024 and the exact Pro M32 point; the latter reported
`0.000715` error.

The broad 20-observation screen did not support Flash:

| Flash point | R36 us | broad R40 us | change |
| --- | ---: | ---: | ---: |
| M16 | 428.556 | 431.738 | +0.74% |
| M32 | 432.121 | 431.568 | -0.13% |
| M64 | 449.785 | 455.793 | +1.34% |

The Pro screen selected only M32. Its broad candidate was `1151.0 us` versus
two 20-observation controls at `1162.5/1171.5 us`, so the compile-time selector
was narrowed to exactly `hidden=7168, M=32` and rerun under the formal
50-observation cold-L2 A/B/A contract. The result was:

| Pro point | first R36 us | exact R40 us | change | second R36 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M32 | 1174.0 | 1168.0 | -0.51% | 1155.0 | +1.13% |

The candidate failed the second control and was fully reverted. The wide
spread between identical controls again demonstrates why one short screen is
insufficient for sub-percent decisions. Evidence is under
`iter70-active-swap-chunks-screen`,
`iter71-pro-active-swap-chunks-screen`, and
`iter72-pro-m32-active-chunks-formal` on the pod and local artifact root.

## Rejected experiment R41: compile-time swap-bucket pruning

### Reason and direction

`kMaxSwapABTokens` proves that M8/M16/M32 specializations can never reach
larger swap-AB WGMMA buckets, but the source still instantiated the full
N8/N16/N32/N64 dispatch chain. R41 used the template bound to omit impossible
larger buckets and made M8 call N8 directly. All six affected eight-rank
Flash/Pro M8/M16/M32 production cases passed correctness.

The compiler response invalidated the intended mechanism. Official benchmark
cubins acquired large local frames despite having fewer bucket bodies:

| point | candidate stack bytes | candidate cubin bytes |
| --- | ---: | ---: |
| Flash M8/M16/M32 | 208 / 208 / 224 | 101208 / 128856 / 168792 |
| Pro M8/M16/M32 | 176 / 176 / 256 | 95064 / 124760 / 164696 |

The 20-observation cold-L2 screen regressed every point:

| point | R36 us | R41 us | change |
| --- | ---: | ---: | ---: |
| Flash M8 | 369.108 | 396.396 | +7.39% |
| Flash M16 | 418.312 | 440.080 | +5.20% |
| Flash M32 | 427.756 | 461.213 | +7.82% |
| Pro M8 | 862.452 | 913.188 | +5.88% |
| Pro M16 | 1072.500 | 1418.500 | +32.26% |
| Pro M32 | 1142.500 | 1607.000 | +40.66% |

The result is far outside measurement noise, so no second control was needed.
R41 was fully reverted. It establishes a PTXAS inlining/register-allocation
cliff: source-level dead-bucket pruning is unsafe here unless a future version
also proves zero stack in the final M8192-capacity cubin before benchmarking.
Evidence is under `iter73-compile-time-swap-buckets` on the pod and local
artifact root.

## Rejected experiment R42: interleave paired Flash decode chains

### Reason and direction

R21's paired Flash decoder shares one E4M3 lookup between two packed words but
serially reuses the same five PTX temporaries. R42 exposed both independent
PRMT/LOP3 chains together, following an instruction-level-parallelism pattern
that had been retained in an older SM90 NVFP4 tuning branch. The selector was
limited to Flash M16, the largest remaining PR383 gap.

Eight-rank correctness passed at `0.000645`. Both production and isolated
cubins stayed at `REG=114, STACK=0, LOCAL=0`. The 20-observation cold-L2
screen was mixed: R36/R42/R36 measured
`458.362/448.270/442.593 us`, or `-2.20%/+1.28%`.

Isolated one-rank, 32-expert NCU showed why the first apparent win was not
credible:

| metric | R36-equivalent | R42 | change |
| --- | ---: | ---: | ---: |
| kernel duration us | 355.232 | 366.430 | +3.15% |
| executed warp instructions | 73,216,037 | 73,206,099 | -0.014% |
| executed thread instructions | 2,288,537,091 | 2,288,373,109 | -0.007% |
| global-load sectors | 844,012 | 844,664 | +0.08% |
| local load/store sectors | 0/0 | 0/0 | unchanged |

PTXAS already scheduled both source forms to effectively identical dynamic
work, and the candidate failed the reverse-order distributed control. R42 was
fully reverted without a formal 50-observation run. Evidence is under
`iter74-flash-m16-decode-ilp` on the pod and local artifact root.

## Rejected experiment R43: one-CTA N256 Flash M16

### Reason and direction

R16 had already shown that replacing the two resident M64N128 CTAs with one
CTA and two N64 math warpgroups loses throughput. R43 tested the stronger
variant suggested by the retained older SM90 NVFP4 work: one 384-thread CTA
per physical H20 SM, an M64N256 task, and two math warpgroups that each own a
complete N128 tile. This preserves the accepted kernel's aggregate N256 work
per physical SM while sharing the A tile, scheduler mailbox, and producer
front end. The selector was exact for routed DSV4 Flash M16; all other
authoritative points retained R36.

The implementation partitioned packed/expanded B, weight SF, L1/L2 C/D
scratch, per-WG barriers, activation SF stores, TMA stores, and NVLink scatter
addresses between the two warpgroups. The host selected 78 workers, a BN256
TMA box, 384 threads, and one-block residency. Eight-rank forced-ring-wrap
correctness passed with the unchanged `0.000645` error. The resulting cubin
used 105 registers/thread with no compiler stack or local allocation, versus
114 registers/thread and no stack for the R36 M16 cubin.

### Performance and profiler result

The 20-observation, ten-warmup, 20-launch cold-L2 screen measured
`451.069 us` for the first R36 control and `1004.500 us` for R43, a
`+122.69%` regression. This is far outside run-to-run noise, so the second
control was stopped before observations and the experiment was reverted.

Profiler evidence isolates a concurrency failure rather than extra dynamic
work. On the same isolated one-rank, 32-expert Flash M16 case, source-counter
NCU reported 73,604,735 executed warp instructions for R43 versus 73,216,037
for the preceding exact R36 control (`+0.53%`), while profiled duration grew
from `355.232 us` to `1056.320 us` (`+197.36%`). A fresh matched NSYS pair
measured `358.398 us` for R36 and `1054.971 us` for R43 (`+194.36%`). Thus two
math warpgroups inside one CTA do not reproduce the tensor-core/front-end
overlap of two independently resident CTAs on H20, even when total WGMMA work
and N coverage are held constant.

The result rules out further one-CTA/two-math-WG widening for this kernel.
Future small-M work should preserve two independent resident CTAs and reduce
per-CTA fixed work, or move work onto currently underused producer/dispatch
warps without reducing the number of resident WGMMA owners. Complete evidence
is under `iter75-wide-flash-m16` on the pod and local artifact root.

## Rejected experiment R44: producer-warp Flash M16 decode

### Reason and direction

R44 preserved R36's 156 independent M64N128 CTAs and moved exact Flash M16
packed-B expansion from the 128-thread math warpgroup to the two 32-thread TMA
producer warps. Each producer owned 64 B rows and published completion through
a per-stage barrier. The intent was to overlap stage N+1 decode with stage N
WGMMA and promotion while keeping two resident WGMMA owners per H20 SM.

The fully unrolled decoder passed eight-rank forced-ring-wrap correctness but
compiled at `REG=128, STACK=136`. A lower-live-range single-word version was
worse at `STACK=144`. Serializing its three small source loops finally produced
an official eight-rank cubin at `REG=128, STACK=0, LOCAL=0`; correctness still
passed at `0.000645`. Only this zero-stack form entered distributed timing.

### Performance and profiler result

The 20-observation, ten-warmup, 20-launch cold-L2 A/B/A screen was decisive:

| Flash point | first R36 us | R44 us | change | second R36 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M16 | 439.522 | 1519.000 | +245.6% | 428.021 | +254.9% |

The official cubin's zero stack rules out local-memory spill as the cause of
this regression. An isolated one-rank NCU comparison also showed that the
looped producer implementation nearly doubled dynamic work:

| isolated Flash M16 metric | R36 | R44 | change |
| --- | ---: | ---: | ---: |
| duration | 367.616 us | 1.738944 ms | +373.0% |
| executed warp instructions | 73,208,071 | 141,328,773 | +93.0% |
| executed thread instructions | 2,288,416,839 | 4,452,359,001 | +94.6% |
| global-load sectors | 844,605 | 944,829 | +11.9% |
| local load/store sectors | 0 / 0 | 13,184 / 13,696 | isolated candidate only |

The one-rank specialization did spill, so its absolute NCU duration is not
used as the acceptance result. A separate eight-rank NSYS capture exercised
the official zero-stack specialization and measured the fastest captured
kernel instance at `473.631 us` for R36 versus `1786.152 us` for R44
(`+277.1%`); slower child-process instances were profiler-serialized.

Moving expansion to two producer warps therefore traded four-way math-WG
decode parallelism for two serial loop nests and placed decode directly in
front of the producer pipeline's next TMA issue. The intended overlap did not
materialize. R44 was fully reverted. Future producer assistance must split a
small, independently useful fraction of decode without serializing the full
tile or delaying producer advance. Complete evidence is under
`iter76-producer-decode-flash-m16` on the pod and local artifact root.

## Rejected experiment R45: direct mapped Flash M16 scale loads

### Reason and direction

The R36 full-section NCU baseline for isolated Flash M16 is spill-free but
reports `61.09%` scheduler cycles with no eligible warp. R45 targeted one
dependency at the head of each paired packed-B decode. R36 loads one SFB word
per lane and then uses two dependent `SHFL` instructions to map the 32 source
rows onto the two row groups. The exact Flash M16 specialization instead had
both lanes for a decoded row load that row's SFB word directly, relying on
shared-memory multicast. No CTA, pipeline, barrier, packed-weight, or numeric
contract changed.

Eight-rank forced-ring-wrap correctness passed at `0.000645`. The official
benchmark cubin remained spill-free, but grew from 114 to 115 registers per
thread (`STACK=0, LOCAL=0` in both cases).

### Performance and profiler result

The 20-observation, ten-warmup, 20-launch cold-L2 A/B/A screen rejected the
change in both launch orders:

| Flash point | first R36 us | R45 us | change | second R36 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M16 | 444.694 | 459.855 | +3.41% | 426.161 | +7.91% |

Matched isolated source-counter NCU shows that the source-level instruction
trade did not survive code generation as intended:

| isolated Flash M16 metric | R36 | R45 | change |
| --- | ---: | ---: | ---: |
| duration us | 370.912 | 590.176 | +59.12% |
| executed warp instructions | 73,213,070 | 73,932,187 | +0.98% |
| executed thread instructions | 2,288,544,046 | 2,310,386,924 | +0.95% |
| shared-load bank conflicts | 3,052,993 | 3,050,758 | -0.07% |
| global-load sectors | 844,360 | 865,705 | +2.53% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

Matched one-rank NSYS measures `357.534 us` for R36 and `573.949 us` for R45
(`+60.53%`). The isolated topology magnifies the latency cost and is not the
acceptance score, but NCU and NSYS agree that multicast did not shorten the
critical chain. Direct row addressing added instructions and one register,
while the shared-bank-conflict count was already effectively unchanged.

R45 was fully reverted. Further decoder work must remove final SASS from the
E2M1-to-E4M3 conversion itself or expose independent packed-word work without
adding address generation. Complete evidence is under
`iter77-r36-pr383-flash-m16` and `iter78-direct-scale-flash-m16` on the pod and
local artifact root.

## Rejected experiment R46: direct Flash M16 scale-byte loads

### Reason and direction

R46 strengthened R45's scale-load experiment. Instead of directly loading a
32-bit row word and retaining the four exponent-extraction PRMTs, exact Flash
M16 issued `LDS.U8` for the active `(row, K32)` scale byte. This was intended
to remove the stage-prologue word load, two row-mapping shuffles, and four
exponent PRMTs. The packed-weight decoder, CTA topology, barriers, and numeric
path were unchanged.

Eight-rank forced-ring-wrap correctness passed at `0.000645`. The official
benchmark cubin remained `STACK=0, LOCAL=0`, but again rose from 114 to 115
registers per thread. The 20-observation cold-L2 A/B/A screen rejected it:

| Flash point | first R36 us | R46 us | change | second R36 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M16 | 435.419 | 451.739 | +3.75% | 431.857 | +4.60% |

Static SASS proves that the desired PRMT and shuffle removal occurred, but
also exposes the larger address-generation replacement:

| static opcode | R36 | R46 | change |
| --- | ---: | ---: | ---: |
| `PRMT` | 761 | 632 | -129 |
| `SHFL.IDX` | 45 | 10 | -35 |
| `IMAD` | 365 | 303 | -62 |
| `LDS.U8` | 0 | 128 | +128 |
| `IADD3` | 237 | 296 | +59 |
| `IMAD.IADD` | 34 | 162 | +128 |

Matched isolated NCU accordingly reports more, not less, dynamic work:

| isolated Flash M16 metric | R36 | R46 | change |
| --- | ---: | ---: | ---: |
| duration us | 370.912 | 586.592 | +58.15% |
| executed warp instructions | 73,213,070 | 74,023,025 | +1.11% |
| executed thread instructions | 2,288,544,046 | 2,312,912,652 | +1.06% |
| shared-load bank conflicts | 3,052,993 | 3,051,233 | -0.06% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

One-rank NSYS measures `357.534 us` for R36 versus `569.597 us` for R46
(`+59.31%`). R46 was fully reverted. It shows that byte-granular scale loads
can only be viable if one shared address is formed per decoded row and the
four unrolled byte loads use immediate offsets; recomputing a generic address
at every K32 loses more instructions than the PRMT path removes. Complete
evidence is under `iter79-direct-scale-byte-flash-m16` on the pod and local
artifact root.

## Rejected experiment R47: immediate-offset Flash M16 scale-byte loads

### Reason and direction

R47 tested the remaining viable form of R46's byte-load idea. Exact Flash M16
formed the shared-memory base address once per decoded row, then selected four
inline `LDS.U8 [base + immediate]` instructions for the unrolled K32 groups.
This was intended to retain the eliminated scale-word shuffle and exponent
PRMTs without rebuilding a generic shared address at every K32. The packed-B
layout, CTA topology, barriers, and numeric path were unchanged.

Eight-rank forced-ring-wrap correctness passed at `0.000645`. The test cubin
remained spill-free but used 115 registers per thread versus R36's 114. Static
SASS shows that immediate offsets recovered only 16 `IADD3` instructions from
R46 and did not remove the 128 added `IMAD.IADD` instructions:

| static opcode | R36 | R47 | change |
| --- | ---: | ---: | ---: |
| `PRMT` | 761 | 632 | -129 |
| `SHFL.IDX` | 45 | 10 | -35 |
| `IMAD` | 365 | 303 | -62 |
| `LDS.U8` | 0 | 128 | +128 |
| `IADD3` | 237 | 280 | +43 |
| `IMAD.IADD` | 34 | 162 | +128 |

### Profiler gate result

Matched isolated source-counter NCU confirms that R47 still executes more
work than R36:

| isolated Flash M16 metric | R36 | R47 | change |
| --- | ---: | ---: | ---: |
| duration us | 370.912 | 585.344 | +57.81% |
| executed warp instructions | 73,213,070 | 74,013,852 | +1.09% |
| executed thread instructions | 2,288,544,046 | 2,312,801,743 | +1.06% |
| shared-load bank conflicts | 3,052,993 | 3,050,939 | -0.07% |
| global-load sectors | 844,360 | 865,055 | +2.45% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

R47 failed the pre-benchmark requirement that both static and dynamic
instruction counts beat R36, so it was deliberately not admitted to the
distributed cold-L2 A/B/A or NSYS stages. It was fully reverted. Together,
R45-R47 rule out direct shared-memory scale addressing for this decoder: a
future conversion experiment must reduce the packed E2M1-to-E4M3 bit path
without adding per-row address or load instructions. Complete evidence is
under `iter80-scale-byte-immediate-flash-m16` on the pod and local artifact
root.

## R48: packed BF16 HFMA2 promotion for Pro M8/M64/M128

### Reason and direction

R36 proved that swap-AB promotion can accumulate the packed BF16 persistent
state with two `HFMA2` instructions instead of unpacking two BF16x2 values,
issuing four scalar FP32 FMAs, and packing them again. Its final selector was
restricted to Flash M8/M64; the equivalent Pro swap-AB path still used the
scalar sequence. R48 extends the packed promotion only to routed Pro buckets
with `kMaxSwapABTokens` 8 or 64. Those buckets correspond to the authoritative
Pro M8, M64, and M128 points. Pro M16/M32 keep their existing packed epilogue
and scalar promotion, while strict math and every regular-orientation kernel
remain unchanged.

The original H20 node became occupied by an unrelated long-lived inference
process, so R48 used the idle eight-H20 node hosting
`molou-deepgemm-sm90-h20-2050-0810`. Every comparison below is matched on that
node with separate control/candidate JIT caches. No cross-node timing is used.

### Correctness and resources

All six Pro production scenarios pass on eight ranks. The changed points are:

| Pro point | calc diff | ring-wrap check |
| --- | ---: | --- |
| M8 | 0.000734 | not required |
| M64 | 0.000720 | pass |
| M128 | 0.000709 | pass |

The generated M8 cubin uses 107 registers/thread; M64 and M128 use 128. All
three report `STACK=0`, `LOCAL=0`, and 1024 bytes of static shared memory.

### Screening and formal performance

The 20-observation, ten-warmup, 20-launch cold-L2 screen selected all three
points against both R36 controls:

| Pro point | first R36 us | R48 us | change | second R36 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 841.450 | 836.696 | -0.57% | 857.706 | -2.45% |
| M64 | 1222.0 | 1199.0 | -1.88% | 1219.0 | -1.64% |
| M128 | 1413.0 | 1325.0 | -6.23% | 1406.0 | -5.76% |

The formal run raises each point to 50 observations while retaining ten
warmups, 20 launches per observation, cold L2, and maximum-rank medians:

| Pro point | first R36 us | R48 us | change | second R36 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 843.399 | 833.309 | -1.20% | 859.197 | -3.01% |
| M64 | 1204.5 | 1180.0 | -2.03% | 1220.5 | -3.32% |
| M128 | 1396.5 | 1308.5 | -6.30% | 1406.5 | -6.97% |

The three-point geometric mean improves by `-3.20%` against the first control
and `-4.45%` against the second. Complete screen and formal logs are under
`iter81-pro-hfma2-screen` and `iter82-pro-hfma2-formal` on the pod and local
artifact root.

### NCU and NSYS attribution

Matched one-rank, 48-expert Pro M128 profiling preserves the changed
specialization while removing distributed replay skew:

| NCU metric | R36 | R48 | change |
| --- | ---: | ---: | ---: |
| duration us | 1425.568 | 1351.392 | -5.20% |
| executed warp instructions | 400,954,627 | 361,698,793 | -9.79% |
| executed thread instructions | 12,647,261,800 | 11,390,901,496 | -9.93% |
| shared-load bank conflicts | 12,425,066 | 12,431,216 | +0.05% |
| global-load sectors | 24,957,733 | 24,957,489 | unchanged |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

Low-perturbation NSYS reports one 156-CTA fused kernel on both sides and a
single-launch duration of `1322.750 us` for R36 versus `1244.382 us` for R48
(`-5.93%`). The profiler evidence isolates the intended arithmetic reduction:
roughly ten percent fewer executed instructions with unchanged global and
shared traffic and no spill cost. Complete reports are under
`iter83-pro-m128-hfma2-profiles` on the pod and local artifact root.

R48 is retained. It materially closes the largest small-Pro gap without
changing topology, communication, memory traffic, or the numerical contract.
The terminal goal remains unmet; the next full candidate/PR383 matrix must
measure the updated Flash and Pro gaps before selecting the next target.

## R48 same-node full matrix against PR383

The authoritative post-R48 comparison ran both implementations on the same
eight-H20 pod, with cold L2, 20 launches per observation, maximum-rank
medians, 50 observations for M <= 128, and three observations for M >= 256.
The candidate used one explicit warmup. PR383 used its native two-phase FP8
driver and reports the sum of L1 and L2. Both logs contain all 22 requested
DSV4 Flash and Pro summaries.

| model | M | R48 MXFP4 us | PR383 us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 386.983 | 302.332 | +28.00% |
| Flash | 16 | 422.599 | 312.559 | +35.21% |
| Flash | 32 | 435.637 | 338.162 | +28.83% |
| Flash | 64 | 457.105 | 366.440 | +24.74% |
| Flash | 128 | 487.960 | 440.793 | +10.70% |
| Flash | 256 | 522.551 | 508.985 | +2.67% |
| Flash | 512 | 912.641 | 909.921 | +0.30% |
| Flash | 1024 | 1507.000 | 1544.968 | -2.46% |
| Flash | 2048 | 2765.000 | 2758.600 | +0.23% |
| Flash | 4096 | 5221.000 | 5118.000 | +2.01% |
| Flash | 8192 | 10013.000 | 9837.000 | +1.79% |
| Pro | 8 | 842.128 | 712.988 | +18.11% |
| Pro | 16 | 1067.500 | 1014.713 | +5.20% |
| Pro | 32 | 1136.000 | 1103.769 | +2.92% |
| Pro | 64 | 1198.000 | 1157.868 | +3.47% |
| Pro | 128 | 1312.000 | 1292.149 | +1.54% |
| Pro | 256 | 1627.000 | 1643.809 | -1.02% |
| Pro | 512 | 2537.000 | 2434.415 | +4.21% |
| Pro | 1024 | 3919.000 | 4029.000 | -2.73% |
| Pro | 2048 | 6933.000 | 7019.000 | -1.23% |
| Pro | 4096 | 13118.000 | 12965.000 | +1.18% |
| Pro | 8192 | 25499.000 | 25145.000 | +1.41% |

The same-epoch geometric gaps are:

- all 22 points: `+6.96%`;
- M <= 128: `+15.26%`;
- M >= 256: `+0.51%`;
- all Flash points: `+11.21%`;
- Flash small-M: `+25.22%`;
- Flash large-M: `+0.74%`;
- all Pro points: `+2.88%`;
- Pro small-M: `+6.08%`;
- Pro large-M: `+0.28%`.

Compared with R32's fresh matrix, the all-point gap falls from `+8.61%` to
`+6.96%`, and the Pro small-M gap falls from `+12.02%` to `+6.08%`. In
particular, Pro M128 falls from `+16.38%` to `+1.54%`, validating R48's
instruction-count mechanism. The terminal goal is still unmet: Flash M8-M64
now dominate the aggregate loss, with Flash M16 the worst point at `+35.21%`.
The next iteration must preserve the two-independent-CTA topology proven by
R43 and reduce the fixed packed-B decode/synchronization critical path rather
than revisit direct scale loads ruled out by R45-R47. Complete logs are under
`iter84-r48-pr383-full-matrix` on the pod and local artifact root.

## R48 versus PR383 matched Flash M16 profiler decomposition

The post-matrix profiler run uses one H20, one rank, and 32 experts for both
implementations. This preserves one production expert shard while avoiding
distributed replay skew. NCU uses the same input seed and M16 shape; NSYS
captures exactly the profiler-delimited production launch. These timings are
diagnostic and do not replace the eight-rank score above.

NSYS attributes the gap inside the GPU kernels:

| NSYS kernel | duration us |
| --- | ---: |
| R48 fused MXFP4 | 358.368 |
| PR383 FP8 L1 | 168.960 |
| PR383 FP8 L2 | 94.080 |
| PR383 L1 + L2 | 263.040 |

The fused candidate is `+36.24%` slower than the two PR383 kernels combined;
its approximately 95.3-us excess is therefore not a second-launch artifact.
Targeted NCU reproduces the same direction at 357.056 versus 266.656 us and
identifies the additional work:

| additive NCU metric | R48 fused | PR383 L1 + L2 | excess |
| --- | ---: | ---: | ---: |
| executed warp instructions | 73,211,822 | 41,097,573 | +78.14% |
| executed thread instructions | 2,288,346,146 | 1,247,705,433 | +83.40% |
| shared-load bank conflicts | 3,052,150 | 71,816 | +4149.96% |
| shared-store bank conflicts | 1,883,753 | 75,861 | +2383.16% |
| global-load sectors | 845,089 | 672,643 | +25.64% |
| global-store sectors | 29,369 | 29,135 | +0.80% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The candidate has 114 registers/thread, 110.816 KiB dynamic shared memory per
CTA, 3.106 resident warps stalled on barriers per active issue cycle, and
4.084 stalled on long scoreboards. PR383 L1/L2 use 168 registers/thread and
230.672 KiB each; their respective barrier ratios are 1.296/3.697 and long-
scoreboard ratios are 3.170/2.693. Ratios are not additive across sequential
kernels, but they confirm that the fused kernel's 16 active warps/SM do not
hide its packed-B expansion dependency chain.

The next experiment therefore targets the B128 expanded-tile publication,
not launch topology, scale addressing, or local spill. Complete NCU reports,
raw CSV, NSYS reports, and kernel summaries are under
`iter85-r48-pr383-flash-m16-profiles` on the pod and local artifact root.

## Rejected experiment R49: split Flash M16 expanded-B stores

### Reason and direction

R49 tested whether each lane's wide `STS.128` publication caused the measured
expanded-B replay. Exact Flash M16 retained the same decoder, B128 byte layout,
addresses, CTA topology, and barriers, but split every 16-byte store into two
adjacent `STS.64` instructions. Eight-rank forced-ring-wrap correctness passed
at `0.000645`; the official cubin remained spill-free at 114 registers/thread.

Static SASS changed from 148 `STS.128` and two `STS.64` instructions to 20 and
258 respectively. Matched one-rank/32-expert targeted NCU rejected the
mechanism before distributed timing:

| Flash M16 NCU metric | R48 | R49 | change |
| --- | ---: | ---: | ---: |
| duration us | 357.056 | 363.584 | +1.83% |
| executed warp instructions | 73,211,822 | 74,745,571 | +2.10% |
| executed thread instructions | 2,288,346,146 | 2,337,318,609 | +2.14% |
| shared-load bank conflicts | 3,052,150 | 3,052,505 | +0.01% |
| shared-store bank conflicts | 1,883,753 | 8,443,400 | +348.22% |
| global-load sectors | 845,089 | 847,211 | +0.25% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The wide store was not the source of replay; splitting it exposes more shared
transactions and multiplies conflicts. R49 was fully reverted. Future layout
work must change the lane-to-bank mapping or use a collective matrix store,
not merely narrow the existing per-lane transaction. Complete evidence is
under `iter86-split-sts64-flash-m16` on the pod and local artifact root.

## Rejected experiment R50: B32 packed-B TMA swizzle for Flash M16

### Reason and direction

R50 targeted the dominant 3.05-million shared-load conflicts in the matched
Flash M16 profile. The exact M16 specialization changed only the packed-B TMA
swizzle from B64 to B32 and changed both packed-byte decoder address mappings
from `Swizzle<2, 4, 3>` to the corresponding `Swizzle<1, 4, 3>`. Expanded B,
WGMMA, CTA topology, barriers, and numerical operations were unchanged.

The mandatory eight-rank `production.flash_m16` correctness gate failed on
its first scenario with `cudaErrorIllegalAddress`. The B32 descriptor/layout
combination is therefore not legal for this packed tile as currently shaped;
no profiler or timing result can be accepted. R50 was fully reverted before
any further experiment. The complete failure log is under
`iter87-b32-packed-flash-m16` on the pod and local artifact root.

## R51: vectorize Flash M16 weight-scale staging

### Reason and direction

Flash preprocessing already stores each K128 group as 128 contiguous scale
words across N, but the B producer still issued four separate scalar LDG/STS
rounds to stage those words. R51 is exact for routed Flash M16 and assigns
four adjacent rows to each lane, replacing the four scalar transfers with one
16-byte global load and one 16-byte shared store. It does not change the
processed-weight format, packed-B TMA, decoder, WGMMA, synchronization, CTA
topology, or numerical operations.

The production eight-rank forced-ring-wrap correctness case passes at
`0.000645`. The official cubin remains spill-free at 114 registers/thread.
Matched one-rank/32-expert NCU shows that the wider producer transaction
substantially improves the generated schedule rather than merely reducing
source statements:

| Flash M16 NCU metric | R48 | R51 | change |
| --- | ---: | ---: | ---: |
| duration us | 357.056 | 304.576 | -14.70% |
| executed warp instructions | 73,211,822 | 71,983,933 | -1.68% |
| executed thread instructions | 2,288,346,146 | 2,249,823,179 | -1.68% |
| shared-load bank conflicts | 3,052,150 | 3,055,680 | +0.12% |
| shared-store bank conflicts | 1,883,753 | 876,071 | -53.49% |
| global-load sectors | 845,089 | 835,314 | -1.16% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The 20-observation cold-L2 screen measured R48/R51/R48 at
`456.802/451.434/466.272 us`, or `-1.18%/-3.18%`. The formal run raises each
side to 50 observations, ten warmups, and 20 launches per observation:

| Flash point | first R48 us | R51 us | change | second R48 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M16 | 432.983 | 364.307 | -15.86% | 429.151 | -15.11% |

The run contained transient multi-millisecond outliers on both implementations,
but the maximum-rank median selects the same large win against both controls.
Low-perturbation one-rank NSYS independently measures `356.129 us` for R48
and `282.273 us` for R51 (`-20.74%`). The agreement among both launch orders,
NCU, and NSYS makes the mechanism strong enough to retain. Complete artifacts
are under `iter88-vector-sfb-flash-m16` through `iter91-vector-sfb-nsys` on
the pod and local artifact root.

## R52: vectorize weight-scale staging for all Flash small-M buckets

### Reason and direction

R51 selected the wider scale transaction only for Flash M16 even though every
Flash swap-AB specialization has the same coalesced 4096-wide preprocessed
scale layout. R52 removes the exact-M16 guard and uses the same one-`uint4`
LDG plus one `STS.128` mapping for M8, M16, M32, and M64. Pro remains on its
strided scalar path because hidden 7168 does not satisfy the coalesced-layout
predicate; Flash M128 and larger remain outside the swap-AB selector.

The eight-rank production Flash correctness suite passed all six configured
shapes (M8, M16, M32, M64, M128, and M1024). Maximum normalized differences
were between `0.000645` and `0.000671`, including the unchanged M128/M1024
controls. The complete log is under `iter92-vector-sfb-flash-small`.

The initial 20-observation cold-L2 A/B/A screen retained every newly affected
point:

| Flash point | first R36 us | R52 us | change | second R36 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 388.768 | 321.328 | -17.35% | 395.989 | -18.85% |
| M32 | 426.876 | 390.405 | -8.54% | 424.663 | -8.06% |
| M64 | 458.963 | 411.167 | -10.41% | 443.947 | -7.38% |

The formal run used the required cold-L2 contract: one warmup, 50
observations, and 20 launches per observation, with the maximum-rank median.
All three wins reproduced against both surrounding controls:

| Flash point | first R36 us | R52 us | change | second R36 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 386.710 | 340.652 | -11.91% | 390.076 | -12.67% |
| M32 | 419.924 | 383.960 | -8.56% | 431.841 | -11.09% |
| M64 | 451.793 | 385.859 | -14.59% | 458.964 | -15.93% |

Together with R51's profiler attribution, the adjacent-shape consistency
confirms that the old four-round scale publication was a common Flash
swap-AB bottleneck rather than an M16-only compiler accident. Complete screen
and formal logs are under `iter93-vector-sfb-adjacent-screen` and
`iter94-vector-sfb-adjacent-formal` on the pod and local artifact root.

## R52 same-node full matrix against PR383

The post-commit matrix reruns both native drivers on the same H20 node under
the authoritative contract. M8 through M128 use 50 observations; M256 through
M8192 use three. Every observation contains 20 launches, reports the
maximum-rank median, and starts from cold L2. PR383 time is its measured FP8
L1 plus L2 duration.

| model | M | R52 MXFP4 us | PR383 FP8 us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 319.272 | 308.524 | +3.48% |
| Flash | 16 | 352.478 | 313.487 | +12.44% |
| Flash | 32 | 357.011 | 333.222 | +7.14% |
| Flash | 64 | 390.460 | 369.663 | +5.63% |
| Flash | 128 | 484.034 | 427.491 | +13.23% |
| Flash | 256 | 501.068 | 525.578 | -4.66% |
| Flash | 512 | 889.924 | 919.206 | -3.19% |
| Flash | 1024 | 1511.000 | 1508.489 | +0.17% |
| Flash | 2048 | 2824.000 | 2728.568 | +3.50% |
| Flash | 4096 | 5209.000 | 5055.000 | +3.05% |
| Flash | 8192 | 10025.000 | 9830.000 | +1.98% |
| Pro | 8 | 819.119 | 705.827 | +16.05% |
| Pro | 16 | 1071.500 | 1004.129 | +6.71% |
| Pro | 32 | 1133.000 | 1097.008 | +3.28% |
| Pro | 64 | 1184.500 | 1153.941 | +2.65% |
| Pro | 128 | 1313.500 | 1284.029 | +2.30% |
| Pro | 256 | 1624.000 | 1624.990 | -0.06% |
| Pro | 512 | 2545.000 | 2420.777 | +5.13% |
| Pro | 1024 | 3933.000 | 4020.000 | -2.16% |
| Pro | 2048 | 6968.000 | 7055.000 | -1.23% |
| Pro | 4096 | 13089.000 | 12906.000 | +1.42% |
| Pro | 8192 | 25337.000 | 25064.000 | +1.09% |

Geometric gaps are `+3.420%` over all 22 points, `+7.190%` over the ten
small-M points, and `+0.381%` over the 12 large-M points. Flash is `+3.750%`
overall (`+8.315%` small, `+0.093%` large); Pro is `+3.092%` overall
(`+6.076%` small, `+0.670%` large). Relative to R48, R52 cuts the all-point
gap from `+6.963%` to `+3.420%` and the Flash small-M gap from `+25.220%` to
`+8.315%`. The next scale-layout experiment therefore targets Pro M8-M128,
where natural row-major scale storage still prevents the successful vector
publication path. Complete logs are under `iter95-r52-pr383-full-matrix` on
the pod and local artifact root.

## R53: coalesce Pro weight-scale layout and vectorize publication

### Reason and direction

R51/R52 proved that four scalar scale transfers were a common small-M
bottleneck, but the preprocessing contract limited K128-by-N coalescing to
hidden sizes at or below 4096. R53 raises that limit to 8192, covering DSV4
Pro hidden 7168. Its transformed scale payload is now physically
`[E,K/128,N,4]`, so the existing small-M selector automatically replaces four
strided scalar LDG/STS rounds with one `uint4` LDG and one `STS.128` per lane.
The logical tensor shape, scale values, packed weights, decoder, WGMMA,
scheduler, and epilogue are unchanged. Hidden sizes above 8192 retain the
natural layout.

All 14 CPU preprocessing contract tests pass. The eight-rank production Pro
correctness suite passes M8, M16, M32, M64, M128, and M256, including forced
ring wrap, with maximum normalized differences from `0.000704` to `0.000716`.
The official one-rank M16 profiler cubins remain identical in resources at
110 registers/thread, zero stack, zero local storage, and 1024 bytes static
shared memory. The correctness log is under
`iter96-coalesced-pro-sf-correctness`.

The 20-observation cold-L2 screen retained every affected production point:

| Pro point | first R52 us | R53 us | change | second R52 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 867.307 | 842.593 | -2.85% | 877.141 | -3.94% |
| M16 | 1099.500 | 1031.500 | -6.18% | 1102.000 | -6.40% |
| M32 | 1138.500 | 1115.000 | -2.06% | 1135.000 | -1.76% |
| M64 | 1189.500 | 1144.500 | -3.78% | 1198.000 | -4.47% |
| M128 | 1334.000 | 1310.500 | -1.76% | 1323.000 | -0.94% |

The formal A/B/A run uses one warmup, 50 observations, 20 launches per
observation, cold L2, and the maximum-rank median:

| Pro point | first R52 us | R53 us | change | second R52 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| M8 | 839.680 | 808.904 | -3.67% | 835.956 | -3.24% |
| M16 | 1067.500 | 1027.000 | -3.79% | 1067.000 | -3.75% |
| M32 | 1134.000 | 1088.000 | -4.06% | 1130.500 | -3.76% |
| M64 | 1179.500 | 1132.500 | -3.98% | 1178.500 | -3.90% |
| M128 | 1317.000 | 1288.000 | -2.20% | 1314.000 | -1.98% |

Matched one-rank/48-expert Pro M16 profiling attributes the gain to the scale
layout rather than launch noise:

| NCU metric | R52 | R53 | change |
| --- | ---: | ---: | ---: |
| duration us | 1007.584 | 972.064 | -3.53% |
| executed warp instructions | 264,581,994 | 262,632,129 | -0.74% |
| executed thread instructions | 8,306,056,968 | 8,225,890,065 | -0.97% |
| shared-load bank conflicts | 10,858,497 | 10,857,608 | -0.01% |
| shared-store bank conflicts | 4,392,846 | 3,880,380 | -11.67% |
| global-load sectors | 21,768,527 | 2,804,888 | -87.12% |
| global-store sectors | 51,625 | 51,625 | unchanged |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| barrier-stall ratio | 2.348 | 2.266 | -3.50% |
| long-scoreboard ratio | 1.951 | 1.913 | -1.92% |

NSYS independently measures the same launch at 929.890 us for R52 and
894.466 us for R53 (`-3.81%`). Complete screening, formal, NCU, NSYS, and
resource evidence is under `iter97-coalesced-pro-sf-screen` through
`iter99-coalesced-pro-sf-profiles` on the pod and local artifact root.

### R53 authoritative DSV4 matrix versus PR383

The fresh same-node comparison uses the required 8-rank DSV4 Flash and Pro
matrix, one warmup, 50 small-M observations, three large-M observations, 20
launches per observation, cold L2, and the maximum-rank median. PR383 uses its
native two-phase FP8 driver and the reported time is L1 plus L2.

| model | M | R53 MXFP4 us | PR383 FP8 us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 316.942 | 304.397 | +4.12% |
| Flash | 16 | 361.037 | 314.023 | +14.97% |
| Flash | 32 | 365.460 | 331.830 | +10.14% |
| Flash | 64 | 388.386 | 368.547 | +5.38% |
| Flash | 128 | 483.836 | 435.046 | +11.22% |
| Flash | 256 | 516.622 | 488.635 | +5.73% |
| Flash | 512 | 892.305 | 918.021 | -2.80% |
| Flash | 1024 | 1500.000 | 1550.606 | -3.26% |
| Flash | 2048 | 2770.000 | 2703.962 | +2.44% |
| Flash | 4096 | 5177.000 | 5054.000 | +2.43% |
| Flash | 8192 | 9999.000 | 9847.000 | +1.54% |
| Pro | 8 | 786.810 | 718.405 | +9.52% |
| Pro | 16 | 1023.000 | 1001.898 | +2.11% |
| Pro | 32 | 1084.500 | 1100.803 | -1.48% |
| Pro | 64 | 1131.000 | 1148.729 | -1.54% |
| Pro | 128 | 1285.500 | 1280.647 | +0.38% |
| Pro | 256 | 1694.000 | 1638.885 | +3.36% |
| Pro | 512 | 2592.000 | 2412.803 | +7.43% |
| Pro | 1024 | 3997.000 | 4052.000 | -1.36% |
| Pro | 2048 | 7049.000 | 7029.000 | +0.28% |
| Pro | 4096 | 13229.000 | 12938.000 | +2.25% |
| Pro | 8192 | 25626.000 | 25019.000 | +2.43% |

Geometric gaps are `+3.317%` over all 22 points, `+5.340%` over the ten
small-M points, and `+1.661%` over the 12 large-M points. Flash is `+4.581%`
overall (`+9.093%` small, `+0.965%` large); Pro is `+2.068%` overall
(`+1.717%` small, `+2.363%` large). The coalesced Pro scale change therefore
reduces the previous Pro small-M geometric gap from `+6.076%` to `+1.717%`,
with M32 and M64 now faster than PR383. The dominant next targets are Flash
M16/M32/M128 and Pro M8. Complete logs, including the harmless failed nested
launcher attempt that exited before measurement, are under
`iter100-r53-pr383-full-matrix` on the pod and local artifact root.

## R54: vectorize coalesced scale publication for regular buckets

### Reason and direction

R51/R52 restricted the 128-word vector scale transfer to swap-AB small-M
buckets even though the R53 preprocessing contract makes the same contiguous
K128-by-N payload available to every DSV4 routed bucket. R54 removes that
selector restriction. The B producer now uses 32 aligned `uint4` loads and
32 `STS.128` stores instead of 128 scalar load/store pairs whenever the scale
payload is coalesced, including regular M128 and compute-bound buckets. Tensor
contents, shared-memory addresses, barriers, decoder math, WGMMA, scheduler,
and epilogue are unchanged.

The eight-rank production correctness suite passes all 12 Flash and Pro
scenarios, including forced ring wrap and Flash M1024, with maximum normalized
differences from `0.000645` to `0.000716`.

The cold-L2 R53/R54/R53 screen uses one warmup, 20 observations, 20 launches
per observation, and the maximum-rank median:

| point | first R53 us | R54 us | change | second R53 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M128 | 501.841 | 502.085 | +0.05% | 504.548 | -0.49% |
| Flash M256 | 519.997 | 513.417 | -1.27% | 506.169 | +1.43% |
| Flash M512 | 907.300 | 891.237 | -1.77% | 909.517 | -2.01% |
| Flash M1024 | 1519.500 | 1502.000 | -1.15% | 1515.500 | -0.89% |
| Flash M2048 | 2775.000 | 2762.500 | -0.45% | 2777.500 | -0.54% |
| Pro M128 | 1333.000 | 1312.500 | -1.54% | 1297.500 | +1.16% |
| Pro M256 | 1685.500 | 1613.000 | -4.30% | 1673.500 | -3.62% |
| Pro M512 | 2610.500 | 2545.000 | -2.51% | 2624.000 | -3.01% |
| Pro M1024 | 3988.000 | 3926.000 | -1.55% | 3998.500 | -1.81% |
| Pro M2048 | 7044.500 | 6951.500 | -1.32% | 7031.000 | -1.13% |

Flash M128 is neutral and Flash M256/Pro M128 are noisy, but both controls
confirm the gains at Flash M512-M2048 and Pro M256-M2048.

Matched one-rank/48-expert Pro M256 profiling confirms that the regular-bucket
gain comes from cheaper scale publication:

| NCU metric | R53 | R54 | change |
| --- | ---: | ---: | ---: |
| duration us | 1707.136 | 1683.648 | -1.38% |
| executed warp instructions | 360,929,083 | 355,936,482 | -1.38% |
| executed thread instructions | 11,332,121,648 | 11,174,378,476 | -1.39% |
| shared-load bank conflicts | 12,453,519 | 12,449,291 | -0.03% |
| shared-store bank conflicts | 6,779,777 | 5,141,683 | -24.16% |
| global-load sectors | 3,321,830 | 3,317,405 | -0.13% |
| global-store sectors | 792,460 | 792,454 | unchanged |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| barrier-stall ratio | 3.627 | 4.034 | +11.22% |
| long-scoreboard ratio | 2.717 | 2.447 | -9.93% |

NSYS independently measures the same launch at 1605.827 us for R53 and
1540.579 us for R54 (`-4.06%`). Exact profiler cubins remain identical in
resources at 128 registers/thread, zero stack, zero local storage, and 1024
bytes static shared memory. Complete correctness, screen, NCU, NSYS, and
resource evidence is under `iter101-vector-all-coalesced-sf-correctness`
through `iter103-vector-all-coalesced-sf-profiles` on the pod and local
artifact root.

### R54 authoritative DSV4 matrix versus PR383

The fresh same-node comparison again uses the required 8-rank DSV4 Flash and
Pro matrix, one warmup, 50 small-M observations, three large-M observations,
20 launches per observation, cold L2, and the maximum-rank median. PR383 uses
its native two-phase FP8 driver and reports L1 plus L2.

| model | M | R54 MXFP4 us | PR383 FP8 us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 320.868 | 293.978 | +9.15% |
| Flash | 16 | 359.835 | 309.514 | +16.26% |
| Flash | 32 | 356.036 | 330.895 | +7.60% |
| Flash | 64 | 383.529 | 376.334 | +1.91% |
| Flash | 128 | 475.163 | 440.190 | +7.95% |
| Flash | 256 | 496.234 | 510.924 | -2.88% |
| Flash | 512 | 891.643 | 916.068 | -2.67% |
| Flash | 1024 | 1499.000 | 1545.368 | -3.00% |
| Flash | 2048 | 2783.000 | 2753.866 | +1.06% |
| Flash | 4096 | 5218.000 | 5064.000 | +3.04% |
| Flash | 8192 | 9988.000 | 9868.000 | +1.22% |
| Pro | 8 | 782.680 | 727.351 | +7.61% |
| Pro | 16 | 1022.000 | 1025.875 | -0.38% |
| Pro | 32 | 1081.500 | 1111.045 | -2.66% |
| Pro | 64 | 1133.500 | 1159.103 | -2.21% |
| Pro | 128 | 1290.000 | 1284.775 | +0.41% |
| Pro | 256 | 1603.000 | 1630.087 | -1.66% |
| Pro | 512 | 2526.000 | 2394.240 | +5.50% |
| Pro | 1024 | 3916.000 | 4011.000 | -2.37% |
| Pro | 2048 | 6937.000 | 7035.000 | -1.39% |
| Pro | 4096 | 13102.000 | 12894.000 | +1.61% |
| Pro | 8192 | 25583.000 | 25096.000 | +1.94% |

Geometric gaps are `+1.979%` over all 22 points, `+4.405%` over the ten
small-M points, and `-0.00045%` over the 12 large-M points. Flash is `+3.446%`
overall (`+8.476%` small, `-0.567%` large); Pro is `+0.532%` overall
(`+0.487%` small, `+0.569%` large). Relative to R53, R54 cuts the all-point
gap from `+3.317%` to `+1.979%` and brings the aggregate large-M comparison to
parity. The remaining dominant cluster is Flash M8-M128, especially M16 and
M128. Complete logs are under `iter104-r54-pr383-full-matrix` on the pod and
local artifact root.

## Rejected R55: direct register-source WGMMA A for Flash M16

### Reason and direction

R54's remaining largest point is Flash M16, where the MXFP4 path is still
`16.26%` behind PR383. R55 prototyped Hopper's
`wgmma.mma_async.m64n{8,16}k32.f32.e4m3.e4m3` register-source-A form for only
the `kHidden=4096`, `kMaxSwapABTokens=16` specialization. The prototype
decoded each packed weight tile directly into CUTE's four-register
`ALayout_64x32` fragment and used staged activations as shared-memory B. Its
goal was to remove expanded-weight STS.128, the decoder rendezvous, and the
subsequent shared-memory WGMMA A reads without changing TMA input, scale
values, promotion, or epilogue semantics.

The eight-rank forced-ring-wrap Flash M16 correctness gate passed at the
unchanged `0.000645` normalized difference. PTXAS also reported zero stack and
zero local storage, but register allocation rose from 114 to 128 registers per
thread. The authoritative 50-observation, 20-launch, cold-L2 R54/RS/R54 screen
was an unambiguous regression:

| point | first R54 us | RS candidate us | change | second R54 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 368.050 | 552.976 | +50.24% | 364.379 | +51.76% |

A follow-up kept only the first `wgmma.fence` rather than fencing every K32
register fragment. It remained correct at `0.000645` but regressed further to
`564.599 us` (`+53.40%/+54.95%` versus the same controls). This rules out the
extra fence count as the main cause. Moving MXFP4 decode from the existing
double-buffered shared-memory preparation into the WGMMA issue path exposes
the full load/decode latency on the tensor-core critical path, and the larger
live register footprint removes compiler scheduling freedom. The entire code
prototype was reverted; only this negative result is retained. Evidence is
under `iter105-flash-m16-register-a-correctness` through
`iter107-flash-m16-register-a-single-fence` on the pod and local artifact
root.

## R56: bank-permute paired packed-weight loads for Flash M16

### Reason and direction

R54 SourceCounters localized all `3,047,424` excessive shared-memory
wavefronts in the Flash M16 profile to the paired decoder's `LDS.64`
instructions. Under the packed B64 swizzle, the previous lane assignment gave
both row `r` and row `r+8` the same adjacent packed-word pair within each half
warp. Their two 64-bit accesses therefore aliased the same banks even though
the complete warp touched the right 16-row by 4-word address set.

R56 retains that exact address set, two-word lookahead, x16 decoder, exponent
lookup, expanded-B STS.128 layout, barriers, WGMMA schedule, and epilogue. It
only alternates packed-word pairs across each half warp's lower and upper
eight rows. Each LDS.64 wave now covers every bank evenly. The selector is
exact for routed DSV4 Flash M16; all other buckets keep their previous lane
mapping.

Eight-rank forced-ring-wrap correctness passes at the unchanged `0.000645`
normalized difference. The formal cold-L2 R54/R56/R54 run uses one warmup, 50
observations, and 20 launches per observation:

| point | first R54 us | R56 us | change | second R54 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 380.262 | 359.316 | -5.51% | 374.251 | -3.99% |

The same mapping was screened over all Flash swap-AB buckets with 20
observations before narrowing the selector:

| point | first R54 us | broad candidate us | change | second R54 us | reverse change | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Flash M8 | 322.562 | 321.075 | -0.46% | 317.695 | +1.06% | reject |
| Flash M16 | 368.073 | 358.178 | -2.69% | 368.002 | -2.67% | accept |
| Flash M32 | 354.332 | 366.552 | +3.45% | 359.463 | +1.97% | reject |
| Flash M64 | 395.642 | 398.453 | +0.71% | 385.525 | +3.35% | reject |

Matched one-rank/32-expert NCU confirms that the formal gain is caused by the
intended shared-memory change:

| Flash M16 NCU metric | R54 | R56 | change |
| --- | ---: | ---: | ---: |
| duration us | 304.608 | 297.632 | -2.29% |
| executed warp instructions | 71,984,926 | 72,365,521 | +0.53% |
| executed thread instructions | 2,249,845,534 | 2,261,890,430 | +0.54% |
| LDS.64 excessive wavefronts | 3,047,424 | 0 | -100.00% |
| shared-load bank conflicts | 3,055,412 | 5,962 | -99.80% |
| shared-store bank conflicts | 868,602 | 994,785 | +14.53% |
| global-load sectors | 835,564 | 835,508 | -0.01% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| barrier-stall ratio | 2.553 | 2.477 | -3.00% |
| long-scoreboard ratio | 2.214 | 2.159 | -2.47% |
| short-scoreboard ratio | 0.765 | 0.583 | -23.76% |

The extra half-warp permutation arithmetic explains the `0.53%` instruction
increase and why the other buckets do not benefit, while removing the exposed
M16 LDS replay still shortens the critical path. NSYS independently measures
`283.296 -> 277.409 us` (`-2.08%`). Both cubins use 114 registers/thread,
zero stack/local storage, 1024 bytes static shared memory, and 110.816 KiB
dynamic shared memory. Complete source-counter, correctness, screening,
formal, NCU, and NSYS evidence is under
`iter108-r54-flash-m16-source-counters` through
`iter112-flash-small-bank-permuted-screen` on the pod and local artifact root.

### R56 authoritative DSV4 matrix against PR383

The post-R56 paired run used the requested `tests/bench_mega_moe_sm90.py`
standard on the same idle eight-H20 pod: Flash and Pro, M=
8/16/32/64/128/256/512/1024/2048/4096/8192, one warmup, 50 observations for
M<=128, three observations for M>=256, 20 launches per observation, cold L2,
and max-rank median. PR383 is timed as its L1+L2 sum; R56 is the fused MXFP4
persistent kernel.

| model | M | R56 us | PR383 us | R56 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 321.680 | 302.466 | +6.35% |
| Flash | 16 | 349.697 | 311.158 | +12.39% |
| Flash | 32 | 380.873 | 331.420 | +14.92% |
| Flash | 64 | 420.351 | 366.732 | +14.62% |
| Flash | 128 | 480.473 | 432.905 | +10.99% |
| Flash | 256 | 508.226 | 506.172 | +0.41% |
| Flash | 512 | 898.071 | 918.616 | -2.24% |
| Flash | 1024 | 1514.000 | 1518.630 | -0.30% |
| Flash | 2048 | 2759.000 | 2734.965 | +0.88% |
| Flash | 4096 | 5214.000 | 5086.000 | +2.52% |
| Flash | 8192 | 9997.000 | 9828.000 | +1.72% |
| Pro | 8 | 794.784 | 715.412 | +11.10% |
| Pro | 16 | 1049.500 | 1005.106 | +4.42% |
| Pro | 32 | 1093.500 | 1099.223 | -0.52% |
| Pro | 64 | 1132.000 | 1166.036 | -2.92% |
| Pro | 128 | 1296.500 | 1273.861 | +1.78% |
| Pro | 256 | 1612.000 | 1686.930 | -4.44% |
| Pro | 512 | 2534.000 | 2449.549 | +3.45% |
| Pro | 1024 | 3926.000 | 4064.000 | -3.40% |
| Pro | 2048 | 6945.000 | 7017.000 | -1.03% |
| Pro | 4096 | 13116.000 | 12901.000 | +1.67% |
| Pro | 8192 | 25530.000 | 25111.000 | +1.67% |

The geometric gaps are `+3.210%` over all 22 points, `+7.137%` over the ten
small-M points, and `+0.048%` over the 12 large-M points. Flash is `+5.483%`
overall and Pro is `+0.986%` overall. At the only changed specialization,
Flash M16, R56 improves the candidate median from R54's `359.835 us` to
`349.697 us` (`-2.82%`) and narrows the paired PR383 gap from `+16.26%` to
`+12.39%`. This is consistent with the formal A/B/A and matched NCU/NSYS
evidence above.

The exact compile-time selector is false for the other 21 points, so their
movement relative to the R54 matrix is not attributable to R56. In
particular, the current Flash M32/M64 values are much slower than R54 even
though those specializations compile the unchanged path; they are treated as
inter-run variance and require a same-session control before choosing the
next code direction. Complete paired logs are under
`iter113-r56-pr383-full-matrix` on the pod and local artifact root.

### R56 same-path variance control and next target

Because the R56 selector is compile-time false for Flash M32/M64, those two
specializations are code-equivalent to the retained R53 control. A same-session
20-observation R56/R53/R56 run nevertheless measured M32 at
`411.147/402.071/375.388 us` and M64 at `442.161/416.373/401.238 us`.
The middle control falls between the two equivalent candidate runs at both
points, demonstrating that the large movement in the R56 full matrix was
node/run drift rather than an R56 side effect. Evidence is under
`iter114-flash-m32-m64-r56-r53-control`.

The next matched profile therefore targeted Pro M8, which remained `11.10%`
behind PR383 in the R56 matrix. One rank and 48 experts preserve Pro's local
expert shard. NCU captures the candidate's one fused launch and both PR383
phase launches; PR383 values below are the L1+L2 sums.

| Pro M8 NCU metric | R56 fused | PR383 L1+L2 | excess |
| --- | ---: | ---: | ---: |
| duration us | 757.824 | 708.736 | +6.93% |
| executed warp instructions | 191,104,980 | 125,792,381 | +51.92% |
| executed thread instructions | 5,975,644,679 | 3,752,731,091 | +59.23% |
| shared-load bank conflicts | 8,526,723 | 27,897 | +30,465.02% |
| shared-store bank conflicts | 3,083,496 | 30,826 | +9,902.91% |
| global-load sectors | 2,207,037 | 1,736,337 | +27.11% |
| global-store sectors | 26,775 | 26,014 | +2.93% |

NSYS measures `699.776 us` for R56 and `684.704 us` for PR383 L1+L2
(`+2.20%`). SourceCounters localizes `8,519,429` excessive shared wavefronts
to the paired decoder's 16 dynamic `LDS.64` sites: eight L1 sites contribute
`696,960` each and eight L2 sites contribute `340,032` each, with the small
remainder in shorter paths. This is the same two-wave replay mechanism removed
from Flash M16 by R56. Complete paired NCU, SourceCounters, and NSYS evidence
is under `iter115-pro-m8-pr383-profiles`.

## R57: bank-permute paired packed-weight loads for selected Pro buckets

### Reason and direction

R57 extends R56's address-set-preserving half-warp permutation to the Pro
swap-AB buckets that pass independent timing. The mapping changes only which
lane loads each adjacent packed-word pair; the complete warp still owns the
same 16-row by 4-word set, and decode, scale values, expanded-B addresses,
WGMMA, scheduler, and epilogue remain unchanged. The final selector includes
Pro `kMaxSwapABTokens=8/32/64`, corresponding to production M8, M32, M64, and
M128, while explicitly excluding the mixed M16 bucket.

Eight-rank Pro M8 correctness first passed at `0.000716`. The broad selector
then passed all six production Pro scenarios, including forced ring wrap for
M16/M32/M64/M128, with differences from `0.000704` to `0.000716`. The 14 CPU
preprocessing contract tests also pass.

The exact Pro M8 formal R56/R57/R56 test uses one warmup, 50 observations, 20
launches per observation, cold L2, and maximum-rank median:

| point | first R56 us | R57 us | change | second R56 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 | 820.531 | 799.677 | -2.54% | 827.003 | -3.30% |

The 20-observation broad screen separates the remaining Pro buckets:

| point | first R56 us | broad us | change | second R56 us | reverse change | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pro M16 | 1047.500 | 1064.000 | +1.58% | 1066.000 | -0.19% | reject |
| Pro M32 | 1098.000 | 1085.000 | -1.18% | 1130.000 | -3.98% | accept |
| Pro M64 | 1142.500 | 1128.000 | -1.27% | 1171.500 | -3.71% | accept |
| Pro M128 | 1328.500 | 1302.500 | -1.96% | 1328.000 | -1.92% | accept |

The retained M32/M64/M128 selector then passes the full 50-observation formal
R56/R57/R56 test:

| point | first R56 us | R57 us | change | second R56 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M32 | 1114.500 | 1086.500 | -2.51% | 1108.500 | -1.98% |
| Pro M64 | 1141.500 | 1125.000 | -1.45% | 1139.000 | -1.23% |
| Pro M128 | 1300.500 | 1277.000 | -1.81% | 1304.000 | -2.07% |

The three-point geometric mean improves by `-1.92%` versus the first control
and `-1.76%` versus the second. Matched Pro M8 profiling confirms the intended
mechanism:

| Pro M8 NCU metric | R56 | R57 | change |
| --- | ---: | ---: | ---: |
| duration us | 757.824 | 744.704 | -1.73% |
| executed warp instructions | 191,104,980 | 192,184,807 | +0.57% |
| executed thread instructions | 5,975,644,679 | 6,009,560,094 | +0.57% |
| excessive shared wavefronts | 8,519,429 | 3,845 | -99.95% |
| shared-load bank conflicts | 8,526,723 | 8,262 | -99.90% |
| shared-store bank conflicts | 3,083,496 | 3,937,409 | +27.69% |
| global-load sectors | 2,207,037 | 2,206,593 | -0.02% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| barrier-stall ratio | 2.385 | 2.302 | -3.50% |
| long-scoreboard ratio | 2.058 | 1.996 | -3.03% |
| short-scoreboard ratio | 0.737 | 0.591 | -19.80% |

Tensor-pipe active rises `2.36%`, eligible warps/cycle rises `5.32%`, and
issue-active rises `4.08%`. NSYS independently measures
`699.776 -> 679.424 us` (`-2.91%`). Both cubins retain 107 registers/thread,
zero stack/local storage, 1024 bytes static shared memory, and 100.576 KiB
dynamic shared memory. Correctness, formal, broad-screen, NCU, SourceCounters,
and NSYS artifacts are under `iter116-pro-m8-bank-permuted-lds` through
`iter120-pro-bank-permuted-formal` on the pod and local artifact root.

### R57 authoritative DSV4 matrix against PR383

The post-R57 paired run again uses the full requested eight-H20 cold-L2
contract: Flash and Pro, all 11 M values, one warmup, 50 observations for
M<=128, three for M>=256, 20 launches per observation, and maximum-rank
median. PR383 is the sum of its native L1 and L2 phase times.

| model | M | R57 us | PR383 us | R57 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 327.500 | 306.927 | +6.70% |
| Flash | 16 | 350.680 | 310.526 | +12.93% |
| Flash | 32 | 366.886 | 329.569 | +11.32% |
| Flash | 64 | 397.852 | 374.104 | +6.35% |
| Flash | 128 | 480.700 | 436.814 | +10.05% |
| Flash | 256 | 489.670 | 511.772 | -4.32% |
| Flash | 512 | 904.737 | 898.654 | +0.68% |
| Flash | 1024 | 1498.000 | 1527.997 | -1.96% |
| Flash | 2048 | 2772.000 | 2716.000 | +2.06% |
| Flash | 4096 | 5194.000 | 5064.000 | +2.57% |
| Flash | 8192 | 10013.000 | 9834.000 | +1.82% |
| Pro | 8 | 770.116 | 709.014 | +8.62% |
| Pro | 16 | 1026.000 | 1004.356 | +2.15% |
| Pro | 32 | 1059.000 | 1103.813 | -4.06% |
| Pro | 64 | 1126.500 | 1153.130 | -2.31% |
| Pro | 128 | 1268.500 | 1280.499 | -0.94% |
| Pro | 256 | 1639.000 | 1654.341 | -0.93% |
| Pro | 512 | 2577.000 | 2417.766 | +6.59% |
| Pro | 1024 | 3947.000 | 4027.000 | -1.99% |
| Pro | 2048 | 6948.000 | 7035.000 | -1.24% |
| Pro | 4096 | 13121.000 | 12875.000 | +1.91% |
| Pro | 8192 | 25554.000 | 25102.000 | +1.80% |

The geometric gaps are `+2.513%` over all 22 points, `+4.925%` over the ten
small-M points, and `+0.546%` over the 12 large-M points. Flash is `+4.248%`
overall (`+9.440%` small, `+0.110%` large); Pro is `+0.808%` overall
(`+0.597%` small, `+0.984%` large). The four R57-changed Pro points have only
a `+0.211%` geometric gap to PR383; M32/M64/M128 now lead PR383 by
`4.06%/2.31%/0.94%`, while Pro M8 remains `8.62%` behind.

Compared with the R56 candidate matrix, the changed four-point geometric mean
improves `2.23%`, agreeing with the formal A/B/A evidence. The all-point
candidate geometric mean improves `0.82%`; movements in unchanged Flash and
large-M points remain run variance. The dominant remaining stable cluster is
Flash M8-M128, especially M16/M32/M128, followed by Pro M8. Complete paired
logs are under `iter121-r57-pr383-full-matrix` on the pod and local artifact
root.

## R58: bank-permute regular Flash packed-weight loads

### Reason and direction

After R57, Flash M32 remained `11.32%` behind PR383. Its matched one-rank,
32-expert profile measures `324.864 us` for the fused candidate versus
`296.384 us` for PR383 L1+L2 (`+9.61%`) and `299.008/291.200 us` under NSYS.
The candidate executes `75.22%` more warp instructions, `80.80%` more thread
instructions, and has `3.16M` shared-load conflicts. SourceCounters localizes
`3,145,728` excessive wavefronts to paired-decoder `LDS.64`. However, R56's
earlier broad screen already showed that applying the permutation to Flash M32
regresses timing; removing replay does not repay the extra scheduling cost in
that swap-AB bucket. This profile is retained under
`iter122-flash-m32-pr383-profiles`, but no M32 code change is made.

Flash M128 is the first regular-orientation point and was still `10.05%`
behind PR383. Its R57 SourceCounters profile has the same `3,145,728`
excessive decoder `LDS.64` wavefronts (`3,553,072` excessive shared
wavefronts overall), 128 registers/thread, and zero local traffic. Because all
regular Flash M values share one JIT specialization, R58 enables the existing
address-set-preserving bank permutation for the entire regular Flash path and
evaluates M128-M8192 as a unit. No data, scale, expanded-B address, WGMMA,
scheduler, or epilogue semantics change. The baseline profile is under
`iter123-flash-m128-source-counters`.

All six production Flash correctness scenarios pass, including forced ring
wrap at M128 and the regular M1024 case; normalized differences remain
`0.000645` to `0.000671`. The initial screen uses 20 observations at M128 and
five at every larger M:

| point | first R57 us | R58 us | change | second R57 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M128 | 519.744 | 500.260 | -3.75% | 505.887 | -1.11% |
| Flash M256 | 560.758 | 507.323 | -9.53% | 518.449 | -2.15% |
| Flash M512 | 911.687 | 914.143 | +0.27% | 905.970 | +0.90% |
| Flash M1024 | 1542.000 | 1501.000 | -2.66% | 1512.000 | -0.73% |
| Flash M2048 | 2801.000 | 2780.000 | -0.75% | 2777.000 | +0.11% |
| Flash M4096 | 5183.000 | 5128.000 | -1.06% | 5194.000 | -1.27% |
| Flash M8192 | 9989.000 | 9915.000 | -0.74% | 10038.000 | -1.23% |

The seven-point geometric mean improves `2.65%/0.79%` against the two
controls. The formal run uses the authoritative 50 observations at M128 and
three observations for larger M:

| point | first R57 us | R58 us | change | second R57 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M128 | 509.186 | 505.036 | -0.82% | 495.622 | +1.90% |
| Flash M256 | 528.685 | 480.897 | -9.04% | 496.425 | -3.13% |
| Flash M512 | 949.545 | 924.335 | -2.65% | 922.429 | +0.21% |
| Flash M1024 | 1513.000 | 1505.000 | -0.53% | 1539.000 | -2.21% |
| Flash M2048 | 2759.000 | 2723.000 | -1.30% | 2774.000 | -1.84% |
| Flash M4096 | 5191.000 | 5134.000 | -1.10% | 5182.000 | -0.93% |
| Flash M8192 | 10023.000 | 9934.000 | -0.89% | 10025.000 | -0.91% |

The formal seven-point geometric mean improves `2.37%/1.00%`; the six large-M
points improve `2.63%/1.47%`. M128 is mixed and M512's `+0.21%` reverse result
is noise-sized, but the indivisible specialization improves the affected set
against both controls and every remaining large point is double-positive.

Matched M128 profiling confirms the mechanism and no resource penalty:

| Flash M128 NCU metric | R57 | R58 | change |
| --- | ---: | ---: | ---: |
| duration us | 464.672 | 462.976 | -0.37% |
| executed warp instructions | 92,008,020 | 90,849,995 | -1.26% |
| executed thread instructions | 2,881,690,645 | 2,844,322,562 | -1.30% |
| excessive shared wavefronts | 3,553,072 | 407,344 | -88.54% |
| shared-load bank conflicts | 3,162,206 | 14,229 | -99.55% |
| shared-store bank conflicts | 1,322,953 | 1,409,686 | +6.56% |
| global-load sectors | 1,064,556 | 1,064,788 | +0.02% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| short-scoreboard ratio | 0.417 | 0.317 | -23.88% |

NSYS independently measures `425.088 -> 420.256 us` (`-1.14%`). Registers
fall from 128 to 126 per thread; both cubins retain zero stack/local storage,
1024 bytes static shared memory, and 110.816 KiB dynamic shared memory. Screen,
formal, NCU, SourceCounters, and NSYS artifacts are under
`iter124-flash-regular-bank-permuted-screen` through
`iter126-flash-regular-bank-permuted-profiles` on the pod and local artifact
root.

### R58 authoritative DSV4 matrix against PR383

The post-R58 matrix uses the same complete eight-H20 cold-L2 contract and a
fresh same-session PR383 baseline:

| model | M | R58 us | PR383 us | R58 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 323.067 | 288.372 | +12.03% |
| Flash | 16 | 348.460 | 319.955 | +8.91% |
| Flash | 32 | 356.480 | 326.608 | +9.15% |
| Flash | 64 | 388.138 | 362.831 | +6.98% |
| Flash | 128 | 472.736 | 427.477 | +10.59% |
| Flash | 256 | 507.785 | 497.910 | +1.98% |
| Flash | 512 | 884.161 | 914.655 | -3.33% |
| Flash | 1024 | 1535.000 | 1523.882 | +0.73% |
| Flash | 2048 | 2773.000 | 2709.603 | +2.34% |
| Flash | 4096 | 5156.000 | 5083.000 | +1.44% |
| Flash | 8192 | 9922.000 | 9818.000 | +1.06% |
| Pro | 8 | 780.301 | 708.382 | +10.15% |
| Pro | 16 | 1041.000 | 1025.902 | +1.47% |
| Pro | 32 | 1068.500 | 1101.980 | -3.04% |
| Pro | 64 | 1133.000 | 1152.183 | -1.66% |
| Pro | 128 | 1270.500 | 1277.056 | -0.51% |
| Pro | 256 | 1631.000 | 1631.402 | -0.02% |
| Pro | 512 | 2527.000 | 2387.071 | +5.86% |
| Pro | 1024 | 3912.000 | 4015.000 | -2.57% |
| Pro | 2048 | 6986.000 | 7037.000 | -0.72% |
| Pro | 4096 | 13091.000 | 12874.000 | +1.69% |
| Pro | 8192 | 25545.000 | 25137.000 | +1.62% |

Geometric gaps are `+2.813%` over all 22 points, `+5.264%` over the ten
small-M points, and `+0.813%` over the 12 large-M points. Flash is `+4.607%`
overall (`+9.517%` small, `+0.684%` large); Pro is `+1.049%` overall
(`+1.177%` small, `+0.942%` large). Relative to the prior R57 candidate run,
the full 22-point geometric mean changes by `-0.25%`, while the seven
R58-affected Flash points are effectively flat at `+0.07%`. This cross-run
comparison does not reproduce the same-session formal A/B/A gain above and is
treated as epoch variance, not change attribution. The terminal gap remains
dominated by Flash M8-M128 and Pro M8. Complete paired logs are under
`iter127-r58-pr383-full-matrix` on the pod and local artifact root.

## R59: use PRMT for selected bank-permuted pair indices

### Reason and direction

R56-R58 removed the dominant packed-weight `LDS.64` replay by assigning the
two adjacent packed-word pairs to complementary lane groups. The accepted
address-set permutation was still expressed as two shifts, an XOR, a mask,
and a multiply in the unrolled decoder. R59 replaces that arithmetic with a
single byte permutation over the constant lookup word `0x00020200`. Its four
bytes encode pair indices `0, 2, 2, 0` for the four eight-lane groups, so the
packed and expanded shared-memory address sets remain identical.

The first prototype applied PRMT to every bank-permuted path. Flash M128
proved that the shorter static expression is not universally safe: targeted
NCU measured `462.976 -> 471.936 us` (`+1.94%`) even though warp/thread
instructions fell `9.88%/10.09%`. Registers rose `126 -> 128` and local
load/store sectors rose from zero to `2,392,064/7,488`. The regular Flash
path therefore retains R58's bit expression. This rejected profile is under
`iter129-prmt-bank-pair-profiles`.

The adjusted screen uses PRMT only for small-M specializations and restores
the regular path. Its 20-observation cold-L2 A/B/A result was:

| point | first R58 us | R59 us | change | second R58 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M8 | 344.847 | 299.094 | -13.27% | 341.952 | -12.53% |
| Flash M16 | 380.548 | 381.250 | +0.18% | 358.198 | +6.44% |
| Flash M32 | 366.900 | 365.789 | -0.30% | 359.410 | +1.78% |
| Flash M64 | 390.264 | 379.396 | -2.78% | 389.217 | -2.52% |
| Pro M8 | 802.935 | 786.372 | -2.06% | 810.288 | -2.95% |
| Pro M32 | 1083.000 | 1046.000 | -3.42% | 1078.000 | -2.97% |
| Pro M64 | 1127.000 | 1085.000 | -3.73% | 1128.000 | -3.81% |
| Pro M128 | 1284.000 | 1241.000 | -3.35% | 1283.000 | -3.27% |

Only double-positive points are retained. Flash M8/M64 and Pro
M8/M32/M64/M128 use PRMT. Flash M16 keeps R56's bank permutation with the
original bit expression, and Flash M32 keeps its original pair mapping. All
28 full-suite correctness scenarios pass, including forced ring wrap, with
no numerical change beyond the existing tolerance.

### Formal A/B/A performance

The selected six points were rerun with one warmup, 50 observations, 20
launches per observation, maximum-rank median, and an explicit L2 flush:

| point | first R58 us | R59 us | change | second R58 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M8 | 322.295 | 295.376 | -8.35% | 317.716 | -7.03% |
| Flash M64 | 379.826 | 369.758 | -2.65% | 389.968 | -5.18% |
| Pro M8 | 805.698 | 770.571 | -4.36% | 794.110 | -2.96% |
| Pro M32 | 1064.500 | 1037.500 | -2.54% | 1066.500 | -2.72% |
| Pro M64 | 1123.000 | 1101.500 | -1.91% | 1127.000 | -2.26% |
| Pro M128 | 1306.000 | 1233.000 | -5.59% | 1272.000 | -3.07% |

The six-point geometric mean improves `4.26%/3.89%` against the first and
second controls. Every retained point is double-positive in both the screen
and formal run.

Matched one-rank profiles preserve each production expert shard: 32 experts
for Flash and 48 for Pro. Flash M8 directly confirms both intended effects:

| Flash M8 NCU metric | R58 | R59 | change |
| --- | ---: | ---: | ---: |
| duration us | 239.712 | 222.400 | -7.22% |
| executed warp instructions | 52,588,063 | 47,446,789 | -9.78% |
| executed thread instructions | 1,640,308,174 | 1,475,871,922 | -10.03% |
| shared-load bank conflicts | 2,364,982 | 3,682 | -99.84% |
| shared-store bank conflicts | 735,098 | 937,651 | +27.56% |
| global-load sectors | 647,578 | 644,477 | -0.48% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| tensor-pipe active | 3.833% | 4.123% | +7.56% |

Both Flash cubins use 114 registers/thread and 110.816 KiB dynamic shared
memory. NSYS independently measures `223.488 -> 206.656 us` (`-7.53%`). For
Pro M8, NCU measures `744.704 -> 715.808 us` (`-3.88%`), with warp/thread
instructions down `10.24%/10.47%`, no local traffic, and 107 registers/thread.
Matched 48-expert NSYS measures `684.512 -> 666.431 us` (`-2.64%`). Screen,
formal, correctness, NCU, and NSYS evidence is under
`iter128-prmt-bank-pair-selector` through
`iter132-prmt-selected-profiles` on the pod and local artifact root.

### R59 authoritative DSV4 matrix against PR383

The post-R59 matrix uses the complete eight-H20 cold-L2 contract and a fresh
same-session PR383 baseline. Candidate times are the fused MXFP4 kernel;
PR383 times are the sum of its native FP8 L1 and L2 kernels.

| model | M | R59 us | PR383 us | R59 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 301.980 | 303.619 | -0.54% |
| Flash | 16 | 346.790 | 315.000 | +10.09% |
| Flash | 32 | 366.952 | 331.214 | +10.79% |
| Flash | 64 | 376.518 | 373.333 | +0.85% |
| Flash | 128 | 502.232 | 433.288 | +15.91% |
| Flash | 256 | 489.029 | 496.087 | -1.42% |
| Flash | 512 | 889.985 | 921.492 | -3.42% |
| Flash | 1024 | 1546.000 | 1516.728 | +1.93% |
| Flash | 2048 | 2777.000 | 2726.481 | +1.85% |
| Flash | 4096 | 5159.000 | 5083.000 | +1.50% |
| Flash | 8192 | 9904.000 | 9824.000 | +0.81% |
| Pro | 8 | 759.296 | 710.803 | +6.82% |
| Pro | 16 | 1032.500 | 1002.712 | +2.97% |
| Pro | 32 | 1039.500 | 1101.702 | -5.65% |
| Pro | 64 | 1080.500 | 1156.624 | -6.58% |
| Pro | 128 | 1227.000 | 1280.133 | -4.15% |
| Pro | 256 | 1622.000 | 1624.145 | -0.13% |
| Pro | 512 | 2587.000 | 2396.005 | +7.97% |
| Pro | 1024 | 3947.000 | 4013.000 | -1.64% |
| Pro | 2048 | 6994.000 | 7032.000 | -0.54% |
| Pro | 4096 | 13143.000 | 12925.000 | +1.69% |
| Pro | 8192 | 25420.000 | 25080.000 | +1.36% |

The geometric gaps are `+1.700%` over all 22 points, `+2.799%` over the ten
small-M points, and `+0.794%` over the 12 large-M points. Flash is `+3.333%`
overall (`+7.237%` small, `+0.189%` large). Pro is effectively at parity at
`+0.093%` overall, with small M leading PR383 by `1.456%` and large M behind
by `1.402%`. Flash M8 now leads PR383 by `0.54%`; the remaining dominant
cluster is Flash M16/M32/M128, followed by Pro M512 and Pro M8. Relative to
R58's prior matrix, the all-point geometric gap contracts from `+2.813%` to
`+1.700%`, consistent with the same-code R59 A/B/A result. Complete paired
logs are under `iter133-r59-pr383-full-matrix` on the pod and local artifact
root.

## Rejected R60: extend incremental WGMMA descriptors below M8192

R59 only enables base-plus-increment shared-memory WGMMA descriptors for
Flash M8192. R60 temporarily extended that existing path to regular Flash
M128-M4096. The experiment removes repeated descriptor bit-field construction
inside each WGMMA group without changing data layout, WGMMA count, pipeline
order, or numerical operations. All 28 full-suite correctness scenarios
passed.

The initial cold-L2 A/B/A screen used 20 observations at M128 and five at the
larger points:

| point | first R59 us | R60 us | change | second R59 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M128 | 519.018 | 496.808 | -4.28% | 506.428 | -1.90% |
| Flash M256 | 561.842 | 551.578 | -1.83% | 538.625 | +2.40% |
| Flash M512 | 924.740 | 944.922 | +2.18% | 907.315 | +4.15% |
| Flash M1024 | 1502.000 | 1525.000 | +1.53% | 1496.000 | +1.94% |
| Flash M2048 | 2798.000 | 2806.000 | +0.29% | 2779.000 | +0.97% |
| Flash M4096 | 5137.000 | 5164.000 | +0.53% | 5137.000 | +0.53% |

Only M128 was double-positive, so the selector was narrowed to that point
plus the pre-existing M8192 path and rerun with the authoritative 50
observations. The formal result was `498.717 -> 499.386 -> 492.754 us`, a
`+0.13%/+1.35%` regression against the two controls. The source change was
fully reverted before commit, and no profiler result is used to overrule the
formal timing gate. Correctness, screen, and formal logs are under
`iter134-incremental-desc-regular-flash-screen` and
`iter135-incremental-desc-flash-m128-formal` on the pod and local artifact
root.

## R61: extend PRMT bank-permuted pairs to Flash M32

### Reason and selection

After R59, Flash M16/M32 remained `10.09%/10.79%` behind PR383. Both execute
the paired packed-word decoder, but R59 retained the old M16 bit expression
and the original conflict-heavy M32 pair assignment because the earlier
20-observation screen was mixed. R61 reran both points directly at the formal
50-observation sample count. M16 used the byte-permutation form of its already
accepted address mapping; M32 used the same `0,2,2,0` bank permutation and
PRMT lookup accepted at Flash M8/M64.

All six production Flash correctness scenarios pass. The cold-L2 A/B/A result
was:

| point | first R59 us | R61 us | change | second R59 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 345.088 | 365.712 | +5.98% | 380.761 | -3.95% |
| Flash M32 | 359.600 | 338.182 | -5.96% | 358.520 | -5.67% |

M16 remains order-dependent and is rejected; it keeps R56's bit expression.
M32 is strongly double-positive and is the only retained specialization.

### NCU and NSYS attribution

The matched one-rank profile uses the production Flash shard of 32 experts:

| Flash M32 NCU metric | R59 | R61 | change |
| --- | ---: | ---: | ---: |
| duration us | 323.328 | 302.368 | -6.48% |
| executed warp instructions | 78,961,660 | 72,075,009 | -8.72% |
| executed thread instructions | 2,471,627,807 | 2,251,317,121 | -8.91% |
| shared-load bank conflicts | 3,156,785 | 8,069 | -99.74% |
| shared-store bank conflicts | 868,621 | 1,062,417 | +22.31% |
| global-load sectors | 906,736 | 904,783 | -0.22% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| tensor-pipe active | 4.109% | 4.414% | +7.42% |

Registers rise from 122 to 125 per thread, but both cubins remain spill-free
and use 110.816 KiB dynamic shared memory. NSYS independently measures
`301.536 -> 282.816 us` (`-6.21%`). Correctness and formal timing are under
`iter136-prmt-flash-m16-m32-formal`; matched NCU/NSYS evidence is under
`iter137-prmt-flash-m32-profiles` on the pod and local artifact root.

### R61 authoritative DSV4 matrix against PR383

The full post-R61 matrix uses the same eight-H20 cold-L2 contract and a fresh
same-session PR383 baseline:

| model | M | R61 us | PR383 us | R61 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 303.287 | 311.990 | -2.79% |
| Flash | 16 | 341.587 | 323.428 | +5.62% |
| Flash | 32 | 340.203 | 335.647 | +1.36% |
| Flash | 64 | 364.724 | 374.774 | -2.68% |
| Flash | 128 | 485.640 | 449.130 | +8.13% |
| Flash | 256 | 515.830 | 491.971 | +4.85% |
| Flash | 512 | 935.094 | 901.984 | +3.67% |
| Flash | 1024 | 1487.000 | 1526.123 | -2.56% |
| Flash | 2048 | 2734.000 | 2712.108 | +0.81% |
| Flash | 4096 | 5162.000 | 5057.000 | +2.08% |
| Flash | 8192 | 9934.000 | 9835.000 | +1.01% |
| Pro | 8 | 756.875 | 710.089 | +6.59% |
| Pro | 16 | 1024.500 | 1002.966 | +2.15% |
| Pro | 32 | 1036.500 | 1098.664 | -5.66% |
| Pro | 64 | 1063.500 | 1150.481 | -7.56% |
| Pro | 128 | 1215.000 | 1280.017 | -5.08% |
| Pro | 256 | 1637.000 | 1633.680 | +0.20% |
| Pro | 512 | 2541.000 | 2439.312 | +4.17% |
| Pro | 1024 | 3957.000 | 4014.000 | -1.42% |
| Pro | 2048 | 6948.000 | 7034.000 | -1.22% |
| Pro | 4096 | 13096.000 | 12908.000 | +1.46% |
| Pro | 8192 | 25432.000 | 24968.000 | +1.86% |

The all-point geometric gap is now `+0.602%`; the ten small-M points lead
PR383 by `0.131%`, while the 12 large-M points trail by `1.217%`. Flash is
`+1.713%` overall (`+1.832%` small, `+1.614%` large). Pro now leads PR383 by
`0.497%` overall and `2.057%` at small M, while trailing by `0.822%` at large
M. Flash M32 contracts from R59's `+10.79%` to `+1.36%`, consistent with the
matched `5.96%/5.67%` A/B/A gain despite the expected cross-epoch movement in
both implementations. The terminal target is not yet met: the total gap is
positive, with Flash M128/M16 and Pro M8/M512 the largest residual points.
Complete paired logs are under `iter138-r61-pr383-full-matrix` on the pod and
local artifact root.

## Rejected R62: sparse dispatch completion for Flash M128

R62 extended the already-correct Flash M32/M1024 sparse dispatch-completion
path to M128. The candidate replaces one completion atomic per expert per CTA
with nonzero-count atomics plus the existing SM0 rank-count aggregation; the
math pipeline and output are unchanged. The production M128 forced-ring-wrap
correctness case passed.

The 20-observation cold-L2 A/B/A screen measured
`500.716/502.781/503.255 us` for R61/R62/R61. R62 regressed `0.41%` against
the first control and improved only `0.09%` against the second. The result is
both order-dependent and noise-sized, so the selector extension was fully
reverted before commit and did not enter formal profiling. Evidence is under
`iter139-sparse-flash-m128-screen` on the pod and local artifact root.

## Rejected R63: single-block scheduler lookup for Flash M128

R63 exposed the R29 scheduler owner-lookup fast path through an explicit JIT
template selector and enabled it only at Flash M128. When every local expert
owns at most one M64 block, the path selects the owner from the nonempty-lane
mask instead of rebuilding a warp prefix sum; skewed experts still take the
unchanged multi-block fallback. The candidate extension was rebuilt so its
host generator and new template ABI matched, and the production M128
forced-ring-wrap scenario passed.

The 20-observation cold-L2 A/B/A screen measured
`512.465/512.149/509.619 us` for R61/R63/R61. The candidate improved only
`0.06%` against the first control and regressed `0.50%` against the second.
The result rejects scheduler owner lookup as the source of M128's remaining
gap. All selector plumbing was reverted and the original R61 extension ABI
restored. Evidence, including the initial stale-extension compile failure and
the successful matched rerun, is under
`iter140-single-block-flash-m128-screen` on the pod and local artifact root.

## Rejected R64: back off Flash M128 ring-reuse polling

### R61 versus PR383 attribution

A matched one-rank, 32-expert profile first compared the fused R61 Flash M128
kernel with PR383's native L1 and L2 kernels. NCU measured R61 at `460.448 us`
versus `259.744 + 163.104 = 422.848 us` for PR383 (`+8.89%`); NSYS measured
`423.391 us` versus `244.160 + 150.719 = 394.879 us` (`+7.22%`). R61 executes
`90,812,797` warp instructions and `2,843,464,552` thread instructions versus
PR383's summed `76,008,346` and `2,330,575,415` (`+19.48%/+22.01%`). Both
paths remain spill-free.

SourceCounters then localized the largest extra instruction stream to a
single acquire-poll loop: `LDG.E.STRONG.GPU`, `CCTL.IVALL`, `YIELD`, compare,
and branch execute `566,230` times. The loop is dispatch's wait for
`l1_empty_count` before overwriting a previous ring generation. Its 768 loop
entries match the 768 Flash M128 routed tokens that wrap onto the second
physical-ring generation. This is a stronger attribution than the remaining
shared-memory conflicts: R61 has only `14,723` shared-load conflicts, while
its shared-store conflicts (`1,437,753`) are concentrated in the BF16 L2
epilogue materialization shared with the output path. The matched NCU/NSYS
reports are under `iter141-flash-m128-pr383-profiles`.

### Experiment and rejection

R64 added `__nanosleep(64)` only inside that Flash M128 ring-reuse poll;
every other Flash M and every Pro specialization retained R61 source. The
eight-rank forced-wrap correctness case passed with the unchanged
`diff=0.000658`. The authoritative one-warmup, 50-observation, 20-launch,
cold-L2 A/B/A result was:

| point | first R61 us | R64 us | change | second R61 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M128 | 494.484 | 505.188 | +2.16% | 497.108 | +1.63% |

The double regression shows that this spin traffic is a symptom of the live
ring dependency, not latency that a 64-cycle sleep can hide: delaying dispatch
also delays the next generation's activation supply. The source change was
fully reverted before commit. Formal logs and the exact rejected source are
under `iter142-flash-m128-poll-backoff-formal` on the pod and local artifact
root. The terminal goal remains unmet; shorter polling or a schedule change
must be measured independently rather than inferred from instruction counts.

## R65: short Flash M128 ring-poll backoff

### Reason and implementation

R65 revisits the R64-attributed dispatch wait with a much shorter delay. Only
the eight-rank Flash M128 specialization executes `__nanosleep(16)` after an
unsuccessful acquire load of `l1_empty_count`; all Pro shapes, other Flash M,
and one-rank profiling builds retain the original tight loop. The intent is to
reduce redundant cross-rank polling traffic without withholding dispatch long
enough to starve the next activation generation, as R64's 64-cycle delay did.

The production forced-ring-wrap correctness scenario passes with the unchanged
`diff=0.000658`. Two independent cold-L2 A/B/A runs were directionally
consistent:

| sample count | first R61 us | R65 us | change | second R61 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20-observation screen | 515.198 | 512.308 | -0.56% | 514.350 | -0.40% |
| 50-observation formal | 493.279 | 492.384 | -0.18% | 501.039 | -1.73% |

The absolute gain is small, but four comparisons against independently placed
controls have the same sign, so the narrowly selected change is retained.
Correctness and timing logs are under
`iter143-flash-m128-short-poll-backoff-screen` and
`iter144-flash-m128-short-poll-backoff-formal`.

### NCU and NSYS mechanism check

Multi-process NCU replay cannot safely profile this persistent collective: it
replays one rank while peers wait at a barrier, so both the application-replay
and NVTX-range attempts were terminated without treating their partial output
as evidence. Eight-rank NSYS also strongly perturbs rank scheduling: R61 and
R65 totals were `93.305/244.737 ms`, with wide per-rank spreads. Those totals
are intentionally excluded from the performance decision.

Instead, a diagnostic one-rank build enabled the same 16-cycle delay at Flash
M128 so the exact poll mechanism could be measured without collective replay.
NCU shows global-load sectors falling from `1,062,666` to `1,034,466`
(`-2.65%`), while executed warp/thread instructions rise only
`0.36%/0.36%`; both variants use 126 registers, 110.816 KiB dynamic shared
memory, and zero local load/store sectors. NCU duration moves
`460.832 -> 462.816 us` (`+0.43%`), while NSYS measures
`423.647 -> 423.167 us` (`-0.11%`). The diagnostic therefore confirms reduced
poll traffic rather than a faster arithmetic path; the production eight-rank
A/B/A timing remains the authority for acceptance. Reports and failed-profiler
logs are under `iter145-flash-m128-short-poll-profiles`. The terminal aggregate
target remains unmet, so the next iteration targets larger weighted residuals
rather than extrapolating this point gain.

## R66: bank-permute Pro M512+ packed-weight loads

### Attribution and selector

R61's full matrix left Pro M512 `4.17%` behind PR383. A matched one-rank,
48-expert profile shows that the fused arithmetic path itself is already
competitive: NCU measures `2.442 ms` versus PR383's
`1.636 + 0.852 = 2.488 ms`, and NSYS measures `2.242 ms` versus
`1.487 + 0.786 = 2.273 ms`. The eight-rank deficit is therefore dominated by
work that is amplified by the persistent multi-rank schedule rather than by a
fundamental tensor-core throughput shortfall.

The same profile exposes `18,223,366` shared-load bank conflicts in the fused
kernel versus only `1,959 + 13,479` across PR383's two phases. R66 extends the
address-set-preserving paired-word lane permutation already validated for
Flash and Pro small-M to regular Pro M512 and above. A dedicated generated
template boolean keeps Pro M256 on its old mapping after a broad screen found
that point order-dependent; all previously accepted selectors are unchanged.
The full warp still loads the identical 16-row by four-word set, so packed
weights, scale lookup, expanded-B addresses, WGMMA, and numerical behavior do
not change. A new `production.pro_m512` regression scenario covers the exact
selector and passes on eight ranks with `diff=0.000708`. The rebuilt extension
also passes all 13 eight-rank production scenarios, including every forced ring
wrap case; that complete log is under `iter152-r66-production-correctness`.

The first broad five-observation R61/R66/R61 screen measured the six regular
Pro points as follows. M256 used the broad prototype and was removed before
the final run; the remaining five points were double-positive as a set.

| M | first R61 us | broad R66 us | change | second R61 us | reverse change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1620 | 1639 | +1.17% | 1663 | -1.44% |
| 512 | 2578 | 2560 | -0.70% | 2562 | -0.08% |
| 1024 | 3950 | 3926 | -0.61% | 3927 | -0.03% |
| 2048 | 6960 | 6903 | -0.82% | 6996 | -1.33% |
| 4096 | 13113 | 13017 | -0.73% | 13123 | -0.81% |
| 8192 | 25549 | 25314 | -0.92% | 25444 | -0.51% |

After narrowing the JIT selector to M512+, the requested three-observation,
20-launch, cold-L2 formal A/B/A result is:

| M | first R61 us | R66 us | change | second R61 us | reverse change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2721 | 2571 | -5.51% | 2598 | -1.04% |
| 1024 | 3918 | 3890 | -0.71% | 3963 | -1.84% |
| 2048 | 7041 | 6918 | -1.75% | 6987 | -0.99% |
| 4096 | 13226 | 12994 | -1.75% | 13168 | -1.32% |
| 8192 | 25472 | 25274 | -0.78% | 25580 | -1.20% |

The five-point geometric mean improves `2.12%/1.28%` against the two
controls. M512's first control is visibly noisy, but its reverse comparison
and all eight other affected-point comparisons remain favorable.

### NCU and NSYS confirmation

Matched one-rank Pro M512 profiling before and after R66 confirms the intended
mechanism:

| metric | R65 | R66 | change |
| --- | ---: | ---: | ---: |
| NCU duration ms | 2.442016 | 2.428736 | -0.54% |
| executed warp instructions | 520,691,733 | 513,922,278 | -1.30% |
| executed thread instructions | 16,351,510,982 | 16,134,537,441 | -1.33% |
| shared-load bank conflicts | 18,223,366 | 122,939 | -99.33% |
| shared-store bank conflicts | 8,442,903 | 8,425,886 | -0.20% |
| global-load sectors | 4,899,826 | 4,899,089 | -0.02% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| tensor-pipe active | 23.128% | 23.249% | +0.52% |
| short-scoreboard ratio | 0.425 | 0.332 | -21.71% |

Both cubins retain 128 registers/thread and 100.576 KiB dynamic shared
memory. NSYS independently measures `2.242493 -> 2.215009 ms` (`-1.23%`).
The first post-selector correctness launch also documented the expected stale
host-extension template mismatch; rebuilding `_C.so` produced the successful
run above. PR383 attribution, broad screen, correctness/build, formal A/B/A,
and final NCU/NSYS evidence are under `iter146` through `iter152` on the pod
and local artifact root. The aggregate terminal target still requires a fresh
full matrix and further iteration.

### R66 authoritative DSV4 matrix against PR383

The fresh same-session comparison uses the requested
`tests/bench_mega_moe_sm90.py` contract: Flash and Pro, M=
8/16/32/64/128/256/512/1024/2048/4096/8192, one warmup, 50 observations for
M<=128, three observations for M>=256, 20 launches per observation, explicit
cold-L2 flushing, and the maximum-rank median. PR383 remains the sum of its
native FP8 L1 and L2 kernels; R66 is the fused MXFP4 persistent kernel.

| model | M | R66 us | PR383 us | R66 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 302.762 | 301.160 | +0.53% |
| Flash | 16 | 354.725 | 314.010 | +12.97% |
| Flash | 32 | 344.835 | 329.506 | +4.65% |
| Flash | 64 | 366.872 | 361.210 | +1.57% |
| Flash | 128 | 475.981 | 435.668 | +9.25% |
| Flash | 256 | 485.714 | 493.876 | -1.65% |
| Flash | 512 | 888.185 | 900.257 | -1.34% |
| Flash | 1024 | 1490.000 | 1506.433 | -1.09% |
| Flash | 2048 | 2743.000 | 2722.000 | +0.77% |
| Flash | 4096 | 5147.000 | 5060.000 | +1.72% |
| Flash | 8192 | 9934.000 | 9825.000 | +1.11% |
| Pro | 8 | 751.436 | 709.835 | +5.86% |
| Pro | 16 | 1021.500 | 1007.957 | +1.34% |
| Pro | 32 | 1034.000 | 1103.014 | -6.26% |
| Pro | 64 | 1063.500 | 1156.148 | -8.01% |
| Pro | 128 | 1227.000 | 1284.181 | -4.45% |
| Pro | 256 | 1668.000 | 1633.804 | +2.09% |
| Pro | 512 | 2530.000 | 2401.817 | +5.34% |
| Pro | 1024 | 3905.000 | 4018.000 | -2.81% |
| Pro | 2048 | 6948.000 | 7048.000 | -1.42% |
| Pro | 4096 | 13002.000 | 12908.000 | +0.73% |
| Pro | 8192 | 25293.000 | 24989.000 | +1.22% |

The 22-point geometric gap is `+0.901%`. Pro now leads PR383 by `0.672%`
overall and its small-M subset leads by `2.438%`, confirming that R66 did not
disturb the already strong Pro small-M specializations. Flash large M is at
effective parity (`-0.089%`), but Flash small M trails by `5.691%`, making it
the dominant aggregate residual. Across both models, small and large M trail
by `1.545%` and `0.366%`, respectively. The largest individual deficits are
Flash M16 (`+12.97%`), Flash M128 (`+9.25%`), Pro M8 (`+5.86%`), and Pro M512
(`+5.34%`). The terminal goal remains unmet; the next iteration targets the
Flash M16/M128 critical paths without modifying the already-leading Pro
small-M or Flash M256-M1024 buckets. Complete logs are under
`iter153-r66-pr383-full-matrix` on the pod and local artifact root.

## R67: move mature Flash M128 below the swap-AB crossover

### Reason and implementation

R66's fresh matrix leaves Flash M128 `9.25%` behind PR383. The regular
orientation computes a full M64 tensor-core tile for every expert block even
when the final block owns far fewer routed tokens. R25 previously tested
swap-AB at M128 and rejected it by `2.07-4.05%`, but that experiment predates
the accepted vectorized weight-scale staging, packed HFMA2 promotion, PRMT
decoder, and bank-permuted packed loads. R67 therefore retests the structural
crossover on the mature path instead of treating the old microarchitecture as
permanent evidence.

The only source change extends routed Flash's compile-time `small_m_swap_ab`
selector from M<=64 to M<=128. The existing runtime N8/N16/N32/N64 buckets
still split expert ownership into M64 tasks, retain the same processed MXFP4
weights and blockwise activation scales, and preserve every other Flash and
Pro specialization. The exact eight-rank forced-ring-wrap M128 case passes at
the unchanged `diff=0.000658`; all 13 production scenarios subsequently pass,
including every Flash/Pro ring-wrap case.

### Screening and formal performance

The 20-observation cold-L2 R66/R67/R66 screen measured
`539.814/444.519/495.906 us`. The first control was visibly slow, but R67 also
beats the faster reverse control by `10.36%`. The authoritative one-warmup,
50-observation, 20-launch, cold-L2 A/B/A result is:

| point | first R66 us | R67 us | change | second R66 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M128 | 503.079 | 439.360 | -12.67% | 501.610 | -12.41% |

Rank-0 medians move from `484.712/476.882 us` to `434.562 us`, so the gain is
not created by a single slow peer. Relative to the immediately preceding
same-session PR383 M128 result (`435.668 us`), R67 is now only `0.85%` behind,
although a fresh complete matrix remains the aggregate authority.

### NCU and NSYS mechanism check

A matched one-rank/32-expert profile preserves one production Flash expert
shard while avoiding collective replay. It confirms that the structural
selector reduces elapsed time despite executing more scalar control and
remapping instructions:

| metric | R66 regular | R67 swap-AB | change |
| --- | ---: | ---: | ---: |
| NCU duration us | 460.736 | 399.744 | -13.24% |
| executed warp instructions | 90,815,495 | 102,993,473 | +13.41% |
| executed thread instructions | 2,843,365,775 | 3,239,074,047 | +13.92% |
| shared-load bank conflicts | 14,715 | 15,135 | +2.85% |
| shared-store bank conflicts | 1,430,057 | 913,437 | -36.13% |
| global-load sectors | 1,060,109 | 1,053,287 | -0.64% |
| local load/store sectors | 0 / 0 | 16,384 / 2,496 | new |
| registers/thread | 126 | 128 | +2 |
| dynamic shared memory KiB | 110.816 | 110.816 | unchanged |

The extra generic instructions and small local frame are outweighed by
bucketed tensor-core work that no longer evaluates padded token rows. NSYS
independently measures `419.392 -> 364.672 us` (`-13.05%`). Correctness,
screening, formal timing, NCU, and NSYS artifacts are under `iter154` through
`iter157` on the pod and local artifact root. The terminal aggregate target is
still not assumed complete; R67 requires a fresh 22-point PR383 matrix before
the next residual is selected.

### R67 authoritative DSV4 matrix against PR383

The fresh same-session matrix repeats the exact R66 contract on the same idle
eight-H20 pod. Only Flash M128 has different generated source; movement at the
other 21 points is retained for honest epoch pairing but is not attributed to
R67.

| model | M | R67 us | PR383 us | R67 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 309.230 | 313.736 | -1.44% |
| Flash | 16 | 348.065 | 316.442 | +9.99% |
| Flash | 32 | 340.196 | 336.476 | +1.11% |
| Flash | 64 | 371.346 | 366.626 | +1.29% |
| Flash | 128 | 439.849 | 437.834 | +0.46% |
| Flash | 256 | 518.505 | 502.696 | +3.14% |
| Flash | 512 | 917.623 | 906.578 | +1.22% |
| Flash | 1024 | 1493.000 | 1546.808 | -3.48% |
| Flash | 2048 | 2804.000 | 2720.567 | +3.07% |
| Flash | 4096 | 5120.000 | 5082.000 | +0.75% |
| Flash | 8192 | 9959.000 | 9819.000 | +1.43% |
| Pro | 8 | 754.309 | 711.933 | +5.95% |
| Pro | 16 | 1033.500 | 1008.123 | +2.52% |
| Pro | 32 | 1027.500 | 1105.444 | -7.05% |
| Pro | 64 | 1071.000 | 1162.555 | -7.88% |
| Pro | 128 | 1228.500 | 1277.919 | -3.87% |
| Pro | 256 | 1622.000 | 1618.674 | +0.21% |
| Pro | 512 | 2526.000 | 2415.377 | +4.58% |
| Pro | 1024 | 3909.000 | 4010.000 | -2.52% |
| Pro | 2048 | 6872.000 | 7033.000 | -2.29% |
| Pro | 4096 | 12968.000 | 12956.000 | +0.09% |
| Pro | 8192 | 25297.000 | 25066.000 | +0.92% |

The 22-point geometric gap falls from R66's `+0.901%` to `+0.296%`. Small M
across both models now leads by `0.027%`, while large M trails by `0.566%`.
Flash remains `+1.545%` overall (`+2.207%` small, `+0.996%` large); Pro leads
by `0.938%` overall and `2.213%` at small M, with large Pro at effective
parity (`+0.138%`). The changed Flash M128 point closes from `+9.25%` to
`+0.46%`, consistent with the direct A/B/A and profiler evidence. Flash
M16 (`+9.99%`), Pro M8 (`+5.95%`), and Pro M512 (`+4.58%`) are the largest
stable residuals. Flash M256/M2048 moved in source-identical code and remain
epoch-sensitive, so they are not selected from one three-observation matrix.
The terminal goal is still unmet by `0.296%`; complete paired logs are under
`iter158-r67-pr383-full-matrix` on the pod and local artifact root.

## Rejected R68: mature-path Flash M16 HFMA2 promotion

R68 retested packed BF16 HFMA2 promotion at exact Flash M16. The old R36
experiment was mixed before M16 gained vectorized scale staging and the
bank-permuted decoder, so this run asked whether the current register schedule
could finally benefit from replacing unpack, four scalar FP32 FMAs, and repack
with two packed operations. A dedicated JIT cache forced recompilation of the
header-only prototype. Eight-rank forced-ring-wrap correctness passed at
`diff=0.000654`.

The 20-observation R67/R68/R67 screen was positive at
`354.249/336.460/361.978 us` (`-5.02%/-7.05%`). The authoritative
50-observation run did not reproduce it:

| point | first R67 us | R68 us | change | second R67 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 371.614 | 377.333 | +1.54% | 387.561 | -2.64% |

Matched one-rank/32-expert profiling shows that the local arithmetic change is
real but too small to control the eight-rank maximum. NCU duration moves
`299.424 -> 292.768 us` (`-2.22%`), warp/thread instructions fall
`72,366,234 -> 68,741,400` (`-5.01%`) and
`2,261,847,566 -> 2,145,999,348` (`-5.12%`), and NSYS moves
`277.952 -> 270.592 us` (`-2.65%`). Both variants remain at 114 registers,
110.816 KiB dynamic shared memory, and zero local load/store sectors; shared
store conflicts increase from `994,555` to `1,065,178` (`+7.10%`). The
formal sign reversal therefore reflects a gain smaller than distributed
scheduling variance, not spill damage. The source change was fully reverted
and is not part of R67. Evidence is under `iter159` through `iter162` on the
pod and local artifact root.

## Rejected R69: extend L2 C/D swizzle to M512

R69 lowered the existing B128-swizzled L2 C/D epilogue threshold from M1024
to M512, targeting the simultaneous Flash/Pro M512 residuals. The swizzle
preserves every BF16 value and final NVLink address while spreading epilogue
stores across shared-memory banks. New exact Flash M512 coverage and the
existing Pro M512 scenario both passed on eight ranks; Flash also verified
physical-ring reuse at `diff=0.000662`.

The five-observation broad R67/R69/R67 screen measured Flash M512 at
`952.286/932.095/939.286 us` (`-2.12%/-0.77%`) and Pro M512 at
`2591/2554/2536 us` (`-1.43%/+0.71%`). Pro was immediately excluded. The
selector was narrowed to routed Flash M512, then rerun with the authoritative
three-observation large-M contract:

| point | first R67 us | R69 us | change | second R67 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M512 | 937.935 | 947.976 | +1.07% | 1034.000 | -8.32% |

The candidate rank-0 median beat both controls, but the required maximum-rank
median changed sign and the second control was visibly slow. One-rank NCU
confirms a real but noise-sized mechanism: shared-store bank conflicts fall
`3,517,322 -> 2,375,929` (`-32.45%`), while executed warp instructions rise
`185,673,652 -> 187,308,560` (`+0.88%`) and duration moves only
`906.016 -> 903.392 us` (`-0.29%`). NSYS similarly moves
`831.007 -> 827.871 us` (`-0.38%`); both paths use 126 registers, 110.816 KiB
dynamic shared memory, and zero local sectors. The local benefit is too small
to control the production max-rank score, so the selector and temporary test
were fully reverted. Evidence is under `iter163` through `iter166` on the pod
and local artifact root.
