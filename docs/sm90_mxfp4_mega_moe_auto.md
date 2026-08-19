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

## Rejected R70-R75: retune Pro M8 ring-poll backoff

R67's fresh matrix leaves Pro M8 `5.95%` behind PR383. Its one-block dispatch
can wrap an L1 ring slot while the previous generation is still live, and the
dispatch lanes then poll the cross-block empty counter. The production path
already sleeps for 64 cycles inside that loop. R70-R75 isolated exact M8 and
swept 16, 128, 256, 512, and 1024 cycles while leaving Pro M16-M64 and all
Flash specializations unchanged. Every header-only variant used a fresh JIT
cache. Exact eight-rank Pro M8 correctness passed at `diff=0.000716`.

The first fresh-cache attempt failed before kernel generation because pod sync
had copied host-absolute CUTLASS/CUTE symlinks. Restoring the pod-local links
made the unchanged test pass; this environment failure is retained in the
R70 artifact directory and is not counted as a kernel failure. The
20-observation, one-warmup, 20-launch, cold-L2 A/B/A screens were:

| M8 sleep cycles | first R67 us | candidate us | change | second R67 us | reverse change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 781.166 | 795.726 | +1.86% | 776.644 | +2.46% |
| 128 | 793.716 | 773.331 | -2.57% | 780.795 | -0.96% |
| 256 | 794.429 | 766.031 | -3.57% | 784.375 | -2.34% |
| 512 | 777.500 | 764.561 | -1.66% | 809.134 | -5.51% |
| 1024 | 792.874 | 765.639 | -3.44% | 778.000 | -1.59% |

Sixteen cycles is insufficient to suppress polling traffic. The 256-1024
cycle results form a noise-sized plateau near `765 us`; 256 has the strongest
two-sided screen lower bound and is the shortest delay, so R75 selected it
for the authoritative 50-observation run:

| point | first R67 us | R75 us | change | second R67 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 | 780.290 | 767.350 | -1.66% | 763.224 | +0.54% |

The formal maximum-rank result changes sign against the faster reverse
control, so the apparent screen gain is not reproducible under the acceptance
contract. The candidate's rank-0 median (`751.060 us`) beats both controls
(`756.952/752.388 us`), but the required maximum-rank score remains mixed.
All polling changes were reverted; R67's 64-cycle path remains production.
Complete logs are under `iter167` through `iter173` on the pod and local
artifact root. The terminal goal remains unmet, and the next structural target
returns to the stable Flash M16 residual instead of further sleep tuning.

## Rejected R76: PRMT pair indexing for regular Pro M512+

R66's bank-permuted regular Pro decoder still expresses the pair index with
two shifts, XOR, mask, and multiply, while selected small-M buckets use one
`PRMT`. R76 extended that PRMT expression to the existing Pro M512+ selector,
without changing the loaded address set, scale lookup, expanded tile, WGMMA,
or epilogue. Exact eight-rank Pro M512 correctness passed at `diff=0.000708`.
Both the eight-rank and matched one-rank cubins remained at 128 registers,
zero stack/local storage, and 100.576 KiB dynamic shared memory.

The matched one-rank/48-expert NCU gate showed that PTXAS had already reduced
the regular-path expression to equivalent machine work:

| metric | R67 bit expression | R76 PRMT source | change |
| --- | ---: | ---: | ---: |
| NCU duration ms | 2.22 | 2.24 | +0.90% |
| executed warp instructions | 513,925,905 | 513,932,492 | +0.0013% |
| executed thread instructions | 16,134,575,840 | 16,134,624,475 | +0.0003% |
| shared-load bank conflicts | 125,498 | 122,844 | -2.11% |
| shared-store bank conflicts | 8,871,142 | 8,835,135 | -0.41% |
| global-load sectors | 4,896,455 | 4,899,161 | +0.06% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The intended instruction reduction is absent and the measured duration moves
in the wrong direction. R76 was therefore stopped at the profiler gate before
distributed timing or NSYS, fully reverted, and is not part of R67. Evidence
is under `iter174` and `iter175` on the pod and local artifact root.

## Rejected R77: pipelined packed-BF16 epilogue for Flash M16

R06's old Flash packed-epilogue prototype regressed M16 while also disabling
the later accepted separate-commit-group weight-half pipeline. R77 tested the
previously uncovered combination: exact routed Flash M16 consumed the packed
BF16 persistent result directly in both epilogues, but retained two compact
weight-half fragments, separate WGMMA commit groups, `wait<1>` overlap, and
the current bank-permuted decoder. Eight-rank correctness passed at
`diff=0.000645`. Both the production and matched one-rank cubins remained at
114 registers, zero stack/local storage, and 110.816 KiB dynamic shared
memory, so the combination avoided R06's historical spill/scheduling issue.

Matched one-rank/32-expert NCU nevertheless showed that compiler motion had
already eliminated nearly all of the intended explicit BF16x2-to-FP32
expansion cost:

| metric | R67 | R77 | change |
| --- | ---: | ---: | ---: |
| NCU duration us | 276.54 | 276.32 | -0.08% |
| executed warp instructions | 72,364,043 | 72,316,358 | -0.066% |
| executed thread instructions | 2,261,797,640 | 2,260,417,397 | -0.061% |
| shared-load bank conflicts | 6,054 | 5,814 | -3.96% |
| shared-store bank conflicts | 1,021,743 | 1,042,610 | +2.04% |
| global-load sectors | 837,015 | 833,381 | -0.43% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

Tensor-pipe activity (`3.94% -> 3.96%`) and barrier/long-/short-scoreboard
ratios (`2.49/2.20/0.58 -> 2.50/2.20/0.58`) are also unchanged. With neither
a material instruction reduction nor a profiler time signal, R77 was stopped
before distributed timing or NSYS and fully reverted. Evidence is under
`iter176` and `iter177` on the pod and local artifact root.

## R78: combine PRMT pair indexing and HFMA2 promotion for Flash M16

### Reason and implementation

Flash M16 remained R67's largest stable deficit at `+9.99%`. R61's PRMT pair
index and R68's packed HFMA2 promotion each removed local instructions but
were individually too small to control the eight-rank maximum. They shorten
independent portions of every K-stage critical path, however: PRMT replaces
the bank-permuted packed-word pair arithmetic, while HFMA2 replaces scalar
unpack, four FP32 FMAs, and repack during accumulator promotion. R78 enables
both existing implementations only for `hidden=4096,
kMaxSwapABTokens=16`. Every other selector remains byte-for-byte unchanged.

Exact eight-rank Flash M16 correctness passes at `diff=0.000654`. The
production and matched one-rank cubins use 115 registers, zero stack/local
storage, and 110.816 KiB dynamic shared memory, compared with R67's 114
registers and otherwise identical resources. The one-register increase does
not alter the two-CTA-per-SM launch topology.

### Screening and formal performance

The one-warmup, 20-observation, 20-launch, cold-L2 screen was double-positive:

| point | first R67 us | R78 us | change | second R67 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 363.168 | 357.418 | -1.58% | 360.297 | -0.80% |

The authoritative 50-observation A/B/A amplified the same direction:

| point | first R67 us | R78 us | change | second R67 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 356.204 | 332.478 | -6.66% | 357.266 | -6.94% |

Rank-0 medians also move from `350.483/336.730 us` to `318.835 us`, so the
accepted maximum-rank gain is not produced by moving one slow peer. All 13
eight-rank production scenarios pass after selection, including every forced
ring-wrap Flash/Pro case.

### NCU and NSYS attribution

Matched one-rank/32-expert profiling confirms that the joint selector crosses
the mechanism threshold missed by either component alone:

| metric | R67 | R78 | change |
| --- | ---: | ---: | ---: |
| NCU duration us | 277.54 | 262.43 | -5.44% |
| executed warp instructions | 72,360,705 | 61,702,788 | -14.73% |
| executed thread instructions | 2,261,903,345 | 1,921,229,504 | -15.06% |
| shared-load bank conflicts | 6,400 | 5,987 | -6.45% |
| shared-store bank conflicts | 1,021,449 | 1,205,729 | +18.04% |
| global-load sectors | 835,059 | 832,670 | -0.29% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| tensor-pipe active | 3.97% | 4.21% | +6.05% |

The shorter decode/promotion body wins despite higher shared-store conflicts
and higher barrier/long-/short-scoreboard ratios
(`2.48/2.19/0.58 -> 2.76/2.50/0.82`). NSYS independently measures
`278.464 -> 263.712 us` (`-5.30%`). Correctness, resource, NCU, NSYS, screen,
formal, and full-production evidence is under `iter178` through `iter182` on
the pod and local artifact root. The isolated R78 point gain is large enough
to erase the prior `+0.296%` 22-point estimate, but a fresh complete PR383
matrix remains the terminal authority.

### R78 authoritative DSV4 matrix against PR383

The fresh same-session run uses the complete requested contract: Flash and
Pro, M=`8,16,32,64,128,256,512,1024,2048,4096,8192`, one warmup, 50
observations for M<=128, three for M>=256, 20 launches per observation,
cold L2, and the maximum-rank median. PR383 is again the sum of its native FP8
L1 and L2 kernels.

| model | M | R78 us | PR383 us | R78 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 309.790 | 294.170 | +5.31% |
| Flash | 16 | 330.545 | 304.229 | +8.65% |
| Flash | 32 | 340.234 | 328.157 | +3.68% |
| Flash | 64 | 371.803 | 370.305 | +0.40% |
| Flash | 128 | 433.155 | 443.427 | -2.32% |
| Flash | 256 | 502.117 | 502.806 | -0.14% |
| Flash | 512 | 903.750 | 941.889 | -4.05% |
| Flash | 1024 | 1493.000 | 1523.372 | -1.99% |
| Flash | 2048 | 2738.000 | 2736.756 | +0.05% |
| Flash | 4096 | 5161.000 | 5079.000 | +1.61% |
| Flash | 8192 | 9943.000 | 9853.000 | +0.91% |
| Pro | 8 | 762.394 | 718.736 | +6.07% |
| Pro | 16 | 993.188 | 1006.552 | -1.33% |
| Pro | 32 | 1032.000 | 1103.037 | -6.44% |
| Pro | 64 | 1070.500 | 1175.370 | -8.92% |
| Pro | 128 | 1218.500 | 1278.790 | -4.71% |
| Pro | 256 | 1650.000 | 1669.937 | -1.19% |
| Pro | 512 | 2569.000 | 2405.394 | +6.80% |
| Pro | 1024 | 3918.000 | 4008.000 | -2.25% |
| Pro | 2048 | 6893.000 | 7014.000 | -1.73% |
| Pro | 4096 | 13001.000 | 12901.000 | +0.78% |
| Pro | 8192 | 25288.000 | 25092.000 | +0.78% |

R78's 22-point geometric gap is `-0.088%`: this is the first complete matrix
on the branch to beat PR383. Small and large subsets lead by `0.113%` and
`0.067%`. Pro leads by `1.207%` overall (`3.202%` small), while Flash trails
by `1.043%` overall (`3.075%` small) despite leading by `0.619%` at large M.
The source-attributed M16 gain is the formal A/B/A above; its fresh PR383 gap
contracts from R67's `+9.99%` to `+8.65%` in this independent matrix. Complete
logs are under `iter183-r78-pr383-full-matrix` on the pod and local artifact
root.

The aggregate target has crossed zero, but the `0.088%` lead is smaller than
observed cross-epoch variance and is not yet treated as robust theoretical
headroom. The next iteration targets the remaining Flash M8/M16 and Pro
M8/M512 residuals while preserving R78's measured M16 selector.

## Rejected R79-R80: incremental WGMMA descriptors for Pro M512+

R79 extended the regular mainloop's existing base-plus-increment WGMMA
descriptor path from large Flash batches to Pro M512 and above. The intended
mechanism was to replace repeated `make_smem_desc` address construction in
each K-stage without changing tile shapes, loads, WGMMA issue order, or the
epilogue. Exact eight-rank Pro M512 correctness passed at `diff=0.000708`.
Both eight-rank and matched one-rank cubins used 128 registers, zero
stack/local storage, and 100.576 KiB dynamic shared memory.

The matched one-rank/48-expert NCU gate confirmed a real instruction-count
reduction, but not a wall-clock reduction:

| metric | R78 | R79 | change |
| --- | ---: | ---: | ---: |
| NCU duration ms | 2.25 | 2.24 | about -0.4% |
| executed warp instructions | 513,923,706 | 507,901,882 | -1.17% |
| executed thread instructions | 16,134,509,700 | 15,941,793,737 | -1.19% |
| shared-load bank conflicts | 123,871 | 120,657 | -2.59% |
| shared-store bank conflicts | 8,878,362 | 9,404,249 | +5.92% |
| global-load sectors | 4,898,617 | 4,901,467 | +0.06% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

Tensor-pipe activity is effectively unchanged (`23.25% -> 23.22%`), while
barrier and long-scoreboard ratios rise slightly (`4.06/2.47% ->
4.14/2.49%`). NSYS measures `2.236093 -> 2.240125 ms` (`+0.18%`), showing
that fewer integer instructions are offset by scheduling or shared-store
pressure.

The five-point, five-observation, one-warmup, 20-launch, cold-L2 A/B/A screen
made the over-broad selector visible:

| Pro M | first R78 us | R79 us | change | second R78 us | reverse change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2576 | 2547 | -1.13% | 2571 | -0.93% |
| 1024 | 3926 | 3926 | 0.00% | 3876 | +1.29% |
| 2048 | 6985 | 6943 | -0.60% | 6917 | +0.38% |
| 4096 | 13017 | 13004 | -0.10% | 13016 | -0.09% |
| 8192 | 25246 | 25317 | +0.28% | 25288 | +0.11% |

The five-point geometric mean is `-0.310%` against the first control but
`+0.148%` against the second. R80 therefore narrowed the selector to exact
Pro M512 and applied the requested authoritative large-M repeat count of
three. The formal maximum-rank medians were `2550 us` (first R78), `2564 us`
(R80), and `2574 us` (second R78): R80 is `+0.55%` slower than the first
control and `-0.39%` faster than the second. This sign reversal fails the
two-sided acceptance contract. Both selectors were fully reverted; R78 is
unchanged. Evidence is under `iter184` through `iter187` on the pod and local
artifact root.

## Rejected R81: single-block owner lookup for mature Flash M16

R81 retested R29's direct nonempty-mask owner lookup at exact Flash M16 on top
of R78's PRMT/HFMA2 mainloop. The scheduler replaces each task's general warp
prefix reconstruction with one multi-block safety ballot followed by a
nonempty mask and `__fns`; any expert owning more than M64 retains the general
fallback. Exact eight-rank correctness passed at `diff=0.000654`. The cubin
remained at 115 registers, zero stack/local storage, and 110.816 KiB dynamic
shared memory.

The 20-observation, one-warmup, 20-launch, cold-L2 screen rejected it before
profiling:

| point | first R78 us | R81 us | change | second R78 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 337.938 | 355.485 | +5.19% | 339.225 | +4.79% |

This reproduces the direction that made the same M16 scheduler selector fail
R29, now without resource or arithmetic confounders. The shorter uniform
lookup does not improve this bucket's dynamic task stream and was fully
reverted. Evidence is under `iter188` and `iter189` on the pod and local
artifact root.

## R82: direct packed-BF16 epilogue on the R78 Flash M16 path

### Reason and implementation

R77 tested the packed swap epilogue before Flash M16 had R78's packed HFMA2
promotion and found no measurable gain. R82 tests the missing composition:
the PRMT decoder and HFMA2 promotion remain unchanged, but the accumulated
BF16 pairs now feed the L1/L2 swap epilogues directly instead of first
expanding all 32 pairs to a 64-float `final_accum` array. The generator selects
this only for routed `hidden=4096, M=16`; the existing Pro M16/M32 packed
selectors and all other specializations are unchanged. The template safety
assertion was generalized from Pro-only to the two supported DSV4 hidden
sizes. Its first stale assertion failure is retained in `iter190`; the fresh
build passes exact eight-rank correctness at `diff=0.000654`.

The production cubin uses 118 registers, zero stack/local storage, and
110.816 KiB dynamic shared memory, compared with R78's 115 registers and
otherwise identical resources. The three-register increase does not alter
the fixed two-CTA-per-SM topology. All 13 eight-rank production scenarios
pass after selection, including every forced ring-wrap case.

### Screening and authoritative performance

The initial 20-observation A/B/A screen was double-positive:

| point | first R78 us | R82 us | change | second R78 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 371.530 | 343.136 | -7.64% | 356.930 | -3.86% |

The requested small-M contract then uses one warmup, 50 observations, 20
launches per observation, cold L2, and maximum-rank median:

| point | first R78 us | R82 us | change | second R78 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 | 337.242 | 331.751 | -1.63% | 348.071 | -4.69% |

Rank-0 medians move from `328.678/328.592 us` to `317.965 us`
(`-3.26%/-3.23%`), confirming that the accepted max-rank improvement is not a
single-peer artifact.

### NCU and NSYS attribution

Matched one-rank/32-expert profiling shows a shorter dependency path rather
than an instruction-count reduction:

| metric | R78 | R82 | change |
| --- | ---: | ---: | ---: |
| NCU duration us | 279.36 | 273.66 | -2.04% |
| executed warp instructions | 61,700,057 | 63,112,924 | +2.29% |
| executed thread instructions | 1,921,278,147 | 1,966,441,116 | +2.35% |
| shared-load bank conflicts | 6,182 | 5,508 | -10.90% |
| shared-store bank conflicts | 1,166,083 | 1,428,718 | +22.52% |
| global-load sectors | 835,041 | 834,062 | -0.12% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |
| tensor-pipe active | 4.24% | 4.34% | +2.36% |

Barrier/long-/short-scoreboard ratios move from `2.74/2.45/0.82` to
`2.73/2.37/0.56`; the short-scoreboard reduction offsets the extra packed
epilogue instructions and shared-store conflicts. NSYS independently
measures `261.632 -> 257.056 us` (`-1.75%`). Correctness, screen, formal,
NCU, and NSYS evidence is under `iter190` through `iter195` on the pod and
local artifact root. R82 requires a fresh complete PR383 matrix before its
aggregate headroom is claimed.

### Fresh PR383 full-matrix comparison

R82 and PR383 were rebuilt and measured back-to-back on the same eight-H20
pod with the fixed production seed and shapes. Small M uses 50 observations;
large M uses three. Each observation contains 20 launches after one warmup,
flushes L2 before every launch, and reports the median of the maximum rank.
Positive gap means R82 is slower than PR383.

| mode | M | R82 us | PR383 us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 321.0955 | 291.2895 | +10.232% |
| Flash | 16 | 325.6950 | 308.2110 | +5.673% |
| Flash | 32 | 344.2885 | 331.5035 | +3.857% |
| Flash | 64 | 368.4330 | 368.1080 | +0.088% |
| Flash | 128 | 429.0355 | 432.0645 | -0.701% |
| Flash | 256 | 512.0040 | 518.9610 | -1.341% |
| Flash | 512 | 890.1730 | 943.1800 | -5.620% |
| Flash | 1024 | 1487.0000 | 1552.6480 | -4.228% |
| Flash | 2048 | 2740.0000 | 2726.3620 | +0.500% |
| Flash | 4096 | 5120.0000 | 5075.0000 | +0.887% |
| Flash | 8192 | 9869.0000 | 9818.0000 | +0.519% |
| Pro | 8 | 745.2310 | 713.5445 | +4.441% |
| Pro | 16 | 997.2045 | 1000.0420 | -0.284% |
| Pro | 32 | 1029.5000 | 1099.8795 | -6.399% |
| Pro | 64 | 1062.5000 | 1155.8095 | -8.073% |
| Pro | 128 | 1213.0000 | 1275.5990 | -4.907% |
| Pro | 256 | 1624.0000 | 1639.6270 | -0.953% |
| Pro | 512 | 2513.0000 | 2432.9410 | +3.291% |
| Pro | 1024 | 3940.0000 | 4062.0000 | -3.003% |
| Pro | 2048 | 6893.0000 | 6990.0000 | -1.388% |
| Pro | 4096 | 12984.0000 | 12924.0000 | +0.464% |
| Pro | 8192 | 25289.0000 | 25104.0000 | +0.737% |

The 22-point geometric mean is `-0.367%`: R82 is faster overall. Flash is
still `+0.810%` slower while Pro is `-1.531%` faster. Small M is `+0.243%`
slower and large M is `-0.873%` faster; split further, Flash small M is the
remaining dominant deficit at `+3.754%`, while Flash large, Pro small, and
Pro large are `-1.580%`, `-3.150%`, and `-0.161%`. This fresh matrix widens
R78's `-0.088%` aggregate lead, validates keeping R82, and identifies Flash
M8/M16/M32 plus Pro M8/M512 as the next optimization targets. Raw evidence is
archived under `iter196-r82-pr383-full-matrix` on the pod and local artifact
root.

## Rejected R83: extend the packed epilogue to Flash M8

R83 tested whether R82's packed-BF16 epilogue benefit transfers to the largest
remaining Flash deficit. The only generator change selected the packed path
for routed `hidden=4096, M=8`; all other specializations remained identical to
R82. Exact eight-rank correctness passed at `diff=0.000671`. The cubin used 118
registers, zero stack/local storage, and 110.816 KiB dynamic shared memory, so
the three-register increase over the unpacked path did not change occupancy.

The 20-observation, one-warmup, 20-launch, cold-L2 A/B/A screen was negative in
both directions:

| point | first R78/R82 us | R83 us | change | second R78/R82 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M8 | 320.1905 | 331.0175 | +3.38% | 326.3920 | +1.42% |

Unlike M16, eliminating the float expansion does not shorten the M8 critical
path. The extra packed-epilogue instructions/registers instead add work to a
bucket dominated by cross-rank tail latency. R83 was fully reverted before
commit. Build, correctness, resource, and screen evidence is archived under
`iter197` and `iter198` on the pod and local artifact root.

## R84: direct source-rank lookup for Flash M8 dispatch pull

### Fresh diagnosis and implementation direction

Fresh matched profiling first separated local compute from distributed tail
latency. One-rank/32-expert NCU measured the unchanged R82 Flash M8 kernel at
`224.064 us`, versus `253.568 us` for the sum of PR383 L1 and L2 (`-11.64%`).
Low-perturbation one-rank NSYS agreed at `208.096 us` versus `245.632 us`
(`-15.28%`). In contrast, rank-0-only tracing in a real eight-rank launch
measured `703.136 us` for R82 and `441.888 + 130.815 = 572.703 us` for PR383
(`+22.78%`). R82 also executes 47.43M warp instructions versus PR383's 38.38M,
but the local-time lead shows that the remaining production deficit is in the
distributed frontend/tail rather than MXFP4 matrix throughput. Complete R82
and PR383 NCU/NSYS evidence is under `iter199`.

The dispatch pull loop previously reconstructed round-robin source ownership
for every received token using warp reductions, division, and a decrementing
loop. At Flash M8, a local expert normally receives at most one token from any
one of the eight source ranks. R84 detects that common case with a ballot and
selects the source directly from the nonempty-rank mask. If any source rank
contributed more than one token to the current expert, the code falls back to
the byte-for-byte general round-robin path. The compile-time guard is exact for
routed `hidden=4096`, swap-AB M8, and at most 32 ranks, so M16+, Pro, shared
experts, and larger-rank configurations retain the old path.

Exact eight-rank Flash M8 correctness passes at `diff=0.000671`. The cubin is
unchanged at 114 registers, zero stack/local storage, and 110.816 KiB dynamic
shared memory. All 13 production scenarios subsequently pass, including every
forced ring-wrap case.

### Screening and formal performance

The 20-observation screen straddled the two controls at max rank but improved
rank 0 against both, so it was escalated rather than accepted:

| point | first R82 us | R84 us | change | second R82 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M8 max rank | 303.4205 | 306.1375 | +0.90% | 314.0120 | -2.51% |
| Flash M8 rank 0 | 291.8550 | 289.7255 | -0.73% | 304.1305 | -4.74% |

The authoritative one-warmup, 50-observation, 20-launch, cold-L2 A/B/A run is
double-positive at both max rank and rank 0:

| point | first R82 us | R84 us | change | second R82 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M8 max rank | 302.1180 | 291.9370 | -3.37% | 311.3570 | -6.24% |
| Flash M8 rank 0 | 297.8795 | 282.6225 | -5.12% | 289.7120 | -2.45% |

One-rank NCU only partially exercises the new common case because duplicate
routes from its sole source rank intentionally select the fallback. It
therefore shows the safety-path cost rather than the eight-rank gain: duration
moves `224.064 -> 223.488 us` (`-0.26%`), warp/thread instructions are flat at
47.43M/1.476B, local traffic remains zero, and registers are unchanged. A real
eight-rank rank-0-only NSYS launch moves `703.136 -> 701.728 us` (`-0.20%`);
the profiler-induced cross-rank wait makes this single launch topology
evidence, while the formal A/B/A above remains the acceptance authority.

A same-session PR383/R84/PR383 50-observation comparison measures
`297.2525/315.0020/299.6185 us`. R84 remains `5.97%/5.13%` behind PR383 at
Flash M8, but approximately halves R82's fresh `+10.23%` full-matrix deficit.
The remaining M8 gap is still distributed synchronization/tail latency. Gate,
screen, formal, profiler, full-production, and PR383 evidence is archived
under `iter200` through `iter205` on the pod and local artifact root.

### R84 full Flash/Pro matrix versus PR383

The accepted R84 source-rank change was then measured over the complete frozen
comparison matrix: DSV4 Flash and Pro, M=`8..8192`, one warmup, 50 observations
through M128 and three observations from M256, 20 launches per observation,
cold L2, seed 0, and max-rank median. PR383 uses the same script's built-in one
warmup and its native L1+L2 sum. Both sides ran serially on the same eight-H20
pod. All 22 summaries were present.

| model | M | R84 us | PR383 us | R84 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 319.7965 | 292.1045 | +9.480% |
| Flash | 16 | 330.8025 | 307.7040 | +7.507% |
| Flash | 32 | 341.6230 | 329.5740 | +3.656% |
| Flash | 64 | 366.3955 | 367.0040 | -0.166% |
| Flash | 128 | 427.6085 | 434.9930 | -1.698% |
| Flash | 256 | 506.6590 | 503.6590 | +0.596% |
| Flash | 512 | 873.6950 | 927.7890 | -5.830% |
| Flash | 1024 | 1478.0000 | 1542.8360 | -4.202% |
| Flash | 2048 | 2746.0000 | 2745.7520 | +0.009% |
| Flash | 4096 | 5120.0000 | 5056.0000 | +1.266% |
| Flash | 8192 | 9895.0000 | 9830.0000 | +0.661% |
| Pro | 8 | 751.2215 | 718.9905 | +4.483% |
| Pro | 16 | 986.4710 | 1012.6985 | -2.590% |
| Pro | 32 | 1021.0000 | 1112.8290 | -8.252% |
| Pro | 64 | 1065.0000 | 1159.6160 | -8.159% |
| Pro | 128 | 1214.0000 | 1281.9265 | -5.299% |
| Pro | 256 | 1619.0000 | 1647.6390 | -1.738% |
| Pro | 512 | 2510.0000 | 2396.7040 | +4.727% |
| Pro | 1024 | 3886.0000 | 4004.0000 | -2.947% |
| Pro | 2048 | 6900.0000 | 7030.0000 | -1.849% |
| Pro | 4096 | 12964.0000 | 12883.0000 | +0.629% |
| Pro | 8192 | 25286.0000 | 25123.0000 | +0.649% |

| geometric-mean slice | R84 gap versus PR383 |
| --- | ---: |
| all 22 points | -0.512243% |
| Flash | +0.934110% |
| Pro | -1.937870% |
| small M (`<=128`) | -0.279466% |
| large M (`>=256`) | -0.705809% |
| Flash small M | +3.667286% |
| Flash large M | -1.288411% |
| Pro small M | -4.075961% |
| Pro large M | -0.119768% |

R84 therefore widens the all-point lead from R82's `-0.367338%` to
`-0.512243%`, while the session-to-session noise in untouched points prevents
attributing the full `0.145` percentage-point change solely to M8. The priority
remains Flash small M: M8 and M16 are the two largest current deficits. Raw
logs are archived in `iter206-r84-pr383-full-matrix` on the pod and local
artifact root.

## R85: extend direct source-rank lookup to Flash M16

### Reason and direction

R84's full matrix left Flash M16 `7.507%` behind PR383, the second-largest
Flash small-M deficit. With 16 tokens per rank, a particular expert still
usually receives no more than one route from each source rank. R85 therefore
extends R84's exact compile-time guard from `kMaxSwapABTokens == 8` to
`kMaxSwapABTokens == 8 or 16`. The runtime ballot is unchanged: it selects the
source directly only when every source count is at most one, and otherwise
falls back to the original round-robin reconstruction. Pro and Flash M32+
remain byte-for-byte on their old compile-time path.

The exact eight-rank forced-wrap Flash M16 gate passes at `diff=0.000654`.
Resource use is unchanged at 118 registers, zero stack/local storage, 1024
bytes static shared memory, and 110.816 KiB dynamic shared memory.

### Screening and formal acceptance

The initial 20-observation cold-L2 screen straddled the two R84 controls, so it
was escalated rather than judged from one side:

| point | first R84 us | R85 us | change | second R84 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 max rank | 373.9255 | 361.4620 | -3.33% | 345.0805 | +4.75% |
| Flash M16 rank 0 | 354.8805 | 343.5240 | -3.20% | 328.0350 | +4.72% |

The authoritative 50-observation A/B/A is double-positive at max rank and
rank 0:

| point | first R84 us | R85 us | change | second R84 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 max rank | 361.9330 | 326.3015 | -9.84% | 332.2255 | -1.78% |
| Flash M16 rank 0 | 340.8110 | 313.3395 | -8.06% | 321.9545 | -2.68% |

All 13 production scenarios pass, including every forced ring-wrap case.

### Profiler attribution and PR383 comparison

The one-rank/32-expert NCU comparison intentionally exercises the general
fallback because one source rank contributes duplicate routes. R84/R85 move
`273.15 -> 274.24 us` (`+0.40%`), with warp instructions
`63,116,003 -> 63,121,399`, thread instructions
`1,966,339,347 -> 1,966,491,432`, 118 registers, and zero local traffic on
both. Shared-load/store bank conflicts move `6,214/1,450,083` to
`5,843/1,434,713`, global-load sectors `831,516 -> 828,927`, and global-store
sectors stay `29,369`. The essentially flat local profile is expected: the
production gain comes from the eight-source common case, not matrix math.

Low-perturbation eight-rank rank-0-only NSYS records one instrumented launch
at `644.192 us` for R84 and `664.576 us` for R85 (`+3.16%`). As in earlier
iterations, tracing only one participant perturbs cooperative wait time and a
single launch is not an acceptance statistic. It is retained as topology
evidence; the 50-observation A/B/A remains authoritative.

A same-session PR383/R85/PR383 50-observation comparison measures max-rank
medians `333.2465/346.5730/322.9320 us`; R85 is still `4.00%/7.32%` behind
PR383. Rank-0 medians are `324.3175/336.5480/303.6290 us`, leaving
`3.77%/10.84%`. Thus R85 is accepted because it is double-positive against
its frozen R84 parent, but it does not close Flash M16 outright. Build,
correctness, screen, formal, NCU, NSYS, full-production, and PR383 evidence is
archived under `iter207` through `iter212` on the pod and local artifact root.

### R85 full Flash/Pro matrix versus PR383

The committed R85 kernel was remeasured over the complete frozen 22-point
matrix, followed immediately by PR383 on the same pod. Both logs contain all
expected summaries.

| model | M | R85 us | PR383 us | R85 gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 305.4725 | 305.1085 | +0.119% |
| Flash | 16 | 324.1490 | 311.0770 | +4.202% |
| Flash | 32 | 337.6540 | 325.3660 | +3.777% |
| Flash | 64 | 360.9805 | 361.6625 | -0.189% |
| Flash | 128 | 438.9910 | 429.8255 | +2.132% |
| Flash | 256 | 489.7440 | 503.8010 | -2.790% |
| Flash | 512 | 922.5290 | 900.8760 | +2.404% |
| Flash | 1024 | 1495.0000 | 1509.8240 | -0.982% |
| Flash | 2048 | 2728.0000 | 2745.7360 | -0.646% |
| Flash | 4096 | 5143.0000 | 5053.0000 | +1.781% |
| Flash | 8192 | 9896.0000 | 9875.0000 | +0.213% |
| Pro | 8 | 759.7415 | 715.6680 | +6.158% |
| Pro | 16 | 999.9960 | 1004.3380 | -0.432% |
| Pro | 32 | 1031.0000 | 1100.8815 | -6.348% |
| Pro | 64 | 1063.5000 | 1156.4840 | -8.040% |
| Pro | 128 | 1218.5000 | 1281.8180 | -4.940% |
| Pro | 256 | 1621.0000 | 1632.3780 | -0.697% |
| Pro | 512 | 2504.0000 | 2402.4260 | +4.228% |
| Pro | 1024 | 3882.0000 | 4019.0000 | -3.409% |
| Pro | 2048 | 6879.0000 | 7026.0000 | -2.092% |
| Pro | 4096 | 12966.0000 | 12892.0000 | +0.574% |
| Pro | 8192 | 25288.0000 | 25096.0000 | +0.765% |

| geometric-mean slice | R85 gap versus PR383 |
| --- | ---: |
| all 22 points | -0.250478% |
| Flash | +0.890508% |
| Pro | -1.378560% |
| small M (`<=128`) | -0.458868% |
| large M (`>=256`) | -0.076487% |
| Flash small M | +1.992352% |
| Flash large M | -0.018597% |
| Pro small M | -2.851176% |
| Pro large M | -0.134343% |

R85 remains ahead in the all-point and small-M aggregates. The all-point lead
is smaller than R84's prior-session `-0.512243%`, but untouched three-sample
large-M points moved by several percent in both directions, so the difference
cannot be attributed to the M16-only change. The reliable next target is the
50-observation Pro M8 deficit at `+6.158%`; raw evidence is archived in
`iter213-r85-pr383-full-matrix`.

## Rejected R86: direct source-rank lookup for Pro M8

R86 tested R84's single-slot source-rank selection on exact routed Pro M8,
which was the largest 50-observation deficit in the R85 full matrix at
`+6.158%`. Pro M8 has even fewer routes per source-rank/expert pair than Flash
M8, so the runtime uniqueness condition is commonly true. The experiment kept
the duplicate-route fallback unchanged and did not affect Pro M16+ or any
Flash specialization.

Exact eight-rank correctness passed at `diff=0.000716`. The cubin used 107
registers, zero stack/local storage, and 1024 bytes static shared memory. The
20-observation cold-L2 A/B/A screen was nevertheless double-negative:

| point | first R85 us | R86 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 767.5590 | 775.2045 | +1.00% | 766.1460 | +1.18% |
| Pro M8 rank 0 | 761.2575 | 771.7005 | +1.37% | 764.8755 | +0.89% |

The common-case ballot and branch cost more than the eliminated round-robin
work on this longer Pro kernel. R86 was reverted in full and not committed.
Gate and screening evidence is archived under `iter214` and `iter215` on the
pod and local artifact root.

## Rejected R87: two-layer direct source lookup for Flash M32

R87 generalized the direct source selector to Flash M32 without using R84's
too-narrow all-counts-at-most-one condition. If every source-rank/expert count
was at most two, it selected first-round tokens from the nonempty mask and
second-round tokens from the count-greater-than-one mask. Any count above two
fell back to the original loop. This was expected to cover roughly 70% of
experts at M32 while leaving M8/M16 unchanged.

Forced-wrap correctness passed at `diff=0.000656`. Resources stayed at 125
registers, zero stack/local storage, and 1024 bytes static shared memory. The
20-observation screen was double-positive:

| point | first R85 us | R87 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M32 max rank | 365.0535 | 361.7645 | -0.90% | 363.1140 | -0.37% |
| Flash M32 rank 0 | 347.9640 | 341.1055 | -1.97% | 352.8795 | -3.34% |

The required 50-observation A/B/A reversed the result:

| point | first R85 us | R87 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M32 max rank | 364.0560 | 368.2355 | +1.15% | 353.1875 | +4.26% |
| Flash M32 rank 0 | 351.4065 | 354.5830 | +0.90% | 325.4075 | +8.97% |

The additional ballots, popcount, second-round branch, and frequent fallback
do not amortize at M32. R87 was reverted in full and not committed. Evidence
is archived under `iter216` through `iter218`.

## Rejected R88: two-layer direct source lookup for Flash M16

R88 narrowed R87's two-layer idea to exact Flash M16, where the probability
that every source-rank/expert count is at most two is roughly 95%. M8 retained
R84's single-slot selector and M32 retained R85. The second layer avoided the
general round-robin loop when one source contributed exactly two routes.

Forced-wrap correctness passed at `diff=0.000654`; resources remained 118
registers, zero stack/local storage, and 1024 bytes static shared memory. The
20-observation screen straddled the two controls:

| point | first R85 us | R88 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 max rank | 341.2815 | 334.0850 | -2.11% | 331.2810 | +0.85% |
| Flash M16 rank 0 | 322.7765 | 315.7180 | -2.19% | 313.4745 | +0.72% |

The 50-observation A/B/A was double-negative:

| point | first R85 us | R88 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M16 max rank | 336.0085 | 346.9630 | +3.26% | 336.5130 | +3.11% |
| Flash M16 rank 0 | 319.5940 | 329.9055 | +3.23% | 309.8935 | +6.46% |

Even at high count-at-most-two coverage, the extra popcount and per-token
round branch cost more than retaining the R85 fallback for duplicate routes.
R88 was reverted in full and not committed. Evidence is archived under
`iter219` through `iter221`.

## R85 fresh Pro M8 profile versus PR383

R85's complete matrix left exact Pro M8 `+6.158%` behind PR383 under the
authoritative eight-rank, 50-observation, 20-launch, cold-L2 maximum-rank
contract. A fresh matched one-rank profile keeps the production shard size at
48 experts and removes distributed replay noise. The PR383 values below sum
its native FP8 L1 and L2 kernels:

| metric | R85 fused | PR383 L1+L2 | R85 excess |
| --- | ---: | ---: | ---: |
| NCU duration us | 715.04 | 707.74 | +1.03% |
| executed warp instructions | 172,497,197 | 125,804,562 | +37.11% |
| executed thread instructions | 5,380,206,390 | 3,753,162,184 | +43.35% |
| global-load sectors | 2,202,710 | 1,736,854 | +26.82% |
| shared-load bank conflicts | 7,864 | 27,895 | -71.81% |
| shared-store bank conflicts | 3,843,854 | 30,741 | +12,404.34% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The remaining local latency gap is only about one percent despite the much
larger instruction and load counts, so the six-percent production residual is
not explained by one-rank arithmetic alone. Rank-0-only eight-rank NSYS is
kept as qualitative evidence: R85 records one `1176.703 us` fused kernel,
whereas PR383 records `689.344 + 624.223 = 1313.567 us`. The profiler changes
cross-rank arrival and even makes the fused compute interval appear faster;
it is therefore not used for acceptance. Reports and traces are under
`iter222-r85-pr383-pro-m8-profiles`.

## Rejected R89: sparse dispatch completion for Pro M8

R89 tested the existing sparse dispatch-completion protocol at exact Pro M8.
The dense path issues one completion atomic for every expert from every CTA,
including zero local counts. The sparse path issues only nonzero-count
atomics, uses the existing grid/NVLink rendezvous for completion, then lets
SM0 aggregate the eight rank-local counts. Flash M32 and M1024 already use
this protocol, but Pro M8 had not been tested. The host selector was exact, so
the other 21 authoritative points were unchanged.

The first screen accidentally used the stale pre-R89 Python extension. Its
generated `kernel.cu` did not contain `DG_SM90_SPARSE_DISPATCH_COMPLETION`, so
the apparent `2.79%/1.74%` improvement in `iter223` and the matching formal
run in `iter224` are explicitly invalid and retained only as an environment
audit. The old extension was then copied to an immutable R85 control path,
the candidate extension was rebuilt, and a fresh JIT cache verified the macro
in generated source. Exact eight-rank correctness passed at `diff=0.000716`.

The valid 20-observation screen was double-positive at maximum rank:

| point | first R85 us | R89 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 774.1740 | 770.2425 | -0.51% | 799.1375 | -3.62% |
| Pro M8 rank 0 | 759.5080 | 768.5690 | +1.19% | 784.2485 | -2.00% |

The authoritative 50-observation A/B/A did not retain the screen gain:

| point | first R85 us | R89 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 765.2960 | 765.5465 | +0.033% | 765.5160 | +0.004% |
| Pro M8 rank 0 | 746.0815 | 750.4770 | +0.59% | 744.7555 | +0.77% |

Matched one-rank NCU confirms that the intended atomic reduction is real but
offset elsewhere:

| metric | R85 dense | R89 sparse | change |
| --- | ---: | ---: | ---: |
| NCU duration us | 714.59 | 719.90 | +0.74% |
| L1 global atomic sectors | 6,408 | 4,536 | -29.21% |
| L2 atomic sectors | 9,359 | 6,631 | -29.15% |
| global-load sectors | 2,204,909 | 2,215,307 | +0.47% |
| executed warp instructions | 172,492,160 | 172,491,772 | -0.0002% |
| executed thread instructions | 5,380,201,593 | 5,380,115,550 | -0.0016% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The added count aggregation and rendezvous loads consume the atomic saving;
the formal maximum-rank result is exactly flat and rank 0 regresses against
both controls. R89 was reverted in source and the candidate extension was
rebuilt byte-identical to the frozen R85 extension. Valid evidence is under
`iter226` through `iter229`; R85 remains the accepted implementation.

## Rejected R90: back off the final Pro M8 NVLink barrier

Fresh R85 Pro M8 SourceCounters localized the largest sampled stall to the
dispatch/epilogue rendezvous immediately before workspace cleanup and the
final cross-rank barrier. Decoder shared-load replay remained negligible; the
largest dynamic opcode groups were the intrinsic MXFP4 conversion operations
(`27.32M` LOP3 and `21.38M` PRMT warp instructions). R90 therefore left the
math and dispatch paths unchanged and added a 64-cycle sleep only while SM0
polled the final workspace-clean NVLink signal at eight-rank Pro M8. All other
NVLink barriers and specializations compiled with no sleep.

Exact eight-rank correctness passed at `diff=0.000716`. The cubin retained 107
registers/thread, zero stack/local allocation, and 1024 bytes static shared
memory. The 20-observation cold-L2 A/B/A screen was double-positive:

| point | first R85 us | R90 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 793.9675 | 760.4100 | -4.23% | 774.7930 | -1.86% |
| Pro M8 rank 0 | 771.7615 | 748.2370 | -3.05% | 759.7130 | -1.51% |

The authoritative 50-observation run reversed the result:

| point | first R85 us | R90 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 761.5280 | 776.8300 | +2.01% | 769.2520 | +0.99% |
| Pro M8 rank 0 | 740.1605 | 761.7805 | +2.92% | 747.2055 | +1.95% |

The final barrier benefits from detecting the last rank immediately; reducing
poll traffic does not repay the added detection latency. R90 was reverted in
full, and further sleep-value sweeps at this barrier are not justified.
SourceCounters, gate, screen, and formal evidence are under `iter230` through
`iter233`; R85 remains the accepted implementation.

## Rejected R91: direct packed-BF16 epilogue for Flash M32

R91 extended the direct packed-BF16 swap epilogue already retained at Flash
M16 and Pro M16/M32 to exact Flash M32. The candidate consumed the 32 packed
BF16x2 persistent values directly instead of expanding them into a 64-float
temporary before the L1/L2 epilogue. No decoder, WGMMA, dispatch, or other
specialization changed.

Forced-ring-wrap correctness passed at `diff=0.000656`; the cubin retained 125
registers/thread, zero stack/local allocation, and 1024 bytes static shared
memory. The 20-observation screen was double-positive:

| point | first R85 us | R91 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M32 max rank | 361.7120 | 351.7945 | -2.74% | 361.2450 | -2.62% |
| Flash M32 rank 0 | 342.8745 | 333.6985 | -2.68% | 346.6350 | -3.73% |

The first authoritative 50-observation A/B/A did not reproduce the result:

| point | first R85 us | R91 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M32 max rank | 370.8065 | 360.4900 | -2.78% | 347.8355 | +3.64% |
| Flash M32 rank 0 | 348.4845 | 333.7745 | -4.22% | 341.3965 | -2.23% |

Matched one-rank NCU then rejected the intended instruction reduction:

| metric | R85 | R91 | change |
| --- | ---: | ---: | ---: |
| duration us | 301.73 | 301.79 | +0.02% |
| executed warp instructions | 72,077,051 | 73,466,457 | +1.93% |
| executed thread instructions | 2,251,467,356 | 2,295,700,615 | +1.96% |
| shared-store bank conflicts | 1,068,364 | 1,181,304 | +10.57% |
| global-load sectors | 897,572 | 894,455 | -0.35% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The direct path increases dynamic epilogue work at M32 and has no isolated
latency benefit. The screen was distributed drift, so no third formal run was
performed. R91 was reverted and the candidate extension rebuilt to R85.
Correctness, screen, formal, and NCU evidence are under `iter234` through
`iter237`.

## R92: combine packed HFMA2 promotion and direct epilogue for Flash M32

### Reason and direction

R36's broad Flash experiment found that packed BF16 `HFMA2` promotion at M32
was mixed, and R91 found that the direct packed-BF16 epilogue was also mixed
when enabled alone. Flash M16 established an important counterexample in R78
and R82: shortening the mainloop promotion and keeping its packed result
through the epilogue changed the compiler schedule enough to pass even though
the components had not won independently. R92 tests that missing composition
at exact routed `hidden=4096, M=32`. It enables `HFMA2` promotion only when the
M32 packed epilogue selector is also true, so the other 21 matrix points retain
their R85 compile-time paths.

Eight-rank forced-ring-wrap correctness passes at `diff=0.000666`. All 13
production correctness scenarios pass after selection. The Flash M32 cubin
uses 127 registers/thread, zero stack/local allocation, 1024 bytes static
shared memory, and 110.82 KiB dynamic shared memory. R85 uses 125 registers
with otherwise identical resources; the fixed two-CTA-per-SM launch topology
is unchanged.

### Screening and authoritative performance

The one-warmup, 20-observation, 20-launch, cold-L2 screen was double-positive:

| point | first R85 us | R92 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M32 max rank | 360.3005 | 348.9885 | -3.14% | 358.6925 | -2.71% |
| Flash M32 rank 0 | 345.2500 | 333.6340 | -3.36% | 349.6485 | -4.58% |

The requested small-M contract then used 50 observations with all other
settings unchanged. Maximum rank reproduced the gain against both controls:

| point | first R85 us | R92 us | change | second R85 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M32 max rank | 347.2895 | 343.6520 | -1.05% | 351.2080 | -2.15% |
| Flash M32 rank 0 | 340.4110 | 330.3350 | -2.96% | 329.7520 | +0.18% |

Maximum-rank median is the acceptance metric. Its two-sided result, the
double-positive screen, and the independent profiler directions justify
retaining R92. Rank 0 is essentially flat against the second formal control;
the production benefit is primarily a reduction in the slowest peer.

### NCU and NSYS attribution

Matched one-rank/32-expert NCU isolates the generated M32 kernel:

| metric | R85 | R92 | change |
| --- | ---: | ---: | ---: |
| duration us | 277.15 | 272.93 | -1.52% |
| executed warp instructions | 72,062,450 | 69,312,840 | -3.82% |
| executed thread instructions | 2,251,253,568 | 2,162,917,053 | -3.92% |
| shared-load bank conflicts | 7,885 | 6,844 | -13.20% |
| shared-store bank conflicts | 1,134,246 | 1,478,216 | +30.33% |
| global-load sectors | 895,948 | 895,857 | -0.01% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The composition wins by shortening the promotion/epilogue instruction and
dependency path, not by reducing global traffic. It tolerates two additional
registers and more shared-store conflicts without introducing local memory.
A low-perturbation eight-rank NSYS run traced rank 0 only while the other seven
ranks ran normally. Its single fused-kernel duration moves from `587.072 us`
to `575.904 us` (`-1.90%`). The absolute profiler time is not a benchmark
score, but its direction agrees with NCU and formal timing. Gate, screen, NCU,
formal, production-correctness, and NSYS evidence is archived under `iter238`
through `iter243`.

### R92 authoritative DSV4 matrix against PR383

R92 and PR383 were measured back-to-back on the same eight-H20 pod with the
full requested contract. PR383 is the sum of its native FP8 L1 and L2 kernels.
Positive gap means R92 is slower.

| model | M | R92 us | PR383 us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 297.333 | 299.096 | -0.59% |
| Flash | 16 | 332.1845 | 307.800 | +7.92% |
| Flash | 32 | 347.338 | 333.132 | +4.26% |
| Flash | 64 | 371.122 | 364.1585 | +1.91% |
| Flash | 128 | 444.733 | 436.578 | +1.87% |
| Flash | 256 | 514.458 | 505.204 | +1.83% |
| Flash | 512 | 914.715 | 911.749 | +0.33% |
| Flash | 1024 | 1522.000 | 1525.493 | -0.23% |
| Flash | 2048 | 2798.000 | 2702.674 | +3.53% |
| Flash | 4096 | 5169.000 | 5062.000 | +2.11% |
| Flash | 8192 | 9895.000 | 9854.000 | +0.42% |
| Pro | 8 | 754.444 | 707.3405 | +6.66% |
| Pro | 16 | 999.932 | 1007.8985 | -0.79% |
| Pro | 32 | 1035.500 | 1106.4865 | -6.42% |
| Pro | 64 | 1062.000 | 1163.8855 | -8.75% |
| Pro | 128 | 1216.500 | 1280.468 | -5.00% |
| Pro | 256 | 1620.000 | 1639.099 | -1.17% |
| Pro | 512 | 2542.000 | 2407.399 | +5.59% |
| Pro | 1024 | 3934.000 | 4031.000 | -2.41% |
| Pro | 2048 | 6912.000 | 7029.000 | -1.66% |
| Pro | 4096 | 12996.000 | 12907.000 | +0.69% |
| Pro | 8192 | 25303.000 | 25102.000 | +0.80% |

The 22-point geometric gap is `+0.419673%`. Flash trails by `+2.097853%`,
while Pro leads by `-1.230924%`. Small M is effectively tied at `-0.030925%`;
large M trails by `+0.796722%`. The more detailed splits are Flash small
`+3.035906%`, Flash large `+1.322670%`, Pro small `-3.006472%`, and Pro large
`+0.273503%`.

The full-matrix snapshot is noisier than the exact causal A/B/A, especially
at three-observation large M, and does not invalidate R92's isolated M32 win.
It does show that the terminal goal is not yet met: the remaining stable
priorities are Flash M16/M32 and Pro M8, with Pro M512 as a secondary
three-observation residual. Complete matrix logs are under
`iter244-r92-pr383-full-matrix`.

## R93: single-slot source-rank lookup for Flash M32

### Reason and direction

R92 shortened Flash M32's packed promotion and epilogue but still left the
full-matrix point `+4.26%` behind PR383. R87's two-layer source selector had
tested counts up to two and lost after adding a second ballot/popcount path.
The cheaper R84/R85 single-layer selector had never been measured at M32: it
uses a nonempty-rank mask only when every source-rank/expert count is at most
one, and otherwise falls back directly to the original round-robin loop. R93
extends that exact selector from Flash M8/M16 to routed Flash M32 while
retaining R92's packed HFMA2/epilogue composition. Pro and every other Flash
bucket are compile-time unchanged.

Eight-rank forced-ring-wrap correctness passes at `diff=0.000666`, and all 13
production correctness scenarios pass. The cubin remains at 127
registers/thread, zero stack/local allocation, 1024 bytes static shared
memory, and 110.82 KiB dynamic shared memory, identical to R92.

### Screening and authoritative performance

The 20-observation cold-L2 R92/R93/R92 screen was strongly double-positive:

| point | first R92 us | R93 us | change | second R92 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M32 max rank | 363.3525 | 342.7040 | -5.68% | 357.4405 | -4.12% |
| Flash M32 rank 0 | 351.4285 | 323.1990 | -8.03% | 348.6520 | -7.30% |

The requested 50-observation, 20-launch, one-warmup formal run retained the
maximum-rank gain against both controls:

| point | first R92 us | R93 us | change | second R92 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M32 max rank | 343.7260 | 339.5325 | -1.22% | 347.3190 | -2.24% |
| Flash M32 rank 0 | 325.9145 | 328.6080 | +0.83% | 343.5265 | -4.34% |

Maximum rank is the acceptance metric and is double-positive. Rank 0 straddles
the controls, so the retained improvement is again primarily a reduction in
the slowest peer rather than a uniform local arithmetic speedup.

### NCU and NSYS attribution

The selector depends on the production eight-rank count distribution, so a
matched one-rank profile would exercise its fallback and cannot attribute the
change. Eight concurrent application-replay NCU profilers were attempted in
`iter248`, but replay phase skew made the result invalid: rank-0 R93 expanded
to `129.48 ms` and 223.58M global-load sectors, versus R92's already-perturbed
`3.29 ms` and 1.47M sectors. These values record cross-rank polling during
desynchronized replay, not kernel work.

`iter249` therefore used one hardware pass and one application execution per
rank for duration and instruction counters. NCU still serializes and perturbs
the peers heavily: rank durations span `2.85-383.46 ms` for R92 and
`11.22-334.43 ms` for R93. Under that limitation, the maximum duration falls
12.79%, while the sums of executed warp and thread instructions fall 17.76%
and 16.34%. This is qualitative evidence that the selector does not add work
to the distributed wait path, but it is not used as a speedup estimate.

Low-perturbation eight-rank NSYS traced rank 0 while the other seven ranks ran
normally. Its single fused-kernel duration moves from `585.471 us` to
`578.848 us` (`-1.13%`), agreeing with the authoritative A/B/A. Gate, screen,
formal, both NCU attempts, NSYS, and production-correctness evidence is under
`iter245` through `iter251`.

### R93 authoritative DSV4 matrix against PR383

R93 and PR383 were rebuilt/measured back-to-back on the same pod with the full
requested contract. Positive gap means R93 is slower.

| model | M | R93 us | PR383 us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 298.706 | 327.4525 | -8.78% |
| Flash | 16 | 318.7405 | 315.042 | +1.17% |
| Flash | 32 | 330.778 | 335.1275 | -1.30% |
| Flash | 64 | 366.0265 | 362.434 | +0.99% |
| Flash | 128 | 422.860 | 436.070 | -3.03% |
| Flash | 256 | 503.820 | 499.788 | +0.81% |
| Flash | 512 | 910.766 | 926.596 | -1.71% |
| Flash | 1024 | 1476.000 | 1519.726 | -2.88% |
| Flash | 2048 | 2722.000 | 2729.932 | -0.29% |
| Flash | 4096 | 5105.000 | 5065.000 | +0.79% |
| Flash | 8192 | 9890.000 | 9823.000 | +0.68% |
| Pro | 8 | 759.5385 | 712.3185 | +6.63% |
| Pro | 16 | 995.926 | 1009.1005 | -1.31% |
| Pro | 32 | 1034.000 | 1105.7015 | -6.48% |
| Pro | 64 | 1067.000 | 1152.313 | -7.40% |
| Pro | 128 | 1222.000 | 1269.3115 | -3.73% |
| Pro | 256 | 1617.000 | 1630.777 | -0.84% |
| Pro | 512 | 2515.000 | 2403.002 | +4.66% |
| Pro | 1024 | 3900.000 | 4011.000 | -2.77% |
| Pro | 2048 | 6909.000 | 7024.000 | -1.64% |
| Pro | 4096 | 12986.000 | 12899.000 | +0.67% |
| Pro | 8192 | 25274.000 | 25096.000 | +0.71% |

R93 leads PR383 by `-1.199250%` across all 22 points. Flash and Pro lead by
`-1.271938%` and `-1.126509%`; small and large M lead by `-2.421019%` and
`-0.169431%`. The detailed splits are Flash small `-2.257601%`, Flash large
`-0.442963%`, Pro small `-2.584164%`, and Pro large `+0.104852%`.

This matrix crosses the terminal comparison target and changes Flash M32 from
R92's prior `+4.26%` snapshot to `-1.30%`. The unusually large Flash M8 lead
also reflects current-run PR383 drift (`327.4525 us` versus `299.096 us` in
R92's matrix), so only the interleaved R92/R93/R92 M32 result is attributed
causally to R93. The remaining stable optimization priority is Pro M8
(`+6.63%`), followed by Pro M512 and the near-flat large-M residuals. Complete
matrix logs are under `iter252-r93-pr383-full-matrix`.

## R94 rejected: packed BF16 epilogue for Pro M8

### Reason and direction

Pro M8 is R93's largest stable residual against PR383 at `+6.63%`. R48 had
already enabled the packed HFMA2 promotion path for this exact bucket, while
the direct packed BF16 epilogue remained restricted to Pro M16/M32. R94
temporarily composed the two existing optimizations by extending only the
host selector to Pro M8; no generated-kernel code was otherwise changed.

Forced-ring-wrap correctness passes at `diff=0.000716`. The cubin remains at
107 registers/thread, zero stack/local allocation, 1024 bytes static shared
memory, and 100.58 KiB dynamic shared memory, identical to R93.

### Performance and rejection decision

The initial 20-observation cold-L2 R93/R94/R93 screen was double-positive:

| point | first R93 us | R94 us | change | second R93 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 797.7555 | 768.6130 | -3.65% | 780.3205 | -1.50% |
| Pro M8 rank 0 | 779.7565 | 747.1730 | -4.18% | 766.1605 | -2.48% |

The requested 50-observation, 20-launch, one-warmup formal R93/R94/R93 run
reduced the apparent gain to the noise floor:

| point | first R93 us | R94 us | change | second R93 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 761.9920 | 761.3530 | -0.08% | 767.6480 | -0.82% |
| Pro M8 rank 0 | 745.8320 | 744.0530 | -0.24% | 745.0300 | -0.13% |

An independent reverse-order R94/R93/R94 formal repeat invalidated the small
positive direction. The first and second R94 maximum-rank samples were
`753.9870 us` and `761.8690 us`, versus R93 at `759.6325 us`: respectively
`-0.74%` and `+0.29%`. Rank 0 was worse for both R94 samples at `742.9725 us`
and `745.6970 us`, versus R93 at `740.1510 us` (`+0.38%` and `+0.75%`). The
sign flip means the selector is not retained.

One-rank NCU shows why this composition is weak: duration moves only from
`658.85 us` to `653.22 us` (`-0.85%`). Shared-load conflicts fall 14.20%, but
shared-store conflicts rise 6.18%; executed warp and thread instructions rise
2.90% and 2.95%, global traffic is flat, and local traffic remains zero. A
low-perturbation NSYS run moves from `1181.121 us` to `1164.064 us`
(`-1.44%`), but profiler direction cannot override the reproducible formal
sign reversal. R94 is therefore rejected and the candidate is restored to
R93. Gate, screen, NCU, formal, NSYS, and reverse-order evidence is archived
under `iter253` through `iter258`.

## R95 rejected: active-range ring-counter cleanup for Pro M8

### Reason and direction

Production Pro uses a capacity-sized `57344`-token physical ring, or 7168
minimum-sized counter slots. The tail cleanup made SM0 clear all four ring
counter arrays on every launch even though Pro M8 touches only the contiguous
prefix occupied by its current routed pool. R95 temporarily changed exact
Pro M8 to clear
`min(scheduler.get_num_total_pool_blocks(), workspace.num_ring_blocks)`.
If the live set wrapped the ring, the expression still selected the complete
physical capacity; synchronization and data-buffer reuse were unchanged.

Eight-rank production correctness passes at `diff=0.000716`. The cubin stays
at 107 registers/thread with zero stack/local allocation, 1024 bytes static
shared memory, and 100.58 KiB dynamic shared memory.

### Performance and rejection decision

The 20-observation cold-L2 R93/R95/R93 screen changed sign against the second
control:

| point | first R93 us | R95 us | change | second R93 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 789.1460 | 774.1510 | -1.90% | 767.8765 | +0.82% |
| Pro M8 rank 0 | 762.9800 | 754.5770 | -1.10% | 747.0820 | +1.00% |

Matched one-rank NCU confirms that the intended stores disappear but are not
on the latency-critical path:

| metric | R93 | R95 | change |
| --- | ---: | ---: | ---: |
| duration us | 659.36 | 662.08 | +0.41% |
| global-store sectors | 26,775 | 26,283 | -1.84% |
| global-load sectors | 2,202,759 | 2,204,502 | +0.08% |
| executed warp instructions | 172,500,308 | 172,496,360 | -0.002% |
| executed thread instructions | 5,380,222,462 | 5,380,249,334 | +0.0005% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The store reduction is real, but cleanup already overlaps the epilogue's
combine work and does not shorten the maximum-rank critical path. The screen
sign reversal and NCU duration regression make a formal run unjustified. R95
is rejected and fully reverted. Correctness/resource, screen, and NCU evidence
is archived under `iter259` through `iter261`.

## R96 rejected: pipelined packed BF16 epilogue for Pro M8

### Reason and direction

R94's packed BF16 epilogue silently disabled R12's accepted two-fragment
weight-half pipeline because `kPipelineWeightHalves` excluded every packed
bucket. That explained R94's roughly 3% increase in executed instructions.
R96 retested the composition at exact Pro M8 while explicitly preserving the
two separate WGMMA commit groups and `wait<1>` overlap. Other packed buckets
and all other authoritative points compiled their original schedule.

Eight-rank correctness passes at `diff=0.000716`. The cubin remains at 107
registers/thread, zero stack/local allocation, 1024 bytes static shared
memory, and 100.58 KiB dynamic shared memory.

### Profiler gate

Matched one-rank NCU verifies that R96 repairs R94's scheduling side effect:

| metric | R93 | R96 | change |
| --- | ---: | ---: | ---: |
| duration us | 665.38 | 664.00 | -0.21% |
| shared-load bank conflicts | 8,079 | 7,255 | -10.20% |
| shared-store bank conflicts | 3,991,725 | 3,997,445 | +0.14% |
| global-load sectors | 2,202,036 | 2,203,133 | +0.05% |
| executed warp instructions | 172,496,995 | 172,487,537 | -0.005% |
| executed thread instructions | 5,380,258,889 | 5,380,006,286 | -0.005% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The instruction counts return to R93 rather than R94's `+2.90%/+2.95%`, and
the isolated duration direction is slightly positive, so R96 proceeded to a
distributed screen.

### Eight-rank rejection

The 20-observation cold-L2 R93/R96/R93 screen was negative against both
controls:

| point | first R93 us | R96 us | change | second R93 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 769.2190 | 791.1800 | +2.85% | 781.6710 | +1.22% |
| Pro M8 rank 0 | 749.6120 | 782.9295 | +4.45% | 755.4285 | +3.64% |

The local NCU conflict reduction does not survive the production cross-rank
dependency path. Because both maximum-rank controls and rank 0 regress well
beyond the profiler's `0.21%` signal, a formal run is unjustified. R96 is
rejected and fully reverted. Correctness/resource, NCU, and screen evidence is
archived under `iter262` through `iter264`.

## R97 rejected: two-task L2 scheduler claims for Pro M512

### Reason and direction

The existing Pro M512 profile is locally faster than PR383, while its
production residual appears only in the multi-rank persistent schedule. R97
therefore reduced scheduler contention rather than changing arithmetic. L1's
task counter also publishes dependency progress and continued to claim one
task at a time. Exact Pro M512 made each dependency-free L2 atomic claim
reserve two consecutive, equal-cost N tiles, cutting claim frequency while
limiting per-CTA tail ownership to one extra tile. Every other specialization
retained single-task claims.

Eight-rank correctness passes at `diff=0.000708`. The cubin remains at 128
registers/thread, zero stack/local allocation, 1024 bytes static shared
memory, and 100.58 KiB dynamic shared memory.

### NCU gate and production screen

Matched one-rank NCU confirms the intended scheduler effect:

| metric | R93 | R97 | change |
| --- | ---: | ---: | ---: |
| duration ms | 2.23 | 2.23 | flat at report precision |
| L1 global-atomic sectors | 10,256 | 8,296 | -19.11% |
| L2 atomic sectors | 14,983 | 12,122 | -19.10% |
| executed warp instructions | 513,921,386 | 514,047,845 | +0.025% |
| executed thread instructions | 16,134,487,793 | 16,135,849,752 | +0.008% |
| shared-load bank conflicts | 119,711 | 123,975 | +3.56% |
| shared-store bank conflicts | 8,951,698 | 8,701,823 | -2.79% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The atomics fall, but isolated duration does not. A five-observation cold-L2
R93/R97/R93 screen then changed sign on the required maximum-rank metric:

| point | first R93 us | R97 us | change | second R93 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M512 max rank | 2574 | 2572 | -0.08% | 2547 | +0.98% |
| Pro M512 rank 0 | 2547 | 2538 | -0.35% | 2544 | -0.24% |

Rank 0 benefits slightly, but pre-claiming one extra tile increases tail
ownership variance and loses against the second maximum-rank control. A larger
claim size would amplify that exact risk. R97 is rejected without a formal
run and fully reverted. Correctness/resource, NCU, and screen evidence is
archived under `iter265` through `iter267`.

## R98 accepted: share completed expert counts inside each Pro M8 CTA

### Reason and direction

Source-counter analysis showed that the dispatch schedulers and the B-loader
scheduler independently polled and then reread the same 48 completed expert
totals from symmetric global memory. In the distributed Pro M8 schedule those
reads sit ahead of the short routed-token pull path, so repeating them across
three scheduler consumers is more visible than at larger M.

R98 specializes exact routed Pro M8. Dispatch warp 0 polls and loads the 48
totals once into shared memory. Both dispatch warps and the B-loader warp join
a 96-thread named barrier, then consume the shared copy. The scheduler exposes
separate global-fetch and cached-fetch paths; all other shapes retain their
original global path. Task ownership, arithmetic, communication buffers, and
ring layout are unchanged.

Eight-rank production correctness passes at `diff=0.000716`. The cubin stays
at 107 registers/thread with zero stack/local allocation, 1024 bytes static
shared memory, and 100.58 KiB dynamic shared memory.

### Screen and formal production performance

The 20-observation cold-L2 R93/R98/R93 screen was positive against both
controls:

| point | first R93 us | R98 us | change | second R93 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 783.1205 | 763.1985 | -2.54% | 767.5765 | -0.57% |
| Pro M8 rank 0 | 766.4310 | 739.0865 | -3.57% | 741.8240 | -0.37% |

The authoritative 50-observation, 20-launch, one-warmup formal run preserved
the double-positive maximum-rank result:

| point | first R93 us | R98 us | change | second R93 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 787.0675 | 765.8580 | -2.69% | 773.7380 | -1.02% |
| Pro M8 rank 0 | 781.2780 | 742.2170 | -5.00% | 743.9510 | -0.23% |

### NCU and NSYS evidence

Matched one-rank full-section NCU confirms the intended traffic reduction
without a resource penalty:

| metric | R93 | R98 | change |
| --- | ---: | ---: | ---: |
| NCU duration us | 716.70 | 718.14 | +0.20% |
| global-load sectors | 2,203,060 | 2,179,933 | -1.05% |
| global-store sectors | 26,780 | 26,821 | +0.15% |
| L2 read sectors | 75,739,543 | 75,622,246 | -0.15% |
| executed warp instructions | 172,471,607 | 171,965,202 | -0.29% |
| executed thread instructions | 5,732,716,687 | 5,763,005,743 | +0.53% |
| excessive shared wavefronts | 3,845 | 3,845 | unchanged |
| local spilling requests | 0 | 0 | unchanged |

The added shared publication and barrier increase thread-level work, while
removing duplicate scheduler loads reduces global traffic and warp-level
instructions. The isolated NCU duration is therefore neutral rather than the
acceptance signal. A matched NSYS trace reports one target-kernel launch at
`663.679 us` for R93 and `662.975 us` for R98 (`-0.11%`), also effectively
neutral on one rank. The reproducible eight-rank maximum-rank improvement is
the deciding evidence because that is where completed expert counts are a
cross-rank dependency.

R98 is accepted. Gate, screen, formal, NCU, and NSYS evidence is archived
under `iter268` through `iter272`.

### R98 authoritative DSV4 matrix against PR383

R98 and PR383 were measured back-to-back on the same eight-H20 pod with the
full requested contract. Positive gap means R98 is slower.

| model | M | R98 us | PR383 us | gap |
| --- | ---: | ---: | ---: | ---: |
| Flash | 8 | 323.5095 | 300.2340 | +7.75% |
| Flash | 16 | 328.8660 | 304.8945 | +7.86% |
| Flash | 32 | 331.9085 | 331.8395 | +0.02% |
| Flash | 64 | 368.1605 | 364.8300 | +0.91% |
| Flash | 128 | 435.6905 | 435.3910 | +0.07% |
| Flash | 256 | 514.8480 | 487.0910 | +5.70% |
| Flash | 512 | 890.0580 | 899.4580 | -1.05% |
| Flash | 1024 | 1512.0000 | 1542.3860 | -1.97% |
| Flash | 2048 | 2740.0000 | 2716.3120 | +0.87% |
| Flash | 4096 | 5109.0000 | 5069.0000 | +0.79% |
| Flash | 8192 | 9902.0000 | 9812.0000 | +0.92% |
| Pro | 8 | 754.5660 | 712.9715 | +5.83% |
| Pro | 16 | 991.8705 | 1004.4275 | -1.25% |
| Pro | 32 | 1033.0000 | 1102.3700 | -6.29% |
| Pro | 64 | 1069.5000 | 1151.7610 | -7.14% |
| Pro | 128 | 1224.5000 | 1279.7300 | -4.32% |
| Pro | 256 | 1616.0000 | 1628.5130 | -0.77% |
| Pro | 512 | 2525.0000 | 2398.3250 | +5.28% |
| Pro | 1024 | 3900.0000 | 4007.0000 | -2.67% |
| Pro | 2048 | 6896.0000 | 7031.0000 | -1.92% |
| Pro | 4096 | 12972.0000 | 12881.0000 | +0.71% |
| Pro | 8192 | 25275.0000 | 25106.0000 | +0.67% |

The 22-point geometric gap is `+0.377920%`. Flash is `+1.936931%` and Pro is
`-1.157248%`; small and large slices are `+0.212648%` and `+0.515855%`.
Detailed splits are Flash small `+3.258646%`, Flash large `+0.848434%`, Pro
small `-2.743498%`, and Pro large `+0.184372%`.

Flash M8 is unchanged by R98 yet moved from R93's prior `298.706 us` versus
`327.4525 us` PR383 comparison to `323.5095 us` versus `300.2340 us`. Flash
M16 and M256 also move far more than any intervening branch change permits.
Those points expose whole-matrix session drift and are not attributed to R98.
The exact interleaved R93/R98/R93 formal run remains R98's causal evidence.
The largest repeatable code-path residual remains Pro M512, followed by Pro
M8. Complete matrix logs are archived under
`iter273-r98-pr383-full-matrix`.

## R99 accepted: extend CTA expert-count sharing to Pro M512

### Reason and direction

R98 proved that publishing one completed-expert snapshot per CTA can reduce
the three scheduler consumers' duplicate global loads. Pro M512's one-rank
arithmetic was already faster than PR383, while its stable residual appears
only in the distributed persistent schedule. R99 therefore extends the same
cache path to exact Pro M512 and leaves M256 plus M1024 and above unchanged.
The bucket is identified by Pro hidden size, regular layout, the accepted
M512+ bank permutation, and absence of the M1024+ L2 C/D swizzle.

Eight-rank production correctness passes at `diff=0.000716`. The cubin stays
at 128 registers/thread, zero stack/local allocation, 1024 bytes static shared
memory, and 100.58 KiB dynamic shared memory.

### Production timing

The requested three-observation, 20-launch, one-warmup cold-L2 R98/R99/R98
screen was positive against both controls:

| point | first R98 us | R99 us | change | second R98 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M512 max rank | 2548 | 2544 | -0.16% | 2548 | -0.16% |
| Pro M512 rank 0 | 2540 | 2525 | -0.59% | 2548 | -0.90% |

Because the signal was small, an independent reverse-order R99/R98/R99 run
was required. Its maximum-rank medians were `2598/2618/2609 us`; the two R99
samples improve `0.76%` and `0.34%` against their shared control. Rank-0
medians were `2598/2591/2601 us`, slightly negative, but the acceptance metric
is the maximum-rank median. All four maximum-rank comparisons are positive.

### NCU and NSYS evidence

Matched one-rank NCU verifies the intended traffic reduction and its offset:

| metric | R98 | R99 | change |
| --- | ---: | ---: | ---: |
| duration ms | 2.42 | 2.42 | flat at report precision |
| global-load sectors | 4,898,944 | 4,874,507 | -0.50% |
| global-store sectors | 1,583,274 | 1,583,285 | flat |
| executed warp instructions | 513,926,085 | 512,867,748 | -0.21% |
| executed thread instructions | 16,134,447,217 | 16,188,445,714 | +0.33% |
| shared-load bank conflicts | 123,021 | 122,054 | -0.79% |
| shared-store bank conflicts | 8,541,210 | 8,980,236 | +5.14% |
| local load/store sectors | 0 / 0 | 0 / 0 | unchanged |

The shared publication adds thread work and store pressure, so isolated
kernel time remains neutral. NSYS similarly moves from `2.225021 ms` to
`2.226526 ms` (`+0.07%`). R99 is accepted on the four reproducible distributed
maximum-rank comparisons, not on profiler duration. Correctness, both timing
orders, NCU, and NSYS evidence is archived under `iter274` through `iter278`.

## R100-R103 rejected: extend expert-count sharing beyond Pro M512

### R100 broad regular-Pro extension

R100 removed R99's M512 upper boundary and cached counts for every regular
Pro M512+ point. The requested three-observation R99/R100/R99 screen was:

| Pro M | first R99 us | R100 us | change | second R99 us | reverse change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2559 | 2557 | -0.08% | 2581 | -0.93% |
| 1024 | 3889 | 3928 | +1.00% | 3933 | -0.13% |
| 2048 | 6942 | 6953 | +0.16% | 6991 | -0.54% |
| 4096 | 13108 | 13031 | -0.59% | 13042 | -0.08% |
| 8192 | 25235 | 25316 | +0.32% | 25307 | +0.04% |

M1024 and M2048 change sign, while M8192 regresses against both controls.
Only the already accepted M512 and M4096 are double-positive, so the broad
selector is rejected.

### R101-R103 exact-M4096 isolation attempts

M1024 through M8192 share one generated specialization, so R101 used a
uniform runtime `num_tokens == 4096` branch. M4096 measured `13082 us` versus
`13231/13113 us` controls (`-1.13%/-0.24%`), but false-branch M8192 regressed
to `25257 us` versus `25207/25237 us` (`+0.20%/+0.08%`). The runtime branch
therefore had a measurable non-target cost.

R102 added an explicit host-generated template boolean to restore a compile-
time M4096 path. All 13 eight-rank production correctness scenarios passed,
but the changed specialization compiled differently and performance reversed:
M4096 was `13135 us` versus `13117/12981 us` controls
(`+0.14%/+1.19%`). The unselected M1024 and M8192 guards also failed to show
identity with R99.

R103 restored R99's template signature and selected exact M4096 with a
generated-source macro. The cubin retained 128 registers/thread and zero
stack/local allocation. M4096 was `13022 us` versus `13049/12996 us`
(`-0.21%/+0.20%`), a direct sign reversal. M8192 also changed sign at
`25284 us` versus `25175/25317 us`. The apparent M4096 gain is below the
distributed three-observation noise floor and does not justify a host selector.

R100 through R103 are fully reverted; R99 remains the control. Broad, runtime,
template-boolean, macro, resource, and correctness evidence is archived under
`iter279` through `iter285`.

## R104 rejected: one L1 warmup wave for Pro M512

### Reason and direction

Pro has 48 L1 and 56 L2 N tasks per routed M block. Its 156 CTA workers make
the first L2 wave touch only three M blocks, whose 144 prerequisite L1 tasks
fit inside one 156-task L1 wave. The generic deadlock-safe formula uses two
warmup waves. R104 specialized exact Pro M512 to one wave so L2 could begin
one global wave earlier; total task count, dynamic claims, dependency polling,
and ring capacity were unchanged.

Eight-rank production correctness passed at `diff=0.001017`. The cubin stayed
at 128 registers/thread, zero stack/local allocation, 1024 bytes static shared
memory, and 100.58 KiB dynamic shared memory.

### Timing and rejection

The first three-observation R99/R104/R99 run looked double-positive:

| point | first R99 us | R104 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M512 max rank | 2619 | 2549 | -2.67% | 2579 | -1.16% |
| Pro M512 rank 0 | 2520 | 2522 | +0.08% | 2579 | -2.21% |

Both controls contained long maximum-rank outliers, so an independent reverse
R104/R99/R104 run was required. Its maximum-rank medians were
`2586/2567/2544 us`: the two candidates are respectively `+0.74%` and
`-0.90%` against the shared control. The direct sign reversal fails the
two-sided contract even though the dependency proof and correctness hold.

R104 is fully reverted without profiler work; changing task order but not
work count is already rejected by the authoritative distributed timing.
Correctness/resource and both timing orders are archived under `iter286`
through `iter288`.

## R105-R106 rejected: publish derived expert totals with the count cache

### Reason and direction

R98/R99 publish completed expert counts once per CTA, but all three scheduler
consumers still reconstruct `num_total_m_blocks`, and the writer dispatch warp
reloads its own shared snapshot. R105 kept the writer's register snapshot,
published the derived total in aligned dispatch scratch, and let the other two
consumers skip their warp prefix/reduce. A static bound protected the metadata
word. M8 and M512 correctness passed at `0.000716/0.000709`; M8 stayed at 107
registers while M512 unexpectedly fell from 128 to 125, with zero spills.

The first R99/R105/R99 run separated the two buckets:

| point | first R99 us | R105 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 793.4555 | 768.1840 | -3.18% | 785.3335 | -2.18% |
| Pro M512 max rank | 2530 | 2537 | +0.28% | 2539 | -0.08% |

M512 changed sign, so R106 restored its original R99 cache APIs and retained
the derived-total path only for exact M8. Correctness passed at
`0.000716/0.000713`; M512 returned to 128 registers and M8 remained at 107,
both without stack/local allocation.

The 50-observation M8 plus three-observation M512 R99/R106/R99 run was
double-positive:

| point | first R99 us | R106 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 773.7680 | 771.4345 | -0.30% | 775.8805 | -0.57% |
| Pro M512 max rank | 2547 | 2518 | -1.14% | 2526 | -0.32% |

An independent reverse R106/R99/R106 run invalidated both directions. M8 was
`780.5625/776.0400/757.8575 us`, so the two candidates were `+0.58%/-2.34%`.
M512 was `2546/2526/2516 us`, likewise `+0.79%/-0.40%`. The instruction-saving
mechanism is smaller than the distributed max-rank variance and is fully
reverted without treating the initial double-positive sample as causal.

Correctness/resources and both timing orders are archived under `iter289`
through `iter293`. R99 remains the accepted control.

## R99 targeted same-session baseline against PR383

The R98 full matrix mixed several large whole-session shifts with the two
repeatable Pro residuals. A focused PR383/R99/PR383 run therefore repeated
Flash M8/M16 and Pro M8/M512 in one session under the authoritative contract:
one warmup, 50 observations for M8/M16, three for M512, 20 launches per
observation, cold L2, seed zero, and the maximum-rank median. PR383 is still
the sum of its native FP8 L1 and L2 kernels.

| point | first PR383 us | R99 us | gap | second PR383 us | reverse gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M8 | 306.2755 | 306.5425 | +0.09% | 302.3385 | +1.39% |
| Flash M16 | 321.9730 | 323.0520 | +0.34% | 304.2195 | +6.19% |
| Pro M8 | 725.6185 | 779.5105 | +7.43% | 726.4450 | +7.30% |
| Pro M512 | 2392.9780 | 2596.0000 | +8.48% | 2407.9420 | +7.81% |

The two Pro controls differ by only `0.11%` at M8 and `0.63%` at M512, so
their deficits are stable and remain the highest-priority code paths. Flash
M16's PR383 controls move `5.51%` within the same session; that point cannot
support causal attribution in this run. Raw logs are archived under
`iter294-r99-pr383-priority-points`.

## R107 rejected: return after the first scheduler owner match

### Reason and direction

The general `create_task()` search iterates two expert groups for DSV4 Pro.
Once a warp-uniform owner ballot succeeds, the pool block has exactly one
owner, but the old loop still reconstructs prefixes for later groups. R107
returned immediately after filling `TaskInfo`, preserving task numbering,
owner arithmetic, memory layout, and all fallback paths. All 13 eight-rank
production correctness scenarios passed, including forced ring wraps; Pro M8
and M512 diffs were `0.000716` and `0.000774`. Their cubins remained at
107/128 registers, zero stack/local allocation, 1024 bytes static shared
memory, and 100.58 KiB dynamic shared memory.

### Formal timing and profiler result

The authoritative R99/R107/R99 comparison changed sign at both targets:

| point | first R99 us | R107 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pro M8 max rank | 769.0650 | 781.0730 | +1.56% | 785.3380 | -0.54% |
| Pro M512 max rank | 2554 | 2573 | +0.74% | 2586 | -0.50% |

Matched one-rank Pro M512 NCU shows why the source-level shortcut is too
small to retain. Executed warp instructions fall only `0.0196%`, executed
thread instructions fall `0.0208%`, and global-load sectors fall `0.0149%`.
Shared-load bank conflicts rise `2.03%`, shared-store conflicts fall `0.63%`,
and both kernels measure `2.21 ms`; local traffic remains zero. The compiler
and the rest of the fused kernel make the eliminated final scans negligible,
while the new early branch has no reproducible distributed benefit.

R107 is fully reverted. Correctness, formal timing, cubin resources, and NCU
evidence are archived under `iter295` through `iter297`; R99 remains the
accepted control.

## R108 rejected: sparse local counts without an extra rendezvous

### Reason and direction

R89 removed zero-count CTA/expert atomics at Pro M8, but paid for a new NVLink
rendezvous, destination aggregation, and grid synchronization before making
the totals ready. R108 isolated the useful half of that experiment. Nonzero
CTAs accumulated only the low count field; the already-existing grid barrier
proved those updates complete, after which SM0 synthesized the original
`kNumSMs` completion field and used the unchanged per-source-rank system
atomic publication. No new grid or cross-rank barrier was introduced, and
remote readiness, token offsets, and routing order were unchanged.

Exact eight-rank correctness passed at `diff=0.000716`. The cubin retained
107 registers/thread, zero stack/local allocation, 1024 bytes static shared
memory, and 100.58 KiB dynamic shared memory.

### Formal timing and profiler result

The authoritative 50-observation R99/R108/R99 run was negative at maximum
rank despite a rank-0 improvement:

| metric | first R99 us | R108 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 774.9495 | 778.3200 | +0.43% | 771.1270 | +0.93% |
| rank-0 median | 761.6515 | 753.9480 | -1.01% | 755.5865 | -0.22% |

Matched one-rank NCU confirms that the mechanism works but is not critical.
L1/L2 atomic sectors fall from `6408/9359` to `4548/6628`, both about `29%`.
Executed warp/thread instructions fall only `0.0066%/0.0022%`, global-load
sectors fall `0.0113%`, and local traffic stays zero. Isolated duration moves
from `658.94 us` to `661.18 us` (`+0.34%`). Removing R89's additional
rendezvous therefore does not rescue sparse count publication: those local
atomics overlap useful work, while the production maximum-rank tail regresses
against both controls.

R108 is fully reverted. Future Pro M8 work should not spend another selector
on local dispatch-count atomics without a separate critical-path change.
Correctness/resources, formal timing, and NCU reports are archived under
`iter298` through `iter300`; R99 remains the accepted control.

## R109 rejected: poll sparse totals without the final grid barrier

### Reason and direction

R108 showed that sparse local completion atomics alone are not critical. The
remaining R89 cost is synchronization after SM0 aggregates all source-rank
counts. Every scheduler consumer already polls each `recv_count_sum` ready
field, while R98's CTA count-cache barrier keeps the B-loader behind dispatch
warp 0's complete snapshot. R109 therefore enabled the existing sparse
completion protocol for exact routed Pro M8 but let CTAs leave the NVLink
rendezvous independently after SM0 publication, omitting only the final grid
barrier. Accepted Flash sparse selectors retained their original barrier.

Exact eight-rank correctness passed at `diff=0.000716`, the generated source
contained the intended sparse macro, and the cubin retained 107 registers,
zero stack/local allocation, 1024 bytes static shared memory, and 100.58 KiB
dynamic shared memory.

### Initial positive run and reverse rejection

The first authoritative R99/R109/R99 maximum-rank comparison was positive:

| first R99 us | R109 us | change | second R99 us | reverse change |
| ---: | ---: | ---: | ---: | ---: |
| 770.3455 | 762.5745 | -1.01% | 767.5610 | -0.65% |

Rank 0 moved `744.029/751.265/755.040 us`, so its direction was mixed while
the maximum rank improved. An independent reverse-order run invalidated the
apparent tail benefit:

| first R109 us | R99 us | change | second R109 us | reverse change |
| ---: | ---: | ---: | ---: | ---: |
| 765.1935 | 759.4120 | +0.76% | 794.1445 | +4.57% |

Both candidate samples lose to their shared control, and the second candidate
has a large tail excursion. Removing the global ordering point changes which
rank becomes the straggler rather than reproducibly shortening that straggler.

### NCU and NSYS mechanism evidence

Matched one-rank NCU and NSYS do show a real isolated benefit. NCU duration
falls `664.90 -> 659.87 us` (`-0.76%`) and NSYS falls
`665.791 -> 660.287 us` (`-0.83%`). L1/L2 atomic sectors fall from
`6408/9361` to `4380/6395` (about `-31.7%`), executed warp instructions fall
`0.014%`, and local traffic stays zero. Global-load sectors rise `0.48%` and
thread instructions rise `0.003%` due to count polling/aggregation.

The profiler result confirms that the omitted barrier shortens an isolated
rank, but the reverse production run proves that it destabilizes cross-rank
arrival order. R109 is fully reverted. Gate, both timing orders, NCU, and
NSYS evidence is archived under `iter301` through `iter305`; R99 remains the
accepted control.

## R110 rejected: sparse dispatch completion for Pro M512

R89, R108, and R109 covered Pro M8 but not the other stable residual at Pro
M512. R110 enabled the existing fully synchronized sparse completion protocol
only for exact routed Pro M512. Unlike R109, it retained the post-aggregation
grid barrier, so the experiment changed count publication work but not CTA
release ordering. Accepted Flash selectors and every other Pro point were
unchanged.

Eight-rank correctness passed at `diff=0.000714`, generated source contained
the sparse macro, and the cubin retained 128 registers/thread, zero
stack/local allocation, 1024 bytes static shared memory, and 100.58 KiB
dynamic shared memory.

Matched one-rank NCU shows a smaller atomic reduction than at M8: L1 atomic
sectors fall `10256 -> 8990` (`-12.34%`) and L2 atomic sectors fall
`14985 -> 12994` (`-13.29%`). Executed warp/thread instructions change only
`-0.0010%/-0.0006%`, global loads rise `0.018%`, local traffic remains zero,
and both durations round to `2.23 ms`.

The requested three-observation R99/R110/R99 production result was:

| first R99 us | R110 us | change | second R99 us | reverse change |
| ---: | ---: | ---: | ---: | ---: |
| 2552 | 2554 | +0.08% | 2551 | +0.12% |

The candidate is slightly slower than both controls and the profiler has no
duration signal. Sparse completion merely exchanges local atomics for count
aggregation at this route density, so R110 is fully reverted without an
unnecessary reverse run or NSYS trace. Gate, NCU, and formal evidence is
archived under `iter306` through `iter308`; R99 remains the accepted control.

## R111 rejected: replenish two L1 waves after each Pro M512 L2 claim

### Reason and direction

The accepted scheduler replenishes one L1 wave after each L2 claim. At exact
Pro M512, an L1 task spans K=7168 while an L2 task spans K=3072, so an L1 task
contains about 2.33 times as much K work. R111 tested whether the tail would be
better balanced by replenishing two L1 waves per L2 claim. The change was
selected only for Pro M512; warmup, task counts, dependency polling, ring
capacity, and all other shapes remained unchanged.

Eight-rank correctness passed at `diff=0.000719`. The cubin retained 128
registers/thread, zero stack/local allocation, and 1024 bytes static shared
memory.

### Profiler and formal result

Matched one-rank NCU showed no material critical-path reduction. Both kernels
measured `2.24 ms`; executed warp/thread instructions changed by
`-0.0046%/-0.0016%`, global-load sectors changed by `-0.044%`, and local
traffic remained zero. Barrier stall increased from `4.06` to `4.09` and wait
stall increased from `1.00` to `1.01`, while long-scoreboard stall stayed at
`2.51`.

The authoritative three-observation R99/R111/R99 production comparison was
negative at maximum rank:

| metric | first R99 us | R111 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 2544 | 2564 | +0.79% | 2526 | +1.50% |
| rank-0 median | 2529 | 2519 | -0.40% | 2501 | +0.72% |

The rank-0 direction is mixed, while the production maximum rank loses to
both controls. Static work-ratio scheduling does not account for expert
skew, remote readiness, or which rank becomes the straggler, and the higher
barrier/wait stalls are consistent with less favorable dependency timing.
R111 is fully reverted; no NSYS run was warranted after the negative NCU and
double-negative production result. Correctness/resources, NCU, and formal
timing evidence is archived under `iter309` through `iter311`; R99 remains the
accepted control.

## R112 rejected: skip idle Pro M8 dispatch-warp expert scans

### Reason and direction

At Pro M8, each rank receives only 48 routed tokens, while the persistent grid
contains 156 CTAs and two dispatch warps per CTA. The original pull loop lets
every warp start from its global token index and scan completed counts for up
to 48 local experts before an out-of-range warp discovers that no token is
assigned to it. R112 added one warp reduction over the already-cached expert
counts and used the resulting total as the pull-loop bound, allowing roughly
264 idle dispatch warps to leave before the expert scan. The selector was
exact for routed Pro M8; atomics, barriers, scheduling, and all other buckets
were unchanged.

Eight-rank correctness passed at `diff=0.000716`. The cubin retained 107
registers/thread, zero stack/local allocation, and 1024 bytes static shared
memory.

### Profiler and timing result

Matched one-rank NCU confirmed that the source change removed work. Executed
warp/thread instructions fell `0.148%/0.152%`, global-load sectors fell
`0.015%`, and duration moved `664.54 -> 662.88 us` (`-0.25%`). Atomic traffic
was unchanged and local traffic remained zero.

The first authoritative R99/R112/R99 run followed a large session drift:

| metric | first R99 us | R112 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 787.6335 | 772.5475 | -1.92% | 763.8320 | +1.14% |
| rank-0 median | 768.1745 | 753.2860 | -1.94% | 750.5160 | +0.37% |

Because both metrics straddled their controls, an independent reverse
R112/R99/R112 run was required:

| metric | first R112 us | R99 us | change | second R112 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 775.4545 | 763.0855 | +1.62% | 765.9360 | +0.37% |
| rank-0 median | 763.9150 | 744.1470 | +2.66% | 750.9460 | +0.91% |

Both candidate samples lose to the shared control. The instruction reduction
is real but occurs in dispatch warps that already reconverge behind longer
cross-rank and math dependencies, so it does not shorten the production
maximum-rank tail. R112 is fully reverted without an unnecessary NSYS run.
Correctness/resources, NCU, and both timing orders are archived under
`iter312` through `iter315`; R99 remains the accepted control.

## R113-R114 rejected: hierarchical Pro M8 grid barriers

### Reason and direction

Exact routed Pro M8 launches 156 CTAs, while PR383 uses 78 CTAs for each
separate phase. The fused kernel crosses several grid-wide synchronization
points, so R113 tested whether pairing the two resident CTAs on each H20 SM
and letting only 78 leaders perform the global atomic barrier could reduce
the synchronization tail. The workspace barrier region was temporarily
expanded by 640 bytes for 78 dispatch pairs and 78 epilogue pairs. All other
shapes retained the accepted barrier implementation.

R113 used a full pair barrier before and after the leader-only global
barrier. Eight-rank correctness passed at `diff=0.000716`; the cubin retained
107 registers/thread, zero spill/local allocation, 1024 bytes static shared
memory, and 100.58 KiB dynamic shared memory. However, each grid rendezvous
now performed 390 atomic operations instead of 156. Matched one-rank NCU
showed duration increasing `660.83 -> 665.66 us` (`+0.73%`), L1 atomic sectors
increasing `6408 -> 7578` (`+18.26%`), and L2 atomic sectors increasing
`9356 -> 11068` (`+18.30%`). Barrier stall also rose `2.53 -> 2.59`.

R114 replaced the two full pair barriers with a phase protocol: the follower
release-stored its next phase into a pair-local counter, the leader
acquire-polled that counter and joined the 78-leader global barrier, and the
follower acquire-polled the global phase directly. This reduced each pair to
one store plus one global atomic operation. Correctness again passed at
`diff=0.000716`, with the same 107-register, zero-spill resource profile.

### Profiler and production result

The optimized protocol removed atomic traffic but not critical-path time:

| NCU metric | R99 | R114 | change |
| --- | ---: | ---: | ---: |
| duration | 661.76 us | 662.66 us | +0.14% |
| L1 atomic sectors | 6408 | 6018 | -6.09% |
| L2 atomic sectors | 9359 | 8793 | -6.05% |
| global-store sectors | 26775 | 27165 | +1.46% |
| global-load sectors | 2179532 | 2182223 | +0.12% |
| barrier stall | 2.52 | 2.55 | +1.19% |
| long-scoreboard stall | 2.27 | 2.28 | +0.44% |
| warp instructions | 171966423 | 171992401 | +0.015% |
| thread instructions | 5405114404 | 5405190331 | +0.0014% |

The 20-observation R99/R114/R99 screen remained inconclusive-to-negative:

| metric | first R99 us | R114 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 756.7775 | 774.8470 | +2.39% | 784.1160 | -1.18% |
| rank-0 median | 744.6410 | 769.7930 | +3.38% | 772.8055 | -0.39% |

Both metrics straddle their two controls, while matched NCU is neutral to
slower despite fewer atomics. The added phase polling and stores merely trade
one synchronization mechanism for another; they do not reduce the
cross-rank straggler. R113 and R114 are fully reverted, including the
temporary workspace ABI expansion. A 50-observation production run and NSYS
trace were not warranted after the screen and NCU rejection. Build,
correctness/resources, NCU, and timing evidence is archived under `iter316`
through `iter321`; R99 remains the accepted control.

## R115-R116 rejected: token-ready Pro M8 combine

### Reason and direction

R99 is locally close to PR383 at Pro M8 but remains slower at the distributed
maximum-rank tail. Both implementations normally wait for every rank to
finish every L2 scatter before any token starts combine. R115 tested a finer
dependency: publish completion for each destination token's L2 output and let
the CTA assigned to that token start combine as soon as its own six routed
contributions are visible. The exact Pro M8 selector moved the local grid
rendezvous after combine and retained a final cross-rank cleanup barrier.

The prototype temporarily added one completion counter per destination token
to symmetric workspace. Every completed L2 row/N tile issued a system-scope
release atomic to the destination token, and combine acquire-polled for
`valid_topk * 56` completions. The final barrier made counter reset safe across
back-to-back launches. Eight-rank correctness passed at `diff=0.000716`, with
116 registers/thread and zero stack/local allocation; R99 uses 107 registers.

### R115 rejection

The 20-observation R99/R115/R99 screen was decisively negative:

| metric | first R99 us | R115 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 787.9225 | 831.8165 | +5.57% | 780.5040 | +6.57% |
| rank-0 median | 772.9755 | 824.3580 | +6.65% | 762.9790 | +8.04% |

Pro M8 produces about 48 routed rows per rank and 56 L2 N tiles per row, so
R115 adds roughly 2688 remote system atomics per rank. Removing one global
barrier does not repay that traffic or the nine-register increase.

### R116 aggregation and rejection

R116 reduced remote completion traffic by reusing the L2 ring completion
counter. A nonempty L2 task released its ring slot only after scatter; the
last of the 56 N tasks for a pool block then published one system-scope
completion per valid row. This reduced remote completion atomics from about
2688 to about 48 per rank. Correctness again passed at `diff=0.000716`; the
cubin used 115 registers/thread with zero spill.

The independent 20-observation R99/R116/R99 screen remained double-negative:

| metric | first R99 us | R116 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 761.6055 | 814.1525 | +6.90% | 772.8285 | +5.35% |
| rank-0 median | 748.0145 | 794.7040 | +6.24% | 753.4130 | +5.48% |

Aggregation recovers `17.664 us` from R115, confirming that system atomic
traffic was material, but token-ready bookkeeping, the eight-register
residual, delayed ring-slot release, and the post-combine cleanup ordering
still exceed the removed rendezvous cost. Both variants are fully reverted,
including the workspace ABI change. Formal 50-observation, NCU, and NSYS runs
were not warranted after two clear double-negative screens. Build,
correctness/resources, and both A/B/A screens are archived under `iter322`
through `iter326`; R99 remains the accepted control.

## R117 rejected: pair adjacent Pro M512 L2 blocks N-major

### Reason and direction

R99 schedules every L2 N tile for one routed M block before moving to the
next M block. R117 tested whether pairing adjacent routed M blocks and
visiting the pair N-major (`N0/M0`, `N0/M1`, `N1/M0`, `N1/M1`, ...) could
reuse an expert's MXFP4 weights across the two tasks. The specialization was
limited to the routed DSV4 Pro M512 bucket; L1 ordering, task count,
dependencies, barriers, and all other benchmark cases were unchanged.

Eight-rank correctness passed at `diff=0.000716`. The generated cubin stayed
at 128 registers/thread, zero stack/local allocation, 1024 bytes static
shared memory, and 100.58 KiB dynamic shared memory.

### Timing and profiler result

The authoritative R99/R117/R99 run was already inconclusive-to-negative
because its second control contained large outliers:

| metric | first R99 us | R117 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 2566 | 2615 | +1.91% | 2776 | -5.80% |
| rank-0 median | 2560 | 2601 | +1.60% | 2769 | -6.07% |

The independent reverse R117/R99/R117 run rejected the candidate against a
stable shared control:

| metric | first R117 us | R99 us | change | second R117 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 2552 | 2528 | +0.95% | 2600 | +2.85% |
| rank-0 median | 2526 | 2524 | +0.08% | 2590 | +2.61% |

Matched one-rank NCU also falsified the intended locality mechanism:

| NCU metric | R99 | R117 | change |
| --- | ---: | ---: | ---: |
| duration | 2.23 ms | 2.24 ms | +0.45% |
| L2 read sectors | 198749333 | 202804783 | +2.04% |
| L2 read hit rate | 69.12% | 68.28% | -0.84 pp |
| L1 global-load sectors | 4873435 | 4877879 | +0.09% |
| warp instructions | 512858470 | 512937729 | +0.015% |
| thread instructions | 16188667868 | 16189648303 | +0.006% |

Adjacent pool blocks are not reliably adjacent blocks of the same expert,
and N-major pairing disrupts the original M-major cache and pipeline
locality. Expert-aware refinement is therefore not justified by this
mechanism. R117 is fully reverted. Correctness/resources, both timing orders,
and NCU evidence are archived under `iter327` through `iter330`; R99 remains
the accepted control.

## R118-R120 rejected: cache or remove Pro M512 dynamic task mapping

### R118: CTA-local pool-block owner table

Every routed L1/L2 N tile reconstructs its pool-block owner from the same 48
Pro expert counts. R118 used the already-allocated expert-count scratch to
build one packed `(expert, expert-M-block, valid-M)` record per routed M64
block in the B-loader warp. Each dynamic task then replaced two warp prefix
scans and ballots with one shared-memory broadcast. The exact M512 selector
and a compile-time capacity proof kept every other bucket unchanged.

The first assertion-enabled prototype passed correctness but introduced a
24-byte stack frame. Removing the redundant device assertions restored 128
registers/thread, zero stack/local allocation, and correctness at
`diff=0.000713`.

Matched one-rank NCU confirmed a small real reduction: duration moved
`2.24 -> 2.23 ms`, warp/thread instructions fell `0.076%/0.076%`, global-load
sectors fell `0.027%`, atomic traffic was unchanged, and local traffic stayed
zero. Long-scoreboard stall, however, moved `2.51 -> 2.52` because each task
now read shared memory.

The first authoritative R99/R118/R99 run straddled its controls:

| metric | first R99 us | R118 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 2566 | 2596 | +1.17% | 2678 | -3.06% |
| rank-0 median | 2553 | 2534 | -0.74% | 2672 | -5.16% |

The independent reverse R118/R99/R118 run remained mixed rather than
double-positive:

| metric | first R118 us | R99 us | change | second R118 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 2545 | 2557 | -0.47% | 2558 | +0.04% |
| rank-0 median | 2543 | 2518 | +0.99% | 2518 | +0.00% |

The instruction reduction is below distributed maximum-rank variance and
does not satisfy the two-sided acceptance contract.

### R119: register-resident expert prefix ends

R119 removed R118's shared table and retained each expert's pool-block prefix
end in scheduler registers after the existing initialization reductions.
Correctness passed at `diff=0.000715`; resources stayed at 128 registers and
zero stack/local allocation. Compared with matched R99, warp/thread
instructions still fell `0.057%/0.054%`, but less than R118, while duration
moved `2.23 -> 2.24 ms` and long-scoreboard stall again moved
`2.51 -> 2.52`. Holding the prefixes across the full kernel did not shorten
the critical path, so R119 was rejected without production timing.

### R120: static strided L1/L2 task assignment

R120 restored R118's owner table and removed dynamic task claims for normal
Pro M512 inputs whose routed blocks fit in one ring generation. Every CTA
received a fixed strided L1 subset followed by a fixed strided L2 subset;
skewed inputs exceeding ring capacity retained the dynamic fallback.
Correctness passed at `diff=0.000709`, and removing dynamic scheduler state
reduced the cubin from 128 to 125 registers with zero spill.

NCU showed that the intended work disappeared but exposed why dynamic
scheduling is required:

| NCU metric | R99 | R120 | change |
| --- | ---: | ---: | ---: |
| duration | 2.23 ms | 2.29 ms | +2.69% |
| L1 atomic sectors | 10256 | 2664 | -74.02% |
| L2 atomic sectors | 14985 | 3888 | -74.05% |
| warp instructions | 512862963 | 511741450 | -0.22% |
| thread instructions | 16188645613 | 16129504406 | -0.37% |
| global-load sectors | 4873851 | 4927760 | +1.11% |
| barrier stall | 4.05 | 4.26 | +5.19% |
| long-scoreboard stall | 2.51 | 2.56 | +1.99% |

Static ownership removes atomics but loses the dynamic scheduler's balancing
of expert skew, readiness, and CTA pipeline progress. R120 was rejected before
formal production timing because both replay duration and critical stalls
were decisively worse. R118-R120 are fully reverted. Correctness/resources,
both R118 timing orders, and all NCU reports are archived under `iter331`
through `iter339`; R99 remains the accepted control.

## R121-R123 rejected: direct Flash small-M pool mapping and count sharing

### R121/R122: one-warp pool-token owner mapping

Flash dispatch originally advances each global pool-token ordinal through a
serial scan of all 32 local expert counts. R121 used the one-expert-per-lane
layout to form expert prefix ends in one warp inclusive scan and select the
owner with a ballot. A runtime guard retained the original scan whenever any
expert exceeded one M64 block. The first candidate selected exact Flash M8
and M16; R122 narrowed the specialization to exact M8 after the M16 result.

Both forced-ring-wrap gates passed: Flash M8 at `diff=0.000671` and Flash M16
at `diff=0.000654`. Their cubins used 114 and 118 registers/thread,
respectively, with zero stack/local allocation. The authoritative
50-observation R99/R121/R99 comparison separated the two buckets:

| point | first R99 us | R121 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flash M8 maximum rank | 304.8880 | 302.8005 | -0.68% | 306.0655 | -1.07% |
| Flash M8 rank 0 | 295.5025 | 282.0530 | -4.55% | 291.4815 | -3.23% |
| Flash M16 maximum rank | 317.3230 | 326.7695 | +2.98% | 323.0615 | +1.15% |
| Flash M16 rank 0 | 306.3745 | 305.3775 | -0.33% | 316.1710 | -3.41% |

The required reverse R122/R99/R122 M8 run then changed sign. Its first R122,
R99, and second R122 maximum-rank medians were `291.6545/304.5890/309.4340
us`; rank-0 medians were `281.6735/290.0390/297.3065 us`. The two candidate
samples were therefore `-4.25%/+1.59%` at maximum rank and `-2.88%/+2.51%`
at rank 0.

Matched one-rank NCU and NSYS do show a real but sub-noise local mechanism:

| metric | R99 | R122 | change |
| --- | ---: | ---: | ---: |
| NCU duration | 206.18 us | 205.60 us | -0.28% |
| NSYS duration | 209.632 us | 209.184 us | -0.21% |
| warp instructions | 47438050 | 47303934 | -0.28% |
| thread instructions | 1475883841 | 1471503455 | -0.30% |
| global-load sectors | 644329 | 645683 | +0.21% |
| barrier stall | 2.99 | 2.97 | -0.67% |
| long-scoreboard stall | 2.57 | 2.60 | +1.17% |

The instruction saving is only about 0.3%, while extra shuffle/ballot work
slightly increases memory-scoreboard pressure. It does not survive the
distributed maximum-rank acceptance contract, so R121/R122 are rejected.

### R123: share completed Flash M8 expert counts per CTA

R123 composed R122 with the expert-count snapshot accepted for Pro M8/M512:
dispatch warp 0 polled all 32 completed Flash counts once, published them in
the existing shared scratch, and synchronized both dispatch warps plus the
B-loader. Correctness remained `diff=0.000671`; resource usage remained 114
registers/thread with zero stack/local allocation.

The count cache removed the expected traffic but moved synchronization onto
the critical path:

| NCU metric | R99 | R123 | change |
| --- | ---: | ---: | ---: |
| duration | 220.42 us | 222.08 us | +0.75% |
| global-load sectors | 642640 | 624374 | -2.84% |
| warp instructions | 47419225 | 47148008 | -0.57% |
| thread instructions | 1475663456 | 1477895546 | +0.15% |
| barrier stall | 2.88 | 2.99 | +3.82% |
| long-scoreboard stall | 2.48 | 2.54 | +2.42% |

The saved polling cannot repay a CTA-wide rendezvous at this very small
workload. R123 was rejected before formal production timing. R121-R123 are
fully reverted; gate, both timing orders, NCU, and NSYS evidence is archived
under `iter340` through `iter345`; R99 remains the accepted control.

## R124 accepted: decode one complete Flash M16 row per lane

### Reason and direction

Fresh matched R99/PR383 NCU narrowed the remaining Flash small-M problem to
M16. At M8, R99's persistent MXFP4 kernel took `221.15 us` versus `249.19
us` for PR383's L1+L2 sum, but at M16 R99 took `274.40 us` versus `270.37
us`. R99 already read 44% fewer DRAM bytes and 32% fewer L2 read sectors at
M16, while executing about 58% more thread instructions. Instruction-mix
profiling attributed the excess primarily to MXFP4 expansion: R99 executed
about 831 million integer and 257 million bit instructions, versus about 283
million and 16 million for PR383's two kernels combined.

The R99 paired decoder assigns two lanes to each N row. Each lane owns one
adjacent two-word pair, independently shuffles the same scale word, and
rebuilds the same E4M3 exponent lookup. R124 specializes exact routed Flash
M16 so each lane owns all four packed words of one N row. It preserves the
two LDS.64 transactions, packed B64 swizzle, and bank-permuted pair order,
but shares one exponent lookup across both pairs and consumes the lane-local
scale without two cross-lane shuffles. All other Flash and Pro buckets keep
the R99 mapping.

### Correctness, resources, and mechanism

The focused forced-ring-wrap gate passed at `diff=0.000654`. The cubin stays
at 118 registers/thread, zero stack/local allocation, and 1024 bytes static
shared memory. The full production suite subsequently passed all 13 Flash
and Pro scenarios, including every forced-wrap case.

Matched one-rank NCU confirms the intended mechanism:

| metric | R99 | R124 | change |
| --- | ---: | ---: | ---: |
| duration | 274.40 us | 264.45 us | -3.63% |
| warp instructions | 63121418 | 60464048 | -4.21% |
| thread instructions | 1966607465 | 1881291342 | -4.34% |
| bit instructions | 256615022 | 232235630 | -9.50% |
| integer instructions | 830061785 | 787080842 | -5.18% |
| inter-thread communication | 81035372 | 68846636 | -15.04% |
| shared-load bank conflicts | 5541 | 5489 | -0.94% |
| shared-store bank conflicts | 1424316 | 1316082 | -7.60% |
| short-scoreboard stall | 0.56 | 0.42 | -25.00% |

The new mapping does not reintroduce R54's former LDS.64 conflicts. NSYS
independently measures `259.903 -> 249.791 us` (`-3.89%`).

### Authoritative distributed timing

The requested 50-observation, 20-launch, one-warmup R99/R124/R99 run is
double-positive despite substantial node tail latency:

| metric | first R99 us | R124 us | change | second R99 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 344.2430 | 319.7520 | -7.11% | 336.5420 | -4.99% |
| rank-0 median | 322.8940 | 304.8415 | -5.59% | 331.2470 | -7.97% |

The independent reverse R124/R99/R124 run also passes both sides:

| metric | first R124 us | R99 us | change | second R124 us | reverse change |
| --- | ---: | ---: | ---: | ---: | ---: |
| maximum-rank median | 331.8845 | 341.9230 | -2.94% | 327.3750 | -4.25% |
| rank-0 median | 317.5255 | 329.1490 | -3.53% | 312.2360 | -5.14% |

A same-session PR383/R124/PR383 production comparison measured maximum-rank
medians `339.0045/330.2475/325.8445 us`: R124 is `2.58%` faster than the
first PR383 control but `1.35%` slower than the second. Rank-0 medians were
`335.4195/309.3840/309.8625 us`, making R124 `7.76%/0.15%` faster. Therefore
R124 clearly removes the R99 regression and locally beats PR383, while a
stable maximum-rank production lead still requires a fresh full-matrix run;
the mixed PR383 sides are not claimed as a conclusive overall win.

R124 is accepted. Matched PR383 decomposition, instruction mix,
correctness/resources, both timing orders, direct PR383 timing, full
production regression, NCU, and NSYS evidence is archived under `iter346`
through `iter354`.
