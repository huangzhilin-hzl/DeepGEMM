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
