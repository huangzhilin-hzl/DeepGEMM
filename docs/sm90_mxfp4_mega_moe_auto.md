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
