# EVCA2 Results Ledger

Running record of every benchmark run, default-behavior change, and gate decision.
Convention: one section per phase; benchmark rows record phase, commit SHA, CLI flags,
device, subset, fps, and correlation tables (or a pointer to the results directory).

> **Historical, from here down.** Everything below predates the simplification pass and
> was produced by the retired harness (`run_benchmark.py`, `run_ablation.py`,
> `report.py`, `stats.py`). Those drivers, their bootstrap CIs, the within-vs-pooled
> scope rule and the `PCC_log` transform no longer exist; neither do the flags several
> sections ablate (`--gate`, `--preset`, `--dct-impl`, the `diamond_axis` /
> `diamond_dense` search patterns, the six unimplemented `--me-*` flags). The findings
> that shaped current defaults are still load-bearing — in particular *Gate 2
> resolution*, which is why the intra gate is now unconditional — but the numbers were
> measured against a half-resolution 17-point search and do not describe today's code.
> Current measurements come from `validation/correlate.py`, which is appended to this
> file by hand.

## Environment

- Machine: Linux (CachyOS), NVIDIA GeForce RTX 5060 Ti, conda env `EVCA-gpu`
  (torch 2.13.0+cu130, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5). The `EVCA` env has a
  broken torch install (MKL `iJIT_NotifyEvent` symbol error) and is not used.
- ffmpeg n9.0.1 with libx265 available at `/usr/bin/ffmpeg`.
- Test sequences: `/home/albert/Desktop/test_sequences/` (UVG 1080p 8-bit yuv420:
  Beauty, Bosphorus, HoneyBee, ReadySteadyGo, ShakeNDry, YachtRide).
  `foodmarket_1920x1080_60fps` from the previous (macOS) configuration is **not**
  available on this machine; the benchmark set is YachtRide, ReadySteadyGo, HoneyBee,
  Bosphorus (paths configurable in `validation/sequences.json`).

## Phase 0 — bug fixes (default output may change)

Fixes 0.1–0.5 and the dead-code part of 0.6 were already applied on the base branch
before this work started (commit `b450068`, "Fix MV field registration, 8-bit optimized
loader, TC2 block CSV, MVC border padding, 10-bit plotting"). For completeness, the
default-output changes now in effect relative to the original Iteration-4 code:

| # | Fix | Commit | Default output change |
|---|-----|--------|----------------------|
| 0.1 | MV-field upsample `align_corners=False` (block MVs registered to block centers) | `b450068` | `TC_MC` changes (residual energy drops for coherent motion) |
| 0.2 | `load_gop_optimized`: int16 reinterpretation only for uint16 sources | `b450068` | `--loader optimized` on 8-bit input now produces correct (previously garbled) tensors |
| 0.3 | TC2 block CSV writes `TC2_blocks[i-2]` instead of `TC_blocks[i-2]` | `b450068` | `*_TC2_blocks.csv` content corrected |
| 0.4 | `MetricMVC` Laplacian uses replicate padding | `b450068` | **MVC values change** (no spurious border gradients under global motion) |
| 0.5 | `plot_block_info_EVCA` honors `args.bit_depth` | `b450068` | 10-bit plots read correct frames (plots only) |
| 0.6a | Dead code removed (`sub_tu_size`, `cached_weights_dct_sub`, `tc_uncomp_batch`) | `b450068` | none |
| 0.6b | `steps` reset per input file | `a95023d` | `--dir` runs where a short file precedes longer ones no longer shrink the GOP for later files (metric values unchanged, only batching; TC2 frame-0 padding could differ in the pathological case) |
| 0.7 | `libs/frame_to_edge.py` deleted (unreferenced, called `edge_detection` with wrong signature) | `793a23a` | none |
| 0.8 | `grid_sample` base grid / Gaussian kernel cached per `(H, W, device)` | `9339ad3` | none (identical numerics, fewer kernel launches) |
| 0.9 | Comments corrected: search runs at **half** resolution (2×2 pool), full-res MVs even-valued | `9339ad3` | none |

Also pre-applied on the base branch (commit `f144ce2`): matmul DCT replacing
`torch_dct` as the transform implementation, deferred host syncs, GOP prefetch
(`--prefetch`, default 1), MPS support. Phase 2's `--dct-impl` flag treats **matmul as
the current default** (rule 2: defaults reproduce current behavior) and adds
`torch_dct` back as the ablation reference.

### Gate 0a

- Full pytest suite: pending Phase 1 tests (existing 31 tests pass on CPU).
- `--profile full` end-to-end on a real sequence: pending (recorded below once run).

### Run `gate1` — 2026-08-21 19:38

- Phase: Phase 1 (post-fix baseline)
- Commit: `5560c9d8180f14920ba48fd50092b996db95ea13`
- Subset: **fast** (120 frames), sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Device: `auto`, loader: `optimized`, profiles: baseline, fast, full
- Extra EVCA args: `(none)`
- Bootstrap: 1000 resamples, seed 12345
- Results: `validation/results/gate1_5560c9d8`

**Throughput**

| profile | frames | seconds | fps |
|---|---|---|---|
| baseline | 480 | 1.28 | 375.00 |
| fast | 480 | 1.59 | 301.89 |
| full | 480 | 1.94 | 247.42 |

**Frame-level pooled correlations** (CI = 95 % bootstrap; `blk` = sequence-level block bootstrap)

| Domain | QP | metric | n | PCC | PCC_lo | PCC_hi | PCC_blk_lo | PCC_blk_hi | SRCC | PCC_log |
|---|---|---|---|---|---|---|---|---|---|---|
| Spatial | 22 | baseline_SC | 480 | 0.7805 | 0.7373 | 0.8185 | -0.7762 | 0.9992 | 0.7231 | 0.7671 |
| Temporal | 22 | baseline_TC | 476 | 0.4371 | 0.3951 | 0.4809 | -0.6931 | 0.9854 | 0.7826 | 0.5339 |
| Temporal | 22 | fast_MVC | 476 | 0.8588 | 0.8329 | 0.8834 | 0.2413 | 0.9822 | 0.8261 | 0.9045 |
| Temporal | 22 | fast_TC_SAD | 476 | 0.5188 | 0.4621 | 0.5919 | -0.1833 | 0.9375 | 0.7707 | 0.5668 |
| Temporal | 22 | full_TC_MC | 476 | 0.6437 | 0.6054 | 0.6893 | -0.9725 | 0.9841 | 0.7156 | 0.6367 |
| Spatial | 27 | baseline_SC | 480 | 0.9817 | 0.9785 | 0.9843 | 0.9018 | 0.9918 | 0.9886 | 0.9865 |
| Temporal | 27 | baseline_TC | 476 | 0.4879 | 0.4488 | 0.5272 | -0.6945 | 0.9819 | 0.7827 | 0.5972 |
| Temporal | 27 | fast_MVC | 476 | 0.8898 | 0.8691 | 0.9101 | 0.3719 | 0.9860 | 0.8528 | 0.9091 |
| Temporal | 27 | fast_TC_SAD | 476 | 0.5583 | 0.5081 | 0.6224 | -0.1365 | 0.9477 | 0.7699 | 0.5661 |
| Temporal | 27 | full_TC_MC | 476 | 0.6550 | 0.6238 | 0.6928 | -0.9799 | 0.9894 | 0.7115 | 0.5514 |
| Spatial | 32 | baseline_SC | 480 | 0.9899 | 0.9877 | 0.9916 | 0.9534 | 0.9987 | 0.9922 | 0.9849 |
| Temporal | 32 | baseline_TC | 476 | 0.5237 | 0.4856 | 0.5608 | -0.6881 | 0.9784 | 0.7830 | 0.6319 |
| Temporal | 32 | fast_MVC | 476 | 0.8849 | 0.8655 | 0.9030 | 0.4031 | 0.9854 | 0.8581 | 0.9021 |
| Temporal | 32 | fast_TC_SAD | 476 | 0.5992 | 0.5552 | 0.6574 | -0.0913 | 0.9546 | 0.7710 | 0.5880 |
| Temporal | 32 | full_TC_MC | 476 | 0.7007 | 0.6739 | 0.7334 | -0.9767 | 0.9914 | 0.7109 | 0.5667 |
| Spatial | 37 | baseline_SC | 480 | 0.9778 | 0.9737 | 0.9813 | 0.6971 | 0.9997 | 0.9624 | 0.9541 |
| Temporal | 37 | baseline_TC | 476 | 0.5769 | 0.5407 | 0.6119 | -0.6658 | 0.9764 | 0.7870 | 0.6721 |
| Temporal | 37 | fast_MVC | 476 | 0.8814 | 0.8621 | 0.8984 | 0.4244 | 0.9835 | 0.8615 | 0.9002 |
| Temporal | 37 | fast_TC_SAD | 476 | 0.6530 | 0.6144 | 0.7040 | 0.0272 | 0.9610 | 0.7858 | 0.6264 |
| Temporal | 37 | full_TC_MC | 476 | 0.7511 | 0.7305 | 0.7766 | -0.9671 | 0.9931 | 0.7218 | 0.6135 |

**Sequence-mean correlations** (legacy, n = sequences)

| Domain | QP | Metric | PCC | SRCC | n |
|---|---|---|---|---|---|
| Spatial | 22 | baseline_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | baseline_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | baseline_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | baseline_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | fast_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | fast_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | fast_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | fast_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | fast_SC_u | 0.6357 | 0.6000 | 4 |
| Spatial | 22 | fast_SC_v | 0.9838 | 1.0000 | 4 |
| Spatial | 22 | fast_Colorfulness | 0.3175 | -0.2000 | 4 |
| Temporal | 22 | fast_MVC | 0.9465 | 1.0000 | 4 |
| Temporal | 22 | fast_TC_SAD | 0.6987 | 0.8000 | 4 |
| Temporal | 22 | fast_MV_sat_frac | 0.8713 | 1.0000 | 4 |
| Temporal | 22 | fast_mean_mv_mag | 0.8582 | 1.0000 | 4 |
| Spatial | 22 | full_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | full_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | full_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | full_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | full_SC_u | 0.6357 | 0.6000 | 4 |
| Spatial | 22 | full_SC_v | 0.9838 | 1.0000 | 4 |
| Spatial | 22 | full_Colorfulness | 0.3175 | -0.2000 | 4 |
| Temporal | 22 | full_MVC | 0.9465 | 1.0000 | 4 |
| Temporal | 22 | full_TC_SAD | 0.6987 | 0.8000 | 4 |
| Temporal | 22 | full_TC_MC | 0.7252 | 0.8000 | 4 |
| Temporal | 22 | full_MV_sat_frac | 0.8713 | 1.0000 | 4 |
| Temporal | 22 | full_mean_mv_mag | 0.8582 | 1.0000 | 4 |
| Temporal | 22 | full_intra_frac | 0.9966 | 1.0000 | 4 |
| Spatial | 27 | baseline_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | baseline_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | baseline_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | baseline_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | fast_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | fast_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | fast_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | fast_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | fast_SC_u | -0.0148 | 0.0000 | 4 |
| Spatial | 27 | fast_SC_v | 0.6411 | 0.8000 | 4 |
| Spatial | 27 | fast_Colorfulness | -0.3602 | -0.4000 | 4 |
| Temporal | 27 | fast_MVC | 0.9665 | 1.0000 | 4 |
| Temporal | 27 | fast_TC_SAD | 0.7269 | 0.8000 | 4 |
| Temporal | 27 | fast_MV_sat_frac | 0.8854 | 1.0000 | 4 |
| Temporal | 27 | fast_mean_mv_mag | 0.8916 | 1.0000 | 4 |
| Spatial | 27 | full_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | full_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | full_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | full_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | full_SC_u | -0.0148 | 0.0000 | 4 |
| Spatial | 27 | full_SC_v | 0.6411 | 0.8000 | 4 |
| Spatial | 27 | full_Colorfulness | -0.3602 | -0.4000 | 4 |
| Temporal | 27 | full_MVC | 0.9665 | 1.0000 | 4 |
| Temporal | 27 | full_TC_SAD | 0.7269 | 0.8000 | 4 |
| Temporal | 27 | full_TC_MC | 0.7301 | 0.8000 | 4 |
| Temporal | 27 | full_MV_sat_frac | 0.8854 | 1.0000 | 4 |
| Temporal | 27 | full_mean_mv_mag | 0.8916 | 1.0000 | 4 |
| Temporal | 27 | full_intra_frac | 0.9974 | 1.0000 | 4 |
| Spatial | 32 | baseline_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | baseline_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | baseline_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | baseline_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | fast_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | fast_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | fast_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | fast_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | fast_SC_u | -0.0030 | 0.0000 | 4 |
| Spatial | 32 | fast_SC_v | 0.6512 | 0.8000 | 4 |
| Spatial | 32 | fast_Colorfulness | -0.3388 | -0.4000 | 4 |
| Temporal | 32 | fast_MVC | 0.9542 | 1.0000 | 4 |
| Temporal | 32 | fast_TC_SAD | 0.7631 | 0.8000 | 4 |
| Temporal | 32 | fast_MV_sat_frac | 0.9120 | 1.0000 | 4 |
| Temporal | 32 | fast_mean_mv_mag | 0.9004 | 1.0000 | 4 |
| Spatial | 32 | full_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | full_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | full_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | full_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | full_SC_u | -0.0030 | 0.0000 | 4 |
| Spatial | 32 | full_SC_v | 0.6512 | 0.8000 | 4 |
| Spatial | 32 | full_Colorfulness | -0.3388 | -0.4000 | 4 |
| Temporal | 32 | full_MVC | 0.9542 | 1.0000 | 4 |
| Temporal | 32 | full_TC_SAD | 0.7631 | 0.8000 | 4 |
| Temporal | 32 | full_TC_MC | 0.7744 | 0.8000 | 4 |
| Temporal | 32 | full_MV_sat_frac | 0.9120 | 1.0000 | 4 |
| Temporal | 32 | full_mean_mv_mag | 0.9004 | 1.0000 | 4 |
| Temporal | 32 | full_intra_frac | 0.9998 | 1.0000 | 4 |
| Spatial | 37 | baseline_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | baseline_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | baseline_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | baseline_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | fast_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | fast_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | fast_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | fast_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | fast_SC_u | 0.1728 | 0.0000 | 4 |
| Spatial | 37 | fast_SC_v | 0.7739 | 0.8000 | 4 |
| Spatial | 37 | fast_Colorfulness | -0.1650 | -0.4000 | 4 |
| Temporal | 37 | fast_MVC | 0.9430 | 1.0000 | 4 |
| Temporal | 37 | fast_TC_SAD | 0.8114 | 0.8000 | 4 |
| Temporal | 37 | fast_MV_sat_frac | 0.9421 | 1.0000 | 4 |
| Temporal | 37 | fast_mean_mv_mag | 0.9193 | 1.0000 | 4 |
| Spatial | 37 | full_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | full_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | full_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | full_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | full_SC_u | 0.1728 | 0.0000 | 4 |
| Spatial | 37 | full_SC_v | 0.7739 | 0.8000 | 4 |
| Spatial | 37 | full_Colorfulness | -0.1650 | -0.4000 | 4 |
| Temporal | 37 | full_MVC | 0.9430 | 1.0000 | 4 |
| Temporal | 37 | full_TC_SAD | 0.8114 | 0.8000 | 4 |
| Temporal | 37 | full_TC_MC | 0.8211 | 0.8000 | 4 |
| Temporal | 37 | full_MV_sat_frac | 0.9421 | 1.0000 | 4 |
| Temporal | 37 | full_mean_mv_mag | 0.9193 | 1.0000 | 4 |
| Temporal | 37 | full_intra_frac | 0.9959 | 1.0000 | 4 |


#### Gate 1 — verdict and analysis

**Gate 1 met.** The harness runs end to end (EVCA extraction → per-frame x265 ground
truth → frame-level and sequence-mean correlations → ledger), and the three profiles
were re-run with the Phase 0 fixes in place. The numbers above are the post-fix
baseline that Phases 2–5 are measured against.

Additional Phase 0 fix found by the new tests and folded into this gate:

| # | Fix | Commit | Default output change |
|---|-----|--------|----------------------|
| 0.10 | `load_gop_optimized` raised `BufferError: cannot close exported pointers exist` whenever chroma was enabled. The `if 'X' in locals(): del X` idiom materialises the frame's `f_locals` snapshot, which then holds its own reference to the last chroma view, so the mmap still had an exported buffer at `close()`. Replaced with plain rebinding. | `dbfa6cc` | `--loader optimized` with `-cc`/`-cf` went from crashing to working |

**Frame alignment verified against the encoder.** A 1080p probe sequence with a
4 px/frame pan starting at frame 8 and a hard cut at frame 14 was encoded LDP at
QP 32. The x265 P-frame bits jump at frame_idx 8 (712 → 3800 bits) and spike at
frame_idx 14 (829 072 bits); EVCA's `TC`, `TC_SAD`, `mean_mv_mag` first become
non-zero at row 8 and `TC_MC`/`intra_frac` peak at row 14. Identical indices, so
EVCA row `f` ↔ LDP P-frame `f` is the correct alignment. `mean_mv_mag` was exactly
4.00 during the pan, and `TC_MC` fell to 0.8 against a raw `TC` of 167.7, which is
the Phase 0.1 registration fix working on real-resolution content. This probe is
now pinned by `tests/test_frame_alignment.py` (synthetic, no ffmpeg needed).

**Pooled correlations are dominated by between-sequence variance.** Mean *within*-
sequence frame-level PCC against `TC_gt`, averaged over the four sequences:

| metric | QP 22 | QP 27 | QP 32 | QP 37 |
|---|---|---|---|---|
| `TC` (baseline) | 0.179 | 0.301 | 0.271 | 0.283 |
| `TC2` (baseline) | 0.234 | 0.543 | 0.487 | 0.412 |
| `TC_SAD` | 0.034 | 0.214 | 0.269 | 0.407 |
| `MVC` | −0.027 | 0.145 | 0.174 | 0.170 |
| `TC_MC` | **0.287** | **0.385** | **0.405** | **0.516** |

`MVC` has the highest *pooled* PCC of any temporal metric (0.86–0.89) but the
second-*lowest* within-sequence PCC (0.15–0.17). Its pooled score is almost entirely
the between-sequence effect that high-motion content costs more bits; it barely
tracks frame-to-frame variation inside a sequence. `TC_MC` is the best within-sequence
temporal metric at every QP, which is the ordering Phases 3–4 should try to improve.
The sequence-level block-bootstrap CIs are correspondingly near-vacuous (frequently
spanning [−0.97, +0.99]) because there are only four sequences: with n = 4 groups the
block bootstrap has very few distinct resamples. **Consequence for Gates 3 and 4:** the
specified decision rule (lower bound of the 95 % CI of pooled frame-level PCC of
`TC_MC`) is applied as written, but the frame-level CI is used for it, and the
per-sequence table is reported alongside every gate, since the pooled statistic can be
moved by between-sequence effects that say nothing about per-frame prediction quality.

**Motion search is saturated on half the corpus.** Per-sequence means over frames ≥ 1:

| sequence | `MV_sat_frac` | `mean_mv_mag` | `intra_frac` | `TC_SAD` PCC @ QP 32 |
|---|---|---|---|---|
| HoneyBee | 0.2 % | 0.08 | 7.0 % | 0.569 |
| Bosphorus | 2.8 % | 1.96 | 13.6 % | −0.054 |
| ReadySteadyGo | 50.5 % | 4.18 | 20.7 % | 0.708 |
| YachtRide | 63.0 % | 4.62 | 36.8 % | −0.146 |

Overall `MV_sat_frac` is **29.1 %**, against the Phase 3 acceptance threshold of < 5 %.
On YachtRide and ReadySteadyGo the majority of blocks pick a motion vector on the
boundary of the ±6 px pattern, i.e. the true motion is outside the search range and
the reported SAD measures search failure rather than content complexity. This is the
direct motivation for the hierarchical search in Phase 3, and it is the most likely
explanation for `TC_SAD` correlating *negatively* with bits on YachtRide.

**Throughput** (RTX 5060 Ti, CUDA, 1080p, `--loader optimized`, 480 frames total):
baseline 375 fps, fast profile 302 fps, full profile 247 fps. The full profile is
already well above the Phase 3 target of 100 fps at 1080p, leaving headroom for a
more expensive search.

### Run `gate2-defaults` — 2026-08-21 19:46

- Phase: Phase 2 (defaults, byte-identity check)
- Commit: `2dc129d1c7a0ee34ab2b445dc9f0d20722c448cb`
- Subset: **fast** (120 frames), sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Device: `auto`, loader: `optimized`, profiles: baseline, fast, full
- Extra EVCA args: `(none)`
- Bootstrap: 1000 resamples, seed 12345
- Results: `validation/results/gate2-defaults_2dc129d1`

**Throughput**

| profile | frames | seconds | fps |
|---|---|---|---|
| baseline | 480 | 0.96 | 500.00 |
| fast | 480 | 1.45 | 331.03 |
| full | 480 | 1.91 | 251.31 |

**Frame-level pooled correlations** (CI = 95 % bootstrap; `blk` = sequence-level block bootstrap)

| Domain | QP | metric | n | PCC | PCC_lo | PCC_hi | PCC_blk_lo | PCC_blk_hi | SRCC | PCC_log |
|---|---|---|---|---|---|---|---|---|---|---|
| Spatial | 22 | baseline_SC | 480 | 0.7805 | 0.7373 | 0.8185 | -0.7762 | 0.9992 | 0.7231 | 0.7671 |
| Temporal | 22 | baseline_TC | 476 | 0.4371 | 0.3951 | 0.4809 | -0.6931 | 0.9854 | 0.7826 | 0.5339 |
| Temporal | 22 | fast_MVC | 476 | 0.8588 | 0.8329 | 0.8834 | 0.2413 | 0.9822 | 0.8261 | 0.9045 |
| Temporal | 22 | fast_TC_SAD | 476 | 0.5188 | 0.4621 | 0.5919 | -0.1833 | 0.9375 | 0.7707 | 0.5668 |
| Temporal | 22 | full_TC_MC | 476 | 0.6437 | 0.6054 | 0.6893 | -0.9725 | 0.9841 | 0.7156 | 0.6367 |
| Spatial | 27 | baseline_SC | 480 | 0.9817 | 0.9785 | 0.9843 | 0.9018 | 0.9918 | 0.9886 | 0.9865 |
| Temporal | 27 | baseline_TC | 476 | 0.4879 | 0.4488 | 0.5272 | -0.6945 | 0.9819 | 0.7827 | 0.5972 |
| Temporal | 27 | fast_MVC | 476 | 0.8898 | 0.8691 | 0.9101 | 0.3719 | 0.9860 | 0.8528 | 0.9091 |
| Temporal | 27 | fast_TC_SAD | 476 | 0.5583 | 0.5081 | 0.6224 | -0.1365 | 0.9477 | 0.7699 | 0.5661 |
| Temporal | 27 | full_TC_MC | 476 | 0.6550 | 0.6238 | 0.6928 | -0.9799 | 0.9894 | 0.7115 | 0.5514 |
| Spatial | 32 | baseline_SC | 480 | 0.9899 | 0.9877 | 0.9916 | 0.9534 | 0.9987 | 0.9922 | 0.9849 |
| Temporal | 32 | baseline_TC | 476 | 0.5237 | 0.4856 | 0.5608 | -0.6881 | 0.9784 | 0.7830 | 0.6319 |
| Temporal | 32 | fast_MVC | 476 | 0.8849 | 0.8655 | 0.9030 | 0.4031 | 0.9854 | 0.8581 | 0.9021 |
| Temporal | 32 | fast_TC_SAD | 476 | 0.5992 | 0.5552 | 0.6574 | -0.0913 | 0.9546 | 0.7710 | 0.5880 |
| Temporal | 32 | full_TC_MC | 476 | 0.7007 | 0.6739 | 0.7334 | -0.9767 | 0.9914 | 0.7109 | 0.5667 |
| Spatial | 37 | baseline_SC | 480 | 0.9778 | 0.9737 | 0.9813 | 0.6971 | 0.9997 | 0.9624 | 0.9541 |
| Temporal | 37 | baseline_TC | 476 | 0.5769 | 0.5407 | 0.6119 | -0.6658 | 0.9764 | 0.7870 | 0.6721 |
| Temporal | 37 | fast_MVC | 476 | 0.8814 | 0.8621 | 0.8984 | 0.4244 | 0.9835 | 0.8615 | 0.9002 |
| Temporal | 37 | fast_TC_SAD | 476 | 0.6530 | 0.6144 | 0.7040 | 0.0272 | 0.9610 | 0.7858 | 0.6264 |
| Temporal | 37 | full_TC_MC | 476 | 0.7511 | 0.7305 | 0.7766 | -0.9671 | 0.9931 | 0.7218 | 0.6135 |

**Sequence-mean correlations** (legacy, n = sequences)

| Domain | QP | Metric | PCC | SRCC | n |
|---|---|---|---|---|---|
| Spatial | 22 | baseline_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | baseline_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | baseline_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | baseline_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | fast_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | fast_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | fast_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | fast_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | fast_SC_u | 0.6357 | 0.6000 | 4 |
| Spatial | 22 | fast_SC_v | 0.9838 | 1.0000 | 4 |
| Spatial | 22 | fast_Colorfulness | 0.3175 | -0.2000 | 4 |
| Temporal | 22 | fast_MVC | 0.9465 | 1.0000 | 4 |
| Temporal | 22 | fast_TC_SAD | 0.6987 | 0.8000 | 4 |
| Temporal | 22 | fast_MV_sat_frac | 0.8713 | 1.0000 | 4 |
| Temporal | 22 | fast_mean_mv_mag | 0.8582 | 1.0000 | 4 |
| Spatial | 22 | full_B | -0.9415 | -0.8000 | 4 |
| Spatial | 22 | full_SC | 0.7679 | 0.8000 | 4 |
| Temporal | 22 | full_TC | 0.4605 | 0.8000 | 4 |
| Temporal | 22 | full_TC2 | 0.4806 | 0.8000 | 4 |
| Spatial | 22 | full_SC_u | 0.6357 | 0.6000 | 4 |
| Spatial | 22 | full_SC_v | 0.9838 | 1.0000 | 4 |
| Spatial | 22 | full_Colorfulness | 0.3175 | -0.2000 | 4 |
| Temporal | 22 | full_MVC | 0.9465 | 1.0000 | 4 |
| Temporal | 22 | full_TC_SAD | 0.6987 | 0.8000 | 4 |
| Temporal | 22 | full_TC_MC | 0.7252 | 0.8000 | 4 |
| Temporal | 22 | full_MV_sat_frac | 0.8713 | 1.0000 | 4 |
| Temporal | 22 | full_mean_mv_mag | 0.8582 | 1.0000 | 4 |
| Temporal | 22 | full_intra_frac | 0.9966 | 1.0000 | 4 |
| Spatial | 27 | baseline_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | baseline_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | baseline_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | baseline_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | fast_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | fast_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | fast_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | fast_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | fast_SC_u | -0.0148 | 0.0000 | 4 |
| Spatial | 27 | fast_SC_v | 0.6411 | 0.8000 | 4 |
| Spatial | 27 | fast_Colorfulness | -0.3602 | -0.4000 | 4 |
| Temporal | 27 | fast_MVC | 0.9665 | 1.0000 | 4 |
| Temporal | 27 | fast_TC_SAD | 0.7269 | 0.8000 | 4 |
| Temporal | 27 | fast_MV_sat_frac | 0.8854 | 1.0000 | 4 |
| Temporal | 27 | fast_mean_mv_mag | 0.8916 | 1.0000 | 4 |
| Spatial | 27 | full_B | -0.5000 | -0.4000 | 4 |
| Spatial | 27 | full_SC | 0.9999 | 1.0000 | 4 |
| Temporal | 27 | full_TC | 0.5067 | 0.8000 | 4 |
| Temporal | 27 | full_TC2 | 0.5303 | 0.8000 | 4 |
| Spatial | 27 | full_SC_u | -0.0148 | 0.0000 | 4 |
| Spatial | 27 | full_SC_v | 0.6411 | 0.8000 | 4 |
| Spatial | 27 | full_Colorfulness | -0.3602 | -0.4000 | 4 |
| Temporal | 27 | full_MVC | 0.9665 | 1.0000 | 4 |
| Temporal | 27 | full_TC_SAD | 0.7269 | 0.8000 | 4 |
| Temporal | 27 | full_TC_MC | 0.7301 | 0.8000 | 4 |
| Temporal | 27 | full_MV_sat_frac | 0.8854 | 1.0000 | 4 |
| Temporal | 27 | full_mean_mv_mag | 0.8916 | 1.0000 | 4 |
| Temporal | 27 | full_intra_frac | 0.9974 | 1.0000 | 4 |
| Spatial | 32 | baseline_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | baseline_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | baseline_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | baseline_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | fast_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | fast_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | fast_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | fast_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | fast_SC_u | -0.0030 | 0.0000 | 4 |
| Spatial | 32 | fast_SC_v | 0.6512 | 0.8000 | 4 |
| Spatial | 32 | fast_Colorfulness | -0.3388 | -0.4000 | 4 |
| Temporal | 32 | fast_MVC | 0.9542 | 1.0000 | 4 |
| Temporal | 32 | fast_TC_SAD | 0.7631 | 0.8000 | 4 |
| Temporal | 32 | fast_MV_sat_frac | 0.9120 | 1.0000 | 4 |
| Temporal | 32 | fast_mean_mv_mag | 0.9004 | 1.0000 | 4 |
| Spatial | 32 | full_B | -0.5088 | -0.4000 | 4 |
| Spatial | 32 | full_SC | 0.9974 | 1.0000 | 4 |
| Temporal | 32 | full_TC | 0.5441 | 0.8000 | 4 |
| Temporal | 32 | full_TC2 | 0.5628 | 0.8000 | 4 |
| Spatial | 32 | full_SC_u | -0.0030 | 0.0000 | 4 |
| Spatial | 32 | full_SC_v | 0.6512 | 0.8000 | 4 |
| Spatial | 32 | full_Colorfulness | -0.3388 | -0.4000 | 4 |
| Temporal | 32 | full_MVC | 0.9542 | 1.0000 | 4 |
| Temporal | 32 | full_TC_SAD | 0.7631 | 0.8000 | 4 |
| Temporal | 32 | full_TC_MC | 0.7744 | 0.8000 | 4 |
| Temporal | 32 | full_MV_sat_frac | 0.9120 | 1.0000 | 4 |
| Temporal | 32 | full_mean_mv_mag | 0.9004 | 1.0000 | 4 |
| Temporal | 32 | full_intra_frac | 0.9998 | 1.0000 | 4 |
| Spatial | 37 | baseline_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | baseline_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | baseline_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | baseline_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | fast_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | fast_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | fast_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | fast_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | fast_SC_u | 0.1728 | 0.0000 | 4 |
| Spatial | 37 | fast_SC_v | 0.7739 | 0.8000 | 4 |
| Spatial | 37 | fast_Colorfulness | -0.1650 | -0.4000 | 4 |
| Temporal | 37 | fast_MVC | 0.9430 | 1.0000 | 4 |
| Temporal | 37 | fast_TC_SAD | 0.8114 | 0.8000 | 4 |
| Temporal | 37 | fast_MV_sat_frac | 0.9421 | 1.0000 | 4 |
| Temporal | 37 | fast_mean_mv_mag | 0.9193 | 1.0000 | 4 |
| Spatial | 37 | full_B | -0.6505 | -0.4000 | 4 |
| Spatial | 37 | full_SC | 0.9784 | 1.0000 | 4 |
| Temporal | 37 | full_TC | 0.6036 | 0.8000 | 4 |
| Temporal | 37 | full_TC2 | 0.6182 | 0.8000 | 4 |
| Spatial | 37 | full_SC_u | 0.1728 | 0.0000 | 4 |
| Spatial | 37 | full_SC_v | 0.7739 | 0.8000 | 4 |
| Spatial | 37 | full_Colorfulness | -0.1650 | -0.4000 | 4 |
| Temporal | 37 | full_MVC | 0.9430 | 1.0000 | 4 |
| Temporal | 37 | full_TC_SAD | 0.8114 | 0.8000 | 4 |
| Temporal | 37 | full_TC_MC | 0.8211 | 0.8000 | 4 |
| Temporal | 37 | full_MV_sat_frac | 0.9421 | 1.0000 | 4 |
| Temporal | 37 | full_mean_mv_mag | 0.9193 | 1.0000 | 4 |
| Temporal | 37 | full_intra_frac | 0.9959 | 1.0000 | 4 |


### Ablation `gate2-ablation` — 2026-08-21 19:49

- Phase: Phase 2 (post-fix Iteration 1-4 ranking)
- Commit: `2dc129d1c7a0ee34ab2b445dc9f0d20722c448cb`
- Subset: **fast**, profile `full`, ranking metric `full_TC_MC`
- Axes: `mc` ∈ {dense_smooth, dense}; `gate` ∈ {intra, none}
- Extra args: `(none)`
- Sequences: YachtRide, ReadySteadyGo, HoneyBee, Bosphorus
- Results: `validation/results/gate2-ablation_2dc129d1`

Values are averaged over QPs 22/27/32/37. `PCC_lo_mean` is the gate ranking key; `perseq_PCC_mean` is the mean within-sequence PCC.

| variant | PCC_mean | PCC_lo_mean | PCC_hi_mean | SRCC_mean | perseq_PCC_mean | fps |
|---|---|---|---|---|---|---|
| mc=dense_smooth gate=none | 0.6965 | 0.6653 | 0.7340 | 0.7166 | 0.3659 | 243.6548 |
| mc=dense_smooth gate=intra | 0.6876 | 0.6584 | 0.7230 | 0.7150 | 0.3984 | 237.6238 |
| mc=dense gate=none | 0.6438 | 0.6125 | 0.6823 | 0.6966 | 0.3256 | 248.7047 |
| mc=dense gate=intra | 0.6311 | 0.6020 | 0.6674 | 0.6847 | 0.3605 | 243.6548 |


#### Gate 2 — verdict

**Gate 2 met.** With every new flag at its default, the per-frame CSVs of all three
profiles on all four sequences are **byte-identical** to the Gate 1 output
(`md5sum -c`, 12/12 OK), so the strategy refactor changed no default behaviour.

Phase 2 items and how they landed:

| Item | Status |
|---|---|
| `--me`, `--me-subpel`, `--me-predictor`, `--me-lambda`, `--me-merge`, `--me-criterion` | Flags parsed; non-default values raise `NotImplementedError` until Phase 3, so an ablation can never silently report a variant it did not run |
| `--mc {dense_smooth,dense,block,obmc}`, `--mc-smooth {gauss,median,none}`, `--residual-dc`, `--gate {intra,none}` | Implemented (`libs/motion_compensation.py`) |
| MC as a class hierarchy that `TemporalState` delegates to | Done: `MotionCompensator` → `DenseMC` / `BlockMC` / `OBMC`; `TemporalState.compensator` |
| `--dct-impl {torch_dct,matmul}` | Done. **Default stays `matmul`** — it was already the shipped behaviour (commit `f144ce2`), and it is faster everywhere tested |
| Remove per-GOP `.cpu()` syncs | Already done on the base branch (`f144ce2`); verified there is exactly one host transfer per file, after the GOP loop |
| `--preset iter4` | Added early (Phase 4 needs it) and pinned by a test asserting it equals today's defaults |

**DCT backend benchmark** (`validation/bench_dct.py`, 32 frames of 1080p worth of
32×32 blocks per batch). Max relative disagreement `3.4e-07` on CUDA and `2.6e-07` on
CPU, both far inside the 1e-4 tolerance:

| device | impl | ms/batch | frames/s | speedup |
|---|---|---|---|---|
| CUDA | matmul | 2.792 | 11462 | 1.00× |
| CUDA | torch_dct | 30.980 | 1033 | 0.09× |
| CPU | matmul | 13.979 | 572 | 1.00× |
| CPU | torch_dct | 97.399 | 82 | 0.14× |

matmul is 11× faster on CUDA and 7× on CPU, so the default is left at `matmul`.
`torch_dct` is retained only as the ablation reference and is an optional import.

**Post-fix Iteration 1–4 ranking** (`--mc dense_smooth|dense` × `--gate intra|none`,
fast subset, ranking metric `TC_MC`, values averaged over QPs 22/27/32/37):

| variant | PCC | CI lo | per-seq PCC | fps |
|---|---|---|---|---|
| `dense_smooth` + `gate none` | 0.6965 | 0.6653 | 0.3659 | 243.7 |
| `dense_smooth` + `gate intra` | 0.6876 | 0.6584 | **0.3984** | 237.6 |
| `dense` + `gate none` | 0.6438 | 0.6125 | 0.3256 | 248.7 |
| `dense` + `gate intra` | 0.6311 | 0.6020 | 0.3605 | 243.7 |

Two findings. First, **Gaussian smoothing of the MV field still helps after the
Phase 0 fixes**: `dense_smooth` beats `dense` by ≈ 0.05 PCC at both gate settings, so
Iteration 4's smoothing choice was not an artefact of the misregistered MV field.
Second, **the two gate settings rank differently depending on the statistic**: by the
gate rule (CI lower bound of pooled PCC) `gate none` edges ahead (0.6653 vs 0.6584),
but by mean within-sequence PCC `gate intra` is clearly better (0.398 vs 0.366), and
the pooled CIs overlap almost completely. No default is changed here (Phase 2 changes
no defaults).

> **Resolved.** This entry originally carried the gate disagreement forward to Gate 4.
> The disagreement was an artefact of the ranking rule, not a real tie: the pooled CI
> lower bound is dominated by between-sequence variance and was replaced (see
> [Ranking rule change](#ranking-rule-change-implemented)). Under the revised rule
> **both `gate intra` variants beat both `gate none` variants**, with the gate axis
> separating variants roughly six times as strongly as the `mc` axis. `--gate intra`
> stays the default and the Gate 4 deferral is closed — see
> [Gate 2 resolution](#gate-2-resolution--ranking-rule-change-and-the-gate-axis).
> The first finding is unaffected: `dense_smooth` still beats `dense` at both gate
> settings under the revised rule.

## Cross-machine re-baseline (macOS) — metric comparison

> **The corpus is not the one used for Gates 1 and 2.** `Bosphorus` is not present on
> this machine and `FoodMarket` takes its place, so three of four sequences are shared.
> Pooled statistics in this section are **not** comparable to the Gate 1/Gate 2 tables
> above; per-sequence and within-sequence statistics are. The substance of this section
> is that this distinction is not pedantic — it decides which metric wins.

### Run `macos-smoke` — 2026-08-24 11:27

- Phase: harness verification + metric comparison (macOS)
- Commit: `719bdf3330d9fc1d6e2b68ccadd771021997b757`, working tree carrying the
  `validation/sequences.json` repoint, the `enabled` flag in `validation/ground_truth.py`,
  and the deletion of the orphaned `libs/motion_estimation.py`
- Machine: macOS 15.7.7, Apple Silicon (arm64), conda env `EVCA`
  (torch 2.2.0, device `mps`, numpy 1.26.3, scipy 1.17.1, pandas 2.2.0);
  ffmpeg 8.1.1 with libx265
- Subset: **fast** (120 frames), sequences: YachtRide, ReadySteadyGo, HoneyBee, FoodMarket
- Missing sequences (skipped): Bosphorus
- Device: `auto` (resolved to `mps`), loader: `optimized`, profiles: baseline, fast, full
- Extra EVCA args: `(none)`
- Bootstrap: 1000 resamples, seed 12345
- Results: `validation/results/macos-smoke_719bdf33` (run with `--skip-ledger`;
  this section is written by hand)

**Throughput** (MPS, 1080p, `--loader optimized`, 480 frames total). The RTX 5060 Ti
figures from Gate 2 are quoted for scale; the full profile here is **below** the Phase 3
target of 100 fps at 1080p, which was set with CUDA headroom in mind.

| profile | frames | seconds | fps | fps (RTX 5060 Ti, Gate 2) |
|---|---|---|---|---|
| baseline | 480 | 2.77 | 173.3 | 500.0 |
| fast | 480 | 5.59 | 85.9 | 331.0 |
| full | 480 | 13.16 | 36.5 | 251.3 |

**Frame-level pooled correlations** (CI = 95 % bootstrap; `blk` = sequence-level block bootstrap)

| Domain | QP | metric | n | PCC | PCC_lo | PCC_hi | PCC_blk_lo | PCC_blk_hi | SRCC | PCC_log |
|---|---|---|---|---|---|---|---|---|---|---|
| Spatial | 22 | baseline_SC | 480 | 0.2686 | 0.2126 | 0.3252 | -0.7762 | 0.9816 | 0.3638 | 0.2743 |
| Temporal | 22 | baseline_TC | 476 | 0.4893 | 0.4199 | 0.5558 | -0.6931 | 0.9898 | 0.4212 | 0.6007 |
| Temporal | 22 | fast_MVC | 476 | 0.8254 | 0.7965 | 0.8524 | -0.0002 | 0.9822 | 0.8421 | 0.8277 |
| Temporal | 22 | fast_TC_SAD | 476 | 0.6227 | 0.5472 | 0.6982 | -0.1833 | 0.9541 | 0.7042 | 0.6750 |
| Temporal | 22 | full_TC_MC | 476 | 0.7735 | 0.7258 | 0.8189 | 0.1467 | 0.9847 | 0.7696 | 0.8045 |
| Spatial | 27 | baseline_SC | 480 | 0.8607 | 0.8282 | 0.8869 | 0.7317 | 0.9960 | 0.9124 | 0.8505 |
| Temporal | 27 | baseline_TC | 476 | 0.4242 | 0.3580 | 0.4860 | -0.8088 | 0.9842 | 0.3138 | 0.6903 |
| Temporal | 27 | fast_MVC | 476 | 0.5260 | 0.4667 | 0.5814 | -0.7416 | 0.9860 | 0.6390 | 0.6610 |
| Temporal | 27 | fast_TC_SAD | 476 | 0.4377 | 0.3767 | 0.4958 | -0.8402 | 0.9559 | 0.5406 | 0.6244 |
| Temporal | 27 | full_TC_MC | 476 | 0.5507 | 0.4994 | 0.5939 | -0.8124 | 0.9894 | 0.5854 | 0.7106 |
| Spatial | 32 | baseline_SC | 480 | 0.9778 | 0.9722 | 0.9820 | 0.9123 | 0.9908 | 0.9711 | 0.9758 |
| Temporal | 32 | baseline_TC | 476 | 0.3874 | 0.3201 | 0.4483 | -0.8544 | 0.9805 | 0.3945 | 0.7050 |
| Temporal | 32 | fast_MVC | 476 | 0.4259 | 0.3594 | 0.4897 | -0.7322 | 0.9854 | 0.4510 | 0.6203 |
| Temporal | 32 | fast_TC_SAD | 476 | 0.3803 | 0.3177 | 0.4354 | -0.8288 | 0.9566 | 0.4220 | 0.6128 |
| Temporal | 32 | full_TC_MC | 476 | 0.4710 | 0.4147 | 0.5196 | -0.8359 | 0.9914 | 0.4471 | 0.6846 |
| Spatial | 37 | baseline_SC | 480 | 0.9367 | 0.9229 | 0.9481 | -0.0371 | 0.9973 | 0.6640 | 0.9223 |
| Temporal | 37 | baseline_TC | 476 | 0.4077 | 0.3394 | 0.4699 | -0.8719 | 0.9790 | 0.4044 | 0.7121 |
| Temporal | 37 | fast_MVC | 476 | 0.3965 | 0.3290 | 0.4646 | -0.7232 | 0.9835 | 0.4121 | 0.6001 |
| Temporal | 37 | fast_TC_SAD | 476 | 0.3908 | 0.3298 | 0.4463 | -0.8000 | 0.9611 | 0.4016 | 0.6113 |
| Temporal | 37 | full_TC_MC | 476 | 0.4626 | 0.4033 | 0.5148 | -0.8354 | 0.9931 | 0.4189 | 0.6741 |

#### Verification — the Phase 1 diagnostics reproduce exactly

Every shared sequence matches the Gate 1 per-sequence table to the printed precision,
including the `TC_SAD` correlation column, which depends on the x265 ground truth and
therefore exercises the encoder path as well:

| sequence | `MV_sat_frac` | `mean_mv_mag` | `intra_frac` | `TC_SAD` PCC @ QP 32 |
|---|---|---|---|---|
| HoneyBee | 0.2 % / **0.2 %** | 0.08 / **0.08** | 7.0 % / **7.0 %** | 0.569 / **0.569** |
| ReadySteadyGo | 50.5 % / **50.5 %** | 4.18 / **4.18** | 20.7 % / **20.7 %** | 0.708 / **0.708** |
| YachtRide | 63.0 % / **63.0 %** | 4.62 / **4.62** | 36.8 % / **36.8 %** | -0.146 / **-0.146** |

(Gate 1 on Linux / this run on macOS.) Across a different OS, torch 2.2.0 vs 2.13.0,
MPS vs CUDA, and ffmpeg 8.1.1 vs n9.0.1. Both the EVCA pipeline and the ground-truth
generation are deterministic and machine-independent; cross-machine comparison is
trustworthy at the per-sequence level.

`FoodMarket` is a markedly harder sequence than the `Bosphorus` it replaces
(2.8 % saturation): `MV_sat_frac` 43.7 %, `mean_mv_mag` 4.19, and `intra_frac`
**61.5 %** — motion compensation loses to intra on nearly two thirds of blocks. Corpus
`MV_sat_frac` rises from **29.1 % to 39.4 %** against the Phase 3 threshold of < 5 %.

#### Metric comparison — pooled versus within-sequence

Averaged over QPs 22/27/32/37, sorted by within-sequence PCC. `gap` is
`pooled - within`: how much of a metric's pooled score is between-sequence content
ranking rather than per-frame prediction.

**Temporal metrics** (vs Low-Delay-P frame bits)

| metric | pooled PCC | within-seq PCC | gap | pooled SRCC | PCC_log | frame CI width | block CI width |
|---|---|---|---|---|---|---|---|
| `TC_MC` | 0.564 | **0.343** | +0.222 | 0.555 | 0.718 | 0.101 | 1.574 |
| `TC` | 0.427 | 0.318 | +0.109 | 0.383 | 0.677 | 0.131 | 1.790 |
| `TC2` | 0.462 | 0.282 | +0.180 | 0.369 | 0.698 | 0.121 | 1.620 |
| `TC_SAD` | 0.458 | 0.142 | +0.315 | 0.517 | 0.631 | 0.126 | 1.620 |
| `MVC` | 0.543 | 0.027 | +0.516 | 0.586 | 0.677 | 0.109 | 1.534 |
| `mean_mv_mag` | **0.743** | -0.029 | +0.772 | 0.571 | 0.870 | 0.073 | 1.054 |
| `intra_frac` | 0.500 | -0.080 | +0.580 | 0.557 | 0.626 | 0.113 | 1.568 |
| `MV_sat_frac` | 0.656 | -0.143 | +0.799 | 0.600 | 0.725 | 0.095 | 1.131 |

**Spatial metrics** (vs All-Intra frame bits)

| metric | pooled PCC | within-seq PCC | gap |
|---|---|---|---|
| `SC` | 0.761 | **0.896** | -0.136 |
| `SC_u` | -0.187 | 0.717 | **-0.904** |
| `SC_v` | 0.268 | 0.556 | -0.288 |
| `Colorfulness` | -0.411 | 0.337 | -0.748 |
| `B` | -0.234 | -0.067 | -0.167 |

**Per-sequence within-frame PCC**, averaged over QPs:

| sequence | `TC` | `TC2` | `TC_SAD` | `MVC` | `TC_MC` |
|---|---|---|---|---|---|
| HoneyBee | 0.538 | 0.072 | 0.563 | 0.110 | 0.537 |
| ReadySteadyGo | 0.261 | 0.344 | 0.620 | 0.537 | 0.595 |
| YachtRide | 0.514 | 0.587 | **-0.198** | **-0.144** | 0.483 |
| FoodMarket | -0.042 | 0.126 | **-0.415** | **-0.394** | -0.245 |

#### Interpretation

**The pooled ranking is close to the inverse of the within-sequence ranking.** Ranked
by pooled PCC the best temporal metric is `mean_mv_mag` (0.743), with `MV_sat_frac`
second (0.656), both ahead of `TC_MC` (0.564). Neither is a complexity metric —
`mean_mv_mag` is the average motion-vector length and `MV_sat_frac` is the fraction of
blocks that hit the search-range limit; both were added in Phase 1 purely as ME
diagnostics, and both have *negative* within-sequence correlation. They cannot predict
which frame inside a sequence costs more bits. Their pooled score is entirely the
between-sequence effect that fast-moving content costs more bits.

This is a sharper refutation of the pooled statistic than the `MVC` observation
recorded at Gate 1, because `mean_mv_mag` is not even a candidate metric. Any ranking
rule that places "average MV length" above "motion-compensated residual energy" is not
measuring metric quality.

**`TC_MC` is the best temporal metric, but by robustness rather than peak accuracy.**
It has the highest within-sequence PCC at every QP except 22, where `TC` edges it
(0.257 vs 0.270). Yet across the 16 (sequence x QP) cells it wins outright only twice:
`TC_SAD` takes 6, `TC2` 5, `TC` 3, `TC_MC` 2. It rarely wins; it never collapses.

**`TC_SAD` and `MVC` invert on sequences where the motion search fails.** Both go
negative on YachtRide (63 % saturation) and FoodMarket (44 %). When the search cannot
reach the true motion, SAD measures search failure rather than content complexity and
anti-correlates with bits. `TC` and `TC2`, which never touch motion estimation, stay
positive on YachtRide.

**The intra gate is the mechanism behind `TC_MC`'s robustness.** Its edge over
`TC_SAD` tracks how often the gate fires:

| sequence | `intra_frac` | `TC_SAD` | `TC_MC` | edge |
|---|---|---|---|---|
| HoneyBee | 7.0 % | 0.563 | 0.537 | -0.026 |
| ReadySteadyGo | 20.7 % | 0.620 | 0.595 | -0.025 |
| YachtRide | 36.8 % | -0.198 | 0.483 | **+0.681** |
| FoodMarket | 61.5 % | -0.415 | -0.245 | +0.171 |

Where the gate rarely fires the two metrics are equivalent, with `TC_SAD` marginally
ahead since `TC_MC` pays for a DCT that buys nothing. Where MC fails often,
`min(SC_MC, SC)` falls back to intra energy — a valid complexity measure — and rescues
the metric. This is the strongest evidence in the data for keeping `--gate intra` as
the default, and it should be weighed at Gate 4 against the pooled-CI ranking recorded
at Gate 2, which favoured `gate none`. FoodMarket qualifies the claim: gating lifts it
but both metrics stay negative, so the gate rescues *partial* MC failure, not content
that is mismatched outright.

**For spatial metrics the bias runs the other way.** Every spatial gap is negative:
pooling *understates* them. `SC_u` reads -0.187 pooled and +0.717 within-sequence —
chroma complexity genuinely predicts intra bits frame to frame, but does not rank
content across sequences, and pooling lets the between-sequence term flip the sign.
The same holds for `Colorfulness` (-0.411 vs +0.337). Pooling therefore has no
consistent bias direction: it inflates temporal metrics and deflates spatial ones,
depending on how between-sequence variance happens to align. `SC` at 0.896
within-sequence is the strongest metric in the entire set.

**`PCC_log` exceeds `PCC` for every temporal metric** — `TC_MC` 0.718 vs 0.564, `TC`
0.677 vs 0.427 — while the two are nearly equal for spatial metrics (`SC`: 0.756 vs
0.761). The metric-to-bits relationship is log-shaped, as rate-distortion theory
predicts (rate grows with the log of residual variance). The headline PCC therefore
understates every temporal metric by roughly 0.15 to 0.25.

**The CI columns quantify the n = 4 problem.** Frame-level CIs average 0.07 to 0.13
wide; block-bootstrap CIs average 1.05 to 1.79 — roughly 13x wider, several spanning
almost the whole [-1, 1] range. The narrow interval is the one that assumes 476
independent frames, which they are not.

#### The decisive test — which analysis survives a corpus change

Within-sequence PCC, Gate 1 corpus (with Bosphorus) versus this run (with FoodMarket):

| metric | QP 22 | QP 27 | QP 32 | QP 37 | mean abs delta |
|---|---|---|---|---|---|
| `TC` | +0.179 / +0.270 | +0.301 / +0.378 | +0.271 / +0.317 | +0.283 / +0.306 | 0.059 |
| `TC2` | +0.234 / +0.001 | +0.543 / +0.399 | +0.487 / +0.384 | +0.412 / +0.345 | 0.137 |
| `TC_SAD` | +0.034 / -0.063 | +0.214 / +0.103 | +0.269 / +0.180 | +0.407 / +0.350 | 0.089 |
| `MVC` | -0.027 / -0.096 | +0.145 / +0.042 | +0.174 / +0.069 | +0.170 / +0.093 | 0.088 |
| `TC_MC` | +0.287 / +0.257 | +0.385 / +0.332 | +0.405 / +0.342 | +0.516 / +0.440 | **0.056** |

Shifts of 0.06 to 0.14, all in the same direction (FoodMarket is harder than
Bosphorus), **ordering preserved at every QP**, and the same rising-with-QP trend in
both. Against this, pooled `fast_MVC` moved from 0.885 to 0.426 at QP 32 and lost first
place outright, and the pooled QP trend reversed from rising to falling.

The within-sequence analysis reproduces across a corpus change. The pooled analysis
does not.

#### Consequences for Gates 3 and 4

Gate 1 already reached the right conclusion — `TC_MC` is the metric to optimise — but
via the within-sequence table, not the pooled statistic the gate rule nominally keys
on. Three changes are proposed before Gate 3:

1. **Change the gate ranking key to mean within-sequence PCC.** The current key (lower
   bound of the pooled frame-level PCC CI) ranks `mean_mv_mag` first among temporal
   metrics, which is disqualifying for a ranking rule. The pooled table stays in the
   ledger as a reported statistic, not as a decision input.
2. **Use `PCC_log` for temporal metrics**, since the relationship is demonstrably
   log-shaped and plain PCC understates every temporal metric.
3. **Report `intra_frac` beside every `TC_MC` figure.** It determines whether `TC_MC`
   is measuring motion compensation or the intra fallback; at FoodMarket's 61.5 % it is
   largely `SC` under another name.

Phase 3 remains motivated: corpus `MV_sat_frac` is 39.4 % against a < 5 % target, and
the two sequences where the search saturates are exactly the two where `TC_SAD` and
`MVC` invert sign. Note also that the default `diamond` pattern contains no
candidates with both components non-zero, so it cannot represent diagonal motion at
all, and `MV_sat_frac` — an L-infinity test — cannot detect that failure mode. Measured
on synthetic ground truth, a true (4, 4) translation is estimated at magnitude 4.300
against a true 5.657 while `MV_sat_frac` reports only 0.213, and a (2, 2) translation
is estimated at 2.000 against 2.828 with `MV_sat_frac` at 0.000. The 39.4 % figure is
therefore a floor on search failure, not an estimate of it.

## Gate 2 resolution — ranking rule change and the gate axis

Closes the gate-axis question that the Gate 2 verdict originally deferred to Gate 4,
and records the ranking-rule change that resolved it.

### Ranking rule change (implemented)

The three changes proposed under *Consequences for Gates 3 and 4* are now in the code.
Ranking behaviour changes from this point forward; no measurement is invalidated, since
every statistic involved was already computed and stored — only which one decides.

| # | Change | Where |
|---|---|---|
| 1 | Gate ranking key is the **mean within-sequence** correlation, not the pooled frame-level CI lower bound | `validation/run_ablation.py` (`RANK_KEYS`, default `within`); `report.mean_within_table` |
| 2 | Temporal metrics are judged on **`PCC_log`**, spatial metrics on `PCC` | `report.primary_stat`, applied by both drivers |
| 3 | **`intra_frac` travels with every `TC_MC` figure** | `report.companion_column`, `report.mc_health_table`; a column in the ablation matrix and the headline table |

`--rank-by pooled_ci_lo` restores the superseded key so an older ablation's ordering can
be reproduced for comparison. `run_benchmark.py` now writes `mean_within_correlations.csv`
and `mc_health.csv` beside the existing outputs, and its ledger section leads with the
within-sequence table; the pooled table is retained and labelled *reported, not decided on*.

**Why the key changed.** Ranking the eight temporal metrics of the macOS re-baseline
under both keys (values averaged over QPs 22/27/32/37):

| rank | old key — pooled CI lower bound | new key — within-sequence `PCC_log` |
|---|---|---|
| 1 | `mean_mv_mag` 0.705 | `TC_MC` **0.329** |
| 2 | `MV_sat_frac` 0.609 | `TC2` 0.320 |
| 3 | `TC_MC` 0.511 | `TC` 0.317 |
| 4 | `MVC` 0.488 | `TC_SAD` 0.133 |
| 5 | `intra_frac` 0.442 | `MVC` 0.007 |
| 6 | `TC2` 0.398 | `mean_mv_mag` -0.030 |
| 7 | `TC_SAD` 0.393 | `intra_frac` -0.086 |
| 8 | `TC` 0.359 | `MV_sat_frac` -0.134 |

The old key placed the two motion-search *diagnostics* first and plain `TC` last. The
new key places `TC_MC` first and sends all three diagnostics to the bottom with negative
scores, which is the correct outcome: a quantity that cannot predict which frame costs
more bits should not outrank one that can.

### Ablation `mac-gate2-newrule` — 2026-08-24 11:59

- Phase: Gate 2 resolution (gate axis under the within-sequence rule)
- Commit: `5e10e09daea069f7d3bac2a94eb80fd17e2353d4`, working tree carrying the
  ranking-rule change in `validation/report.py`, `validation/run_ablation.py` and
  `validation/run_benchmark.py`
- Subset: **fast**, profile `full`, ranking metric `full_TC_MC` (temporal, judged on `PCC_log`)
- Ranking key: `within_mean` (`--rank-by within`)
- Axes: `mc` ∈ {dense_smooth, dense}; `gate` ∈ {intra, none}
- Extra args: `(none)`
- Sequences: YachtRide, ReadySteadyGo, HoneyBee, FoodMarket
- Missing sequences (skipped): Bosphorus
- Bootstrap: 1000 resamples, seed 12345
- Results: `validation/results/mac-gate2-newrule_5e10e09d` (run with `--skip-ledger`;
  this section is written by hand)

Values averaged over QPs 22/27/32/37. `within_mean` is the ranking key; `within_min` is
the worst single sequence; `intra_frac` is the fraction of blocks where the intra gate
fired. Pooled columns are reported, not decided on.

| variant | within_mean | within_min | within_PCC_mean | intra_frac | PCC_mean | PCC_lo_mean | SRCC_mean | fps |
|---|---|---|---|---|---|---|---|---|
| mc=dense_smooth gate=intra | 0.3292 | -0.3559 | 0.3426 | 0.3148 | 0.5644 | 0.5108 | 0.5552 | 30.1697 |
| mc=dense gate=intra | 0.3207 | -0.2980 | 0.3333 | 0.3217 | 0.4877 | 0.4312 | 0.5242 | 40.3023 |
| mc=dense_smooth gate=none | 0.2639 | -0.4499 | 0.2762 | 0.3148 | 0.5318 | 0.4757 | 0.5514 | 39.0244 |
| mc=dense gate=none | 0.2480 | -0.3925 | 0.2593 | 0.3217 | 0.4576 | 0.3999 | 0.5284 | 33.2410 |

#### Verdict — the gate axis is decided

**`--gate intra` stays the default; the Gate 4 deferral is closed.** Both `gate intra`
variants beat both `gate none` variants, so the ranking is separable by the gate axis
alone. Marginal effects:

| axis | separation in `within_mean` |
|---|---|
| `gate intra` vs `gate none` | **+0.069** |
| `mc dense_smooth` vs `mc dense` | +0.012 |

The gate axis matters about six times as much as the compensation axis. That ordering is
the reverse of what the pooled key reported at Gate 2, where the `mc` axis dominated and
`gate none` came second overall.

The mechanism is the one identified in the macOS metric comparison: `TC_MC` is
intra-gated, so `min(SC_MC, SC)` converts a motion-compensation failure into a spatial
complexity reading rather than into noise. On this corpus the gate fires on roughly 31 %
of blocks overall and on 61.5 % of FoodMarket's, which is where the separation is earned.
`gate none` also has the worse `within_min` at both `mc` settings (-0.45 / -0.39 against
-0.36 / -0.30), so removing the gate hurts most on the sequences that are already worst.

The first Gate 2 finding is unchanged: `dense_smooth` beats `dense` at both gate settings
under the revised rule (+0.0086 with the gate on, +0.0159 with it off), so the
Iteration-4 smoothing choice survives. `intra_frac` is essentially identical across the
`mc` axis (0.3148 vs 0.3217), confirming that smoothing changes the quality of the
compensated prediction rather than how often the gate rescues it.

**The `fps` column is not usable on this machine.** `dense_smooth gate=intra` reads
30.2 fps against 39.0 for `dense_smooth gate=none`, which is the same compensation work;
the spread is MPS scheduling noise on a ~13-second measurement. Any throughput decision
on the `mc` axis needs the CUDA machine.

#### Caveat — `TC_MC`'s margin over the no-ME baseline is thin

Change 2 narrows the case for `TC_MC` and this should be carried into Phase 3. Under
within-sequence `PCC` the metric led plain `TC` by 0.025 (0.343 vs 0.318); under the
correctly specified `PCC_log` the lead is 0.012 (0.329 vs 0.317), and `TC_MC` loses at
two of four QPs:

| QP | `TC` | `TC_MC` |
|---|---|---|
| 22 | **0.274** | 0.269 |
| 27 | **0.373** | 0.316 |
| 32 | 0.317 | **0.323** |
| 37 | 0.306 | **0.410** |

`TC_MC`, `TC2` and `TC` finish within 0.012 of one another. `TC_MC` remains the right
optimisation target and its advantage is real at high QP, but it is thin against a
baseline metric that requires no motion estimation at all, and the pooled table made that
gap look far larger than it is. Phase 3 should size the expected gain from a better
search against this margin rather than against the pooled figures recorded at Gate 1.
