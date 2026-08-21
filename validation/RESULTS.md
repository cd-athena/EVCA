# EVCA2 Results Ledger

Running record of every benchmark run, default-behavior change, and gate decision.
Convention: one section per phase; benchmark rows record phase, commit SHA, CLI flags,
device, subset, fps, and correlation tables (or a pointer to the results directory).

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
no defaults); the disagreement is carried into Gate 4, where the gate axis is decided.
