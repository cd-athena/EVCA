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
