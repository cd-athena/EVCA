# EVCA

EVCA is an open-source video complexity analyzer with the following functionalities:

- EVCA is an advanced tool that integrates the functionalities of both VCA and SITI approaches,
- EVCA is developed in Python, ensuring compatibility with GPU processing,
- EVCA enhances the definition of temporal complexity that was originally used in VCA.

This fork extends upstream EVCA with block-based motion estimation and motion
compensation (`-me`), chroma complexity, a colorfulness metric, and a correlation
harness that scores the features against x265 ground truth.

## Installation

With conda (recommended — `environment_gpu.yml` pins a CUDA build of PyTorch):

```bash
conda env create -f environment_gpu.yml
```

Use `environment.yml` for a CPU-only install, or `pip3 install -r requirements.txt`.

## Usage

```bash
python main.py -i input.yuv -r 1920x1080 -c ./csv/out.csv
```

Input is raw (uncompressed) YUV. `-r`, `-p` and `--bit_depth` must match the file:
nothing in the container tells EVCA how to interpret the bytes.

```bash
python main.py -i input.yuv -r 1920x1080 -me --profile full -cc -cf -c ./csv/out.csv
```

## Command Line Options

`python main.py --help` prints this list, generated from the parser itself, with
defaults and grouped by topic. The tables below mirror it.

### General

| Flag | Default | Description |
|---|---|---|
| `-h`, `--help` | | Show the generated help and exit. |
| `--version` | | Print the version and exit. |
| `-i`, `--input` | `test.yuv` | Raw YUV input file. |
| `-d`, `--dir` | | Directory of `.yuv` files to process in turn. Takes precedence over `-i`. |
| `-m`, `--method` | `EVCA` | `EVCA`, `VCA`, or `SITI`. |
| `-r`, `--resolution` | `1920x1080` | Source resolution as `[w]x[h]`. |
| `-p`, `--pix_fmt` | `yuv420` | `yuv420` or `yuv444`. |
| `--bit_depth` | `8` | `8`, `10`, `12`, or `16`. Metrics are rescaled to an 8-bit equivalent. |
| `-f`, `--frames` | `0` | Maximum frames to analyze; `0` means all. |
| `-s`, `--sample_rate` | `1` | Analyze every Nth frame. |
| `-b`, `--block_size` | `32` | Block size; must be a multiple of 4. |
| `-g`, `--gopsize` | `32` | Frames processed per batch. Larger uses more VRAM. |
| `-c`, `--csv` | `./csv/test.csv` | Output CSV path. |
| `-t`, `--transform` | `DCT` | `DCT`, `DWT`, or `DCT_B` (`DCT_B` requires `--block_size 32`). |
| `-fi`, `--filter` | `sobel` | Edge filter for `--method SITI`. Only `sobel` is implemented. |

### Feature toggles

| Flag | Default | Description |
|---|---|---|
| `-cc`, `--chroma_complexity` | off | Per-frame U and V complexity (`SC_u`, `SC_v`). |
| `-cf`, `--colorfulness` | off | Hasler & Süsstrunk M³ colorfulness, computed natively on the chroma planes. |
| `-me`, `--motion_estimation` | off | Block-based motion estimation; adds `MVC`, `TC_SAD`, `mean_mv_mag`. |
| `--profile` | `fast` | `fast` keeps ME to the search itself. `full` additionally transforms the motion-compensated residual, adding `TC_MC` and `intra_frac` at real cost (~366 vs ~585 fps at 1080p). |

### Motion estimation

The search scores candidates per block by SAD against whole-frame shifts of the
reference: the collocated block plus four neighbours at `--me-offset` pixels, or the
whole square with `--heuristic grid`. `--me-offset` is always denominated in
full-resolution pixels, whatever grid the search runs on.

| Flag | Default | Description |
|---|---|---|
| `--heuristic` | `diamond` | `diamond` puts four neighbours on the axes, `square` on the diagonals, `grid` fills the whole `(2r+1)²` square so reach and granularity are independent. |
| `--me-offset` | `2` | Reach of the pattern in full-resolution pixels (≥ 1). Motion beyond it cannot be tracked. |
| `--temporal-pool` | `1` | Box-filter the temporal path down by N before searching and before building the motion-compensated residual. |
| `--me-pool` | *(= `--temporal-pool`)* | Pooling factor for the search alone; set it to decouple the search from the residual path. |

### Temporal pooling

`--temporal-pool N` runs the whole temporal path — search, motion compensation,
residual, transform and intra gate — on a luma plane box-filtered down by N. The
spatial metrics (`B`, `SC`, `TC`, `TC2`) are untouched and stay full-resolution.

Each step of N quarters the cost of every search candidate and of the compensation
warp, and quantises motion vectors onto an N-pixel grid. `--me-offset` stays in source
pixels, so a wider reach costs nothing extra: at `--temporal-pool 4` a `--me-offset 8`
`grid` pattern covers ±8 px with the same 25 candidates that would cover ±2 px at full
resolution.

`--temporal-pool 1` (the default) reproduces full-resolution behaviour byte for byte.

```bash
# ~1.7x faster --profile full at 1080p, and 4K no longer exhausts a 16 GB card
python main.py -i input.yuv -r 1920x1080 -me --profile full --temporal-pool 2 -c ./csv/out.csv
```

Pooling changes what the temporal metrics measure, so values are comparable within a
pooling factor and not across factors:

- `TC_MC` is the weighted DCT energy of a residual computed on `block_size / N` blocks,
  so its scale drops with N. EVCA's weighting is written in normalised frequency,
  `exp(((i·j)/(N·N))² − 1)`, so the smaller matrix is the same weighting for the smaller
  block rather than an arbitrary rescale.
- `TC_SAD` is scored in the search domain, where box filtering has already averaged
  detail away, so it reads lower at higher `--me-pool`.
- `MVC` and `mean_mv_mag` stay in full-resolution pixels, but the vectors they summarise
  are quantised to the pooling grid.
- `intra_frac` falls, because a search with more reach loses to intra less often.

`--transform DCT_B` and `--transform DWT` are defined for the full-size block only and
are rejected with `--temporal-pool > 1`. The residual block must stay at least 8×8, so
`--temporal-pool 4` needs `--block_size 32`.

`validation/bench_pooling.py` measures the correlation against x265 P-frame bits for
each setting, so the trade-off can be re-derived on your own corpus.

### Motion compensation

Applies only with `-me --profile full`.

| Flag | Default | Description |
|---|---|---|
| `--mc` | `dense_smooth` | `dense_smooth` (bilinear upsample of a filtered MV field), `dense` (unfiltered), `block` (piecewise-constant), `obmc` (overlapped, raised-cosine blend). |
| `--mc-smooth` | `gauss` | MV-field filter for the dense modes: `gauss`, `median` (vector median), or `none`. |
| `--residual-dc` | off | Keep the DC coefficient in the residual energy. For a compensated residual, DC is the block's mean prediction error and carries real rate cost, unlike an intra block's DC. |

The motion-compensated residual energy is always capped at the block's own intra
energy (`min(SC_MC, SC)`), modelling an encoder's per-block inter/intra decision.
`intra_frac` reports how often that cap binds: the higher it is, the more `TC_MC`
reflects spatial complexity rather than motion compensation.

### Performance

| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | `auto` (CUDA → MPS → CPU), `cuda`, `mps`, or `cpu`. |
| `--loader` | `standard` | `standard` reads sequentially at low memory; `optimized` memory-maps for higher throughput but can exhaust RAM on very large files. |
| `--prefetch` | `1` | Overlap GOP loading with compute on a background thread. `0` disables. |

### Output and plotting

| Flag | Default | Description |
|---|---|---|
| `-bi`, `--block_info` | `0` | Also write per-block features to `<csv>_<METRIC>_blocks.csv`. |
| `-pi`, `--plot_info` | `0` | Plot per-block features per frame to `./png/`. Implies `-bi 1`. Also accepts the legacy spelling `-plot_info`. |
| `-pm`, `--plot_metrics` | `0` | Plot per-frame metrics over time to `./png/frame_metrics/`. Also accepts the legacy spelling `-plot_metrics`. |
| `-dp`, `--dpi` | `100` | Plot resolution. |

Every run writes a `<csv>.meta.json` sidecar recording the git SHA and the full
argument namespace, so a CSV can always be traced back to the code that produced it.

## Output columns

One row per analyzed frame. Frame 0 has no predecessor, so temporal metrics are 0 there.

| Column | Method / flag | Meaning |
|---|---|---|
| `B` | EVCA, VCA | Mean block brightness (DCT DC term). |
| `SC` / `E` | EVCA / VCA | Spatial complexity: weighted DCT energy per block, averaged. |
| `TC` / `h` | EVCA / VCA | Temporal complexity against the previous frame. |
| `TC2` / `h2` | EVCA / VCA | Temporal complexity against the frame before that. |
| `SC_u`, `SC_v` | `-cc` | Chroma spatial complexity. |
| `Colorfulness` | `-cf` | Hasler & Süsstrunk M³. |
| `MVC` | `-me` | Motion-vector field complexity (Laplacian of the MV field). |
| `TC_SAD` | `-me` | Mean minimum block SAD from the search. |
| `mean_mv_mag` | `-me` | Mean motion-vector magnitude in pixels. |
| `TC_MC` | `-me --profile full` | Weighted DCT energy of the motion-compensated residual, intra-gated. |
| `intra_frac` | `-me --profile full` | Fraction of blocks where the intra gate bound. |
| `SI`, `TI`, `TI2` | `-m SITI` | Spatial and temporal information (ITU-T P.910). |

## Validation

`validation/correlate.py` scores the features against x265 ground truth using the
paper's definition: spatial complexity is the bit count to code a frame as an I-frame
at a fixed QP, temporal complexity the bits to code the next frame as a P-frame
against it.

```bash
python validation/correlate.py --pairs 5 --pair-stride 50 --device cuda
```

For each `(sequence, pair)` it copies frames *k* and *k+1* out of the raw YUV into a
two-frame file, encodes exactly that with x265 (frame 0 forced to I, frame 1 to P),
reads the two frame sizes back, and runs EVCA over the same file — so both sides
consume one identical input and frame alignment is structural. It reports plain Pearson
and Spearman correlations and writes `pairs.csv` and `correlations.csv` under
`validation/results/<label>/`.

Requires `ffmpeg`/`ffprobe` built with `libx265`. Sequences are configured in
`validation/sequences.json`; override the root with `--sequence-root` or the
`EVCA_SEQUENCE_ROOT` environment variable.

`validation/bench_pooling.py` compares several EVCA configurations against one identical
set of encodes, caching them by (sequence, frame, QP):

```bash
python validation/bench_pooling.py --pairs 16 --qps 22,27,32 --device cuda
```

It reports mean *within-sequence* PCC as the headline with the per-sequence vector
beside it. On this corpus the pooled and within-sequence statistics rank the variants
differently, and the pooled one is dominated by between-sequence offsets.

Interpretation note: with a small corpus the pooled correlation is dominated by
between-sequence offsets rather than by per-frame prediction quality, and a single
atypical sequence can move it a long way. Check the per-pair values in `pairs.csv`
before reading much into a pooled figure.

## Tests

```bash
python -m pytest tests/ -q
```

CPU-only and self-contained: the suite generates its own synthetic YUV sequences with
known ground-truth motion (`validation/synthetic.py`) and needs no test assets.

## Citation

If this work is helpful for your research, please consider citing EVCA.

```
@inproceedings{amirpour_evca_2024,
author = {Amirpour, Hadi and Ghasempour, Mohammad and Qu, Lingfeng and Hamidouche, Wassim and Timmerer, Christian},
title = {{EVCA: Enhanced Video Complexity Analyzer}},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 15th ACM Multimedia Systems Conference},
series = {MMSys '24} }
```
