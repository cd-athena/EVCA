"""EVCA correlation against x265 ground truth.

    python validation/correlate.py --pairs 5 --pair-stride 50

Ground truth follows the EVCA paper's definition: spatial complexity is the bit count
needed to code a frame as an I-frame at a fixed QP, temporal complexity the bit count
needed to code the next frame as a P-frame against it. For each (sequence, pair) the
harness therefore:

  1. copies frames `k` and `k+1` out of the raw YUV into a two-frame file,
  2. encodes exactly that file with x265 at a fixed QP, forcing frame 0 to I and
     frame 1 to P (`bframes=0`, `no-scenecut`),
  3. reads the two frame sizes back: `SC_gt` = I bits, `TC_gt` = P bits,
  4. runs EVCA over the *same* two-frame file.

Both sides consume one identical input file, so frame alignment is structural rather
than an assumption about two independent readers agreeing on frame indices.

Correlations are plain Pearson (PCC) and Spearman (SRCC) over the (sequence, pair)
rows -- no transform, no resampling.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / 'sequences.json'
DEFAULT_OUT_DIR = SCRIPT_DIR / 'results'

# Profile -> extra flags for main.py. `baseline` is upstream EVCA with no motion
# estimation; `full` adds the motion-compensated residual transform (TC_MC).
PROFILE_FLAGS = {
    'baseline': [],
    'fast': ['-me', '-cc', '-cf', '--profile', 'fast'],
    'full': ['-me', '-cc', '-cf', '--profile', 'full'],
}

# EVCA columns compared against each ground truth. Spatial metrics are read from row 0
# (the I-frame), temporal metrics from row 1 (the P-frame).
#
# `TC2` is deliberately absent: it compares frame f against f-2, which does not exist
# in a two-frame clip, so EVCA reports a structural zero for it.
SPATIAL_METRICS = ['SC', 'B', 'SC_u', 'SC_v', 'Colorfulness']
TEMPORAL_METRICS = ['TC', 'TC_SAD', 'TC_MC', 'MVC', 'mean_mv_mag', 'intra_frac']


def run(cmd: list) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'command failed: {" ".join(shlex.quote(c) for c in cmd)}\n'
                           f'{proc.stderr}')
    return proc


def ffmpeg_available() -> bool:
    """True when ffmpeg/ffprobe exist and ffmpeg carries libx265."""
    if not (shutil.which('ffmpeg') and shutil.which('ffprobe')):
        return False
    try:
        out = run(['ffmpeg', '-hide_banner', '-encoders']).stdout
    except Exception:
        return False
    return 'libx265' in out


def load_config(config_path: Path, sequence_root: str = None) -> dict:
    """Loads sequences.json and resolves paths.

    Root precedence: explicit argument > EVCA_SEQUENCE_ROOT > the value in the file.
    Sequences whose file is absent are dropped and reported under `missing`.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    root = Path(sequence_root or os.environ.get('EVCA_SEQUENCE_ROOT')
                or cfg.get('sequence_root', '.'))
    resolved, missing = [], []
    for seq in cfg['sequences']:
        p = Path(seq['path'])
        full = p if p.is_absolute() else root / p
        seq = dict(seq, full_path=str(full))
        (resolved if full.exists() else missing).append(seq)
    cfg['sequences'] = resolved
    cfg['missing'] = missing
    cfg['sequence_root'] = str(root)
    return cfg


def frame_bytes(seq: dict) -> int:
    width, height = (int(v) for v in seq['res'].split('x'))
    samples = 1.5 if seq['pix_fmt'] == 'yuv420' else 3.0
    return int(width * height * samples * (1 if seq.get('bit_depth', 8) == 8 else 2))


def extract_pair(seq: dict, start: int, out_path: Path) -> None:
    """Copies frames `start` and `start+1` of a raw YUV into a two-frame file."""
    size = frame_bytes(seq)
    with open(seq['full_path'], 'rb') as src:
        src.seek(start * size)
        data = src.read(2 * size)
    if len(data) < 2 * size:
        raise ValueError(f"{seq['name']}: frames {start},{start + 1} run past end of file")
    out_path.write_bytes(data)


def encode_pair(seq: dict, yuv_path: Path, qp: int, out_path: Path,
                x265_preset: str = 'medium') -> None:
    """Encodes a two-frame YUV as I then P at a fixed QP.

    `bframes=0` keeps frame 1 a P frame, `no-scenecut` stops the encoder promoting it
    to an I frame on a busy transition, and `keyint` large forbids a second IDR. The
    mp4 container keeps VPS/SPS/PPS in extradata, so they do not inflate the I-frame
    packet the way a raw Annex-B stream would.
    """
    pix_fmt = 'yuv420p' if seq['pix_fmt'] == 'yuv420' else 'yuv444p'
    if seq.get('bit_depth', 8) > 8:
        pix_fmt += f"{seq['bit_depth']}le"
    params = (f'keyint=9999:no-open-gop=1:bframes=0:no-scenecut=1:qp={qp}:log-level=none')
    run(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', pix_fmt,
         '-video_size', seq['res'], '-framerate', str(seq['fps']), '-i', str(yuv_path),
         '-c:v', 'libx265', '-preset', x265_preset, '-x265-params', params,
         str(out_path)])


def probe_pair(mp4_path: Path) -> tuple:
    """Returns (I-frame bits, P-frame bits) for a two-frame bitstream.

    Raises if the stream is not exactly one I frame followed by one P frame, so a
    silently mis-encoded pair can never reach the correlation as a plausible number.
    """
    out = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'frame=pict_type,pkt_size', '-of', 'csv=p=0', str(mp4_path)]).stdout
    frames = []
    for line in out.strip().split('\n'):
        # Field order varies across ffprobe builds; identify tokens by content.
        parts = [p.strip() for p in line.split(',') if p.strip()]
        size = next((int(p) for p in parts if p.isdigit()), None)
        ptype = next((p.upper() for p in parts if p.upper() in ('I', 'P', 'B')), None)
        if size is not None:
            frames.append((ptype, size * 8))
    types = [t for t, _ in frames]
    if types != ['I', 'P']:
        raise RuntimeError(f'{mp4_path.name}: expected frame types [I, P], got {types}')
    return frames[0][1], frames[1][1]


def run_evca(seq: dict, yuv_path: Path, out_csv: Path, profile: str,
             device: str, extra_args: list) -> dict:
    """Runs EVCA over a two-frame clip; returns {metric: value} for both domains."""
    cmd = [sys.executable, str(PROJECT_ROOT / 'main.py'),
           '-i', str(yuv_path), '-r', seq['res'], '-p', seq['pix_fmt'],
           '--bit_depth', str(seq.get('bit_depth', 8)), '-f', '2',
           '--device', device, '-c', str(out_csv)]
    cmd += PROFILE_FLAGS[profile] + extra_args
    run(cmd)

    df = pd.read_csv(out_csv)
    if len(df) != 2:
        raise RuntimeError(f'{yuv_path.name}: EVCA returned {len(df)} rows, expected 2')
    values = {}
    for col in SPATIAL_METRICS:
        if col in df.columns:
            values[col] = float(df.loc[0, col])       # I-frame row
    for col in TEMPORAL_METRICS:
        if col in df.columns:
            values[col] = float(df.loc[1, col])       # P-frame row
    return values


def collect(cfg: dict, args, work_dir: Path) -> pd.DataFrame:
    """One row per (sequence, pair) carrying every EVCA metric and both ground truths."""
    rows = []
    starts = [i * args.pair_stride for i in range(args.pairs)]
    for seq in cfg['sequences']:
        for start in starts:
            tag = f"{seq['name']}_f{start}"
            print(f'  [{tag}]', flush=True)
            pair_yuv = work_dir / f'{tag}.yuv'
            extract_pair(seq, start, pair_yuv)

            mp4 = work_dir / f'{tag}_qp{args.qp}.mp4'
            encode_pair(seq, pair_yuv, args.qp, mp4, args.x265_preset)
            sc_gt, tc_gt = probe_pair(mp4)

            metrics = run_evca(seq, pair_yuv, work_dir / f'{tag}.csv',
                               args.profile, args.device, args.extra_args_list)
            rows.append({'seq_name': seq['name'], 'start_frame': start,
                         'SC_gt': sc_gt, 'TC_gt': tc_gt, **metrics})
            pair_yuv.unlink()          # ~6 MB each at 1080p; the mp4 and csv stay
    return pd.DataFrame(rows)


def correlate(df: pd.DataFrame) -> pd.DataFrame:
    """Plain Pearson and Spearman correlations of each metric against its ground truth."""
    records = []
    for domain, metrics, gt_col in (('Spatial', SPATIAL_METRICS, 'SC_gt'),
                                    ('Temporal', TEMPORAL_METRICS, 'TC_gt')):
        for metric in metrics:
            if metric not in df.columns:
                continue
            sub = df[[metric, gt_col]].dropna()
            # A metric that is constant across the corpus has no correlation to report.
            if len(sub) < 3 or sub[metric].nunique() < 2:
                continue
            x, y = sub[metric].to_numpy(float), sub[gt_col].to_numpy(float)
            records.append({'Domain': domain, 'metric': metric, 'gt': gt_col,
                            'n': len(sub), 'PCC': pearsonr(x, y)[0],
                            'SRCC': spearmanr(x, y)[0]})
    return pd.DataFrame(records)


def to_markdown(df: pd.DataFrame, floatfmt: str = '{:.4f}') -> str:
    if df.empty:
        return '_(empty)_'
    disp = df.copy()
    for col in disp.columns:
        if pd.api.types.is_float_dtype(disp[col]):
            disp[col] = disp[col].map(lambda v: '' if pd.isna(v) else floatfmt.format(v))
    header = '| ' + ' | '.join(str(c) for c in disp.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(disp.columns)) + '|'
    rows = ['| ' + ' | '.join(str(v) for v in row) + ' |'
            for row in disp.itertuples(index=False)]
    return '\n'.join([header, sep] + rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--sequence-root', default=None)
    p.add_argument('--qp', type=int, default=22, help='fixed QP for the ground truth')
    p.add_argument('--pairs', type=int, default=4,
                   help='(I, P) frame pairs per sequence; 1 uses only frames 0 and 1')
    p.add_argument('--pair-stride', type=int, default=60,
                   help='frame distance between successive pairs')
    p.add_argument('--profile', default='full', choices=sorted(PROFILE_FLAGS))
    p.add_argument('--extra-args', default='', help='extra flags passed to main.py')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--x265-preset', default='medium')
    p.add_argument('--label', default='corr', help='names the output directory')
    p.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()
    args.extra_args_list = shlex.split(args.extra_args)

    if not ffmpeg_available():
        print('ERROR: ffmpeg with libx265 is required.', file=sys.stderr)
        return 1

    cfg = load_config(Path(args.config), args.sequence_root)
    if cfg['missing']:
        print('WARNING: missing sequences (skipped): '
              + ', '.join(s['name'] for s in cfg['missing']), file=sys.stderr)
    if not cfg['sequences']:
        print('ERROR: no sequences found.', file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    n_expected = len(cfg['sequences']) * args.pairs
    print(f"{len(cfg['sequences'])} sequences x {args.pairs} pairs = {n_expected} "
          f"observations, QP {args.qp}, profile {args.profile}"
          + (f", extra args: {args.extra_args}" if args.extra_args else ''), flush=True)

    with tempfile.TemporaryDirectory(prefix='evca_corr_') as tmp:
        pairs = collect(cfg, args, Path(tmp))
    pairs.to_csv(out_dir / 'pairs.csv', index=False)

    corr = correlate(pairs)
    corr.to_csv(out_dir / 'correlations.csv', index=False)

    print(f'\nPer-pair measurements written to {out_dir / "pairs.csv"}')
    print('\n=== Correlations (Pearson / Spearman, n = observations) ===\n')
    print(to_markdown(corr))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
