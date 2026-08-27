"""Sweep temporal-pooling variants against x265 ground truth.

    python validation/bench_pooling.py --pairs 16 --qps 22,27,32

Ground truth is `correlate.py`'s, unchanged: a two-frame clip, frame 0 forced to I and
frame 1 to P at a fixed QP, with the P-frame packet size read back through ffprobe.
This driver differs from `correlate.py` in three ways that the pooling question needs:

  1. It runs several EVCA configurations over one identical set of encodes, so
     variants are compared on the same ground truth rather than on separate runs.
  2. It reports **mean within-sequence PCC** as the headline, with the per-sequence
     vector always printed beside it. The pooled correlation is reported too, but it is
     dominated by between-sequence offsets -- on this corpus the two statistics rank the
     variants differently, and the within-sequence one is the honest answer to "does
     this track frame-to-frame complexity".
  3. Encodes are cached on disk by (sequence, frame, QP), because re-encoding dominates
     runtime and the ground truth does not depend on the EVCA side at all.

EVCA runs in-process rather than through a subprocess: 6 sequences x 16 pairs x 6
variants is ~600 invocations, and paying torch's start-up cost each time would dominate.
Variants are still spelled as real command lines and parsed by the real parser, so what
is measured is what a user would type.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.correlate import (DEFAULT_CONFIG, encode_pair, extract_pair,
                                  ffmpeg_available, load_config, probe_pair,
                                  to_markdown)

# Each variant is the extra command line a user would add to a `-me --profile full` run.
VARIANTS = {
    'Q0_current':      [],
    'Q1_search_pool2': ['--me-pool', '2'],
    'Q3_pool2':        ['--temporal-pool', '2'],
    'Q4_pool2_reach4': ['--temporal-pool', '2', '--heuristic', 'dense', '--me-offset', '4'],
    'Q5_resid2_me4':   ['--temporal-pool', '2', '--me-pool', '4',
                        '--heuristic', 'dense', '--me-offset', '8'],
    'Q6_pool4_reach8': ['--temporal-pool', '4', '--heuristic', 'dense', '--me-offset', '8'],
}

METRIC = 'TC_MC'


def ground_truth(seq, start, qp, cache_dir, work_dir, preset):
    """P-frame bit count for (sequence, frame, QP), cached across runs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_dir / f"{seq['name']}_f{start}_qp{qp}.json"
    if key.exists():
        return json.loads(key.read_text())['tc_gt']
    yuv = work_dir / f"{seq['name']}_{start}.yuv"
    mp4 = work_dir / f"{seq['name']}_{start}_qp{qp}.mp4"
    extract_pair(seq, start, yuv)
    encode_pair(seq, yuv, qp, mp4, preset)
    sc_gt, tc_gt = probe_pair(mp4)
    key.write_text(json.dumps({'sc_gt': sc_gt, 'tc_gt': tc_gt}))
    mp4.unlink(missing_ok=True)
    return tc_gt


def run_variant(seq, yuv_path, csv_path, extra, device):
    """One EVCA run over a two-frame clip; returns the P-frame row's metrics."""
    import torch
    from main import get_parser_arguments
    from libs.EVCA import EVCA
    argv = ['-i', str(yuv_path), '-r', seq['res'], '-p', seq['pix_fmt'],
            '--bit_depth', str(seq.get('bit_depth', 8)), '-f', '2',
            '--device', device, '-c', str(csv_path),
            '-me', '--profile', 'full'] + extra
    args = get_parser_arguments(argv)
    dev = torch.device(device if device != 'auto'
                       else ('cuda' if torch.cuda.is_available() else 'cpu'))
    EVCA(args, [args.input], dev)
    df = pd.read_csv(csv_path)
    if len(df) != 2:
        raise RuntimeError(f'{yuv_path.name}: EVCA returned {len(df)} rows, expected 2')
    return {c: float(df.loc[1, c]) for c in ('TC_MC', 'TC_SAD', 'MVC',
                                             'mean_mv_mag', 'intra_frac')
            if c in df.columns}


def collect(cfg, args, work_dir, cache_dir):
    rows = []
    starts = [i * args.pair_stride for i in range(args.pairs)]
    for seq in cfg['sequences']:
        for start in starts:
            yuv = work_dir / f"{seq['name']}_{start}.yuv"
            try:
                extract_pair(seq, start, yuv)
            except ValueError as e:
                print(f'  skip {seq["name"]} f{start}: {e}', file=sys.stderr)
                continue
            gts = {qp: ground_truth(seq, start, qp, cache_dir, work_dir, args.x265_preset)
                   for qp in args.qp_list}
            for name, extra in args.variant_items:
                m = run_variant(seq, yuv, work_dir / f'{name}.csv', extra, args.device)
                rows.append({'seq': seq['name'], 'start': start, 'variant': name,
                             **{f'TC_gt_qp{qp}': v for qp, v in gts.items()}, **m})
            yuv.unlink(missing_ok=True)
        print(f"  {seq['name']}: done", flush=True)
    return pd.DataFrame(rows)


def summarise(df, qps):
    """Mean within-sequence PCC (headline) and the pooled figure, per variant per QP."""
    out = []
    for qp in qps:
        gt = f'TC_gt_qp{qp}'
        for name in df.variant.unique():
            sub = df[df.variant == name]
            pooled = pearsonr(sub[METRIC], sub[gt])[0] if len(sub) > 2 else np.nan
            per_seq = {}
            for s in sorted(sub.seq.unique()):
                g = sub[sub.seq == s]
                # A sequence with a constant metric has no correlation to report.
                if len(g) > 2 and g[METRIC].nunique() > 1 and g[gt].nunique() > 1:
                    per_seq[s] = pearsonr(g[METRIC], g[gt])[0]
            out.append({'QP': qp, 'variant': name, 'n': len(sub),
                        'within_PCC': np.mean(list(per_seq.values())) if per_seq else np.nan,
                        'pooled_PCC': pooled,
                        'pooled_SRCC': spearmanr(sub[METRIC], sub[gt])[0] if len(sub) > 2 else np.nan,
                        'intra_frac': sub['intra_frac'].mean() if 'intra_frac' in sub else np.nan,
                        **{f'PCC_{s[:9]}': v for s, v in per_seq.items()}})
    return pd.DataFrame(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=str(DEFAULT_CONFIG))
    p.add_argument('--sequence-root', default=None)
    p.add_argument('--qps', default='22,27,32',
                   help='comma-separated QPs for the ground truth')
    p.add_argument('--pairs', type=int, default=16)
    p.add_argument('--pair-stride', type=int, default=25)
    p.add_argument('--variants', default=','.join(VARIANTS),
                   help='comma-separated subset of the built-in variants')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--x265-preset', default='medium')
    p.add_argument('--cache-dir', default=str(SCRIPT_DIR / 'gt_cache' / 'pairs'))
    p.add_argument('--out-dir', default=str(SCRIPT_DIR / 'results' / 'pooling'))
    args = p.parse_args()
    args.qp_list = [int(q) for q in args.qps.split(',')]
    unknown = [v for v in args.variants.split(',') if v not in VARIANTS]
    if unknown:
        print(f'ERROR: unknown variants: {unknown}. Choose from {list(VARIANTS)}',
              file=sys.stderr)
        return 1
    args.variant_items = [(v, VARIANTS[v]) for v in args.variants.split(',')]

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

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(cfg['sequences'])} sequences x {args.pairs} pairs x "
          f"{len(args.variant_items)} variants, QPs {args.qp_list}", flush=True)

    with tempfile.TemporaryDirectory(prefix='evca_pool_') as tmp:
        pairs = collect(cfg, args, Path(tmp), Path(args.cache_dir))
    pairs.to_csv(out_dir / 'pairs.csv', index=False)
    summary = summarise(pairs, args.qp_list)
    summary.to_csv(out_dir / 'summary.csv', index=False)

    print(f'\nPer-pair measurements: {out_dir / "pairs.csv"}')
    print(f'\n=== {METRIC} vs x265 P-frame bits '
          f'(within_PCC is the decision statistic) ===\n')
    print(to_markdown(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
