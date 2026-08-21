"""Cross-product ablation driver.

    python validation/run_ablation.py --label gate2 --subset fast \
        --axis mc=dense_smooth,dense --axis gate=intra,none

Builds the ground truth once, then runs one EVCA extraction per cell of the
flag cross-product and reports pooled frame-level correlation (with CI) plus the
mean within-sequence correlation for the chosen metric. Boolean flags take the
values `on`/`off`, e.g. `--axis residual-dc=off,on`.

Ranking follows the gate rule: highest lower bound of the 95 % frame-level
bootstrap CI of the pooled PCC, averaged over QPs; ties go to the cheaper variant.
"""
import argparse
import itertools
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from validation import ground_truth as gt  # noqa: E402
from validation.report import format_markdown, frame_level_report  # noqa: E402
from validation.run_benchmark import (CACHE_DIR, RESULTS_DIR, RESULTS_MD,  # noqa: E402
                                      SUBSET_FRAMES, collect_evca_frames, git_sha)

BOOL_VALUES = {'on': True, 'off': False}


def parse_axis(spec: str):
    """'mc=dense,block' -> ('mc', ['dense', 'block'])."""
    if '=' not in spec:
        raise argparse.ArgumentTypeError(f'--axis expects KEY=V1,V2 (got {spec!r})')
    key, values = spec.split('=', 1)
    return key.strip(), [v.strip() for v in values.split(',') if v.strip()]


def variant_flags(combo: dict) -> list:
    """Turns {'mc': 'block', 'residual-dc': 'on'} into CLI flags."""
    flags = []
    for key, value in combo.items():
        if value in BOOL_VALUES:
            if BOOL_VALUES[value]:
                flags.append(f'--{key}')
        else:
            flags += [f'--{key}', value]
    return flags


def variant_name(combo: dict) -> str:
    return ' '.join(f'{k}={v}' for k, v in combo.items())


def main() -> int:
    p = argparse.ArgumentParser(description='EVCA2 flag cross-product ablation.')
    p.add_argument('--label', required=True)
    p.add_argument('--phase', default='')
    p.add_argument('--subset', choices=['fast', 'full'], default='fast')
    p.add_argument('--axis', action='append', required=True, type=parse_axis,
                   help='KEY=V1,V2 (repeatable); values on/off mean a boolean flag')
    p.add_argument('--profile', default='full', choices=['baseline', 'fast', 'full'])
    p.add_argument('--metric', default='full_TC_MC', help='metric column to rank by')
    p.add_argument('--extra-args', default='', help='flags applied to every variant')
    p.add_argument('--device', default='auto')
    p.add_argument('--loader', default='optimized')
    p.add_argument('--config', default=str(gt.DEFAULT_CONFIG))
    p.add_argument('--sequence-root', default=None)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--skip-ledger', action='store_true')
    args = p.parse_args()

    cfg = gt.load_config(Path(args.config), args.sequence_root)
    if not cfg['sequences']:
        print('ERROR: no sequences available.', file=sys.stderr)
        return 1
    qps = cfg.get('qps', [22, 27, 32, 37])
    n_frames = SUBSET_FRAMES[args.subset]
    sha = git_sha()
    out_dir = RESULTS_DIR / f'{args.label}_{sha}'
    out_dir.mkdir(parents=True, exist_ok=True)
    base_extra = shlex.split(args.extra_args)

    if not gt.ffmpeg_available():
        print('ERROR: ffmpeg with libx265 is required for an ablation.', file=sys.stderr)
        return 1
    print('Building ground truth (cached across variants)...', flush=True)
    gt_frames = gt.build_ground_truth(cfg['sequences'], qps, n_frames, CACHE_DIR)
    gt_frames.to_csv(out_dir / 'ground_truth_frames.csv', index=False)

    keys = [k for k, _ in args.axis]
    combos = [dict(zip(keys, values))
              for values in itertools.product(*[v for _, v in args.axis])]
    print(f'{len(combos)} variants x {len(cfg["sequences"])} sequences\n', flush=True)

    rows, per_variant_frames = [], {}
    for combo in combos:
        name = variant_name(combo)
        flags = variant_flags(combo) + base_extra
        print(f'--- {name}', flush=True)
        vdir = out_dir / ('v_' + '_'.join(f'{k}-{v}' for k, v in combo.items()))
        evca, fps_df = collect_evca_frames(cfg['sequences'], [args.profile], n_frames,
                                           vdir, args.device, flags, args.loader)
        evca.to_csv(vdir / 'evca_frames.csv', index=False)
        per_variant_frames[name] = evca

        if args.metric not in evca.columns:
            print(f'    metric {args.metric} absent; columns: {list(evca.columns)}',
                  file=sys.stderr)
            continue
        corr = frame_level_report(evca, gt_frames, qps, n_boot=args.n_boot,
                                  seed=args.seed, metrics=[args.metric])
        corr.to_csv(vdir / 'frame_level_correlations.csv', index=False)

        pooled = corr[corr['scope'] == 'pooled']
        per_seq = corr[corr['scope'] != 'pooled']
        row = {'variant': name,
               'fps': fps_df['frames'].sum() / fps_df['seconds'].sum(),
               'PCC_mean': pooled['PCC'].mean(),
               'PCC_lo_mean': pooled['PCC_lo'].mean(),
               'PCC_hi_mean': pooled['PCC_hi'].mean(),
               'SRCC_mean': pooled['SRCC'].mean(),
               'perseq_PCC_mean': per_seq['PCC'].mean()}
        for qp in qps:
            sub = pooled[pooled['QP'] == qp]
            row[f'PCC_qp{qp}'] = sub['PCC'].iloc[0] if len(sub) else np.nan
        rows.append(row)
        print(f'    PCC {row["PCC_mean"]:.4f} (CI lo {row["PCC_lo_mean"]:.4f}), '
              f'per-seq {row["perseq_PCC_mean"]:.4f}, {row["fps"]:.1f} fps', flush=True)

    if not rows:
        print('ERROR: no variant produced the requested metric.', file=sys.stderr)
        return 1

    df = pd.DataFrame(rows).sort_values('PCC_lo_mean', ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / 'ablation_matrix.csv', index=False)

    with open(out_dir / 'run_meta.json', 'w') as f:
        json.dump({'label': args.label, 'phase': args.phase, 'subset': args.subset,
                   'git_sha': git_sha(short=False), 'metric': args.metric,
                   'profile': args.profile, 'axes': dict(args.axis),
                   'extra_args': base_extra, 'qps': qps,
                   'sequences': [s['name'] for s in cfg['sequences']],
                   'n_boot': args.n_boot, 'seed': args.seed,
                   'timestamp': datetime.now().isoformat(timespec='seconds')}, f, indent=2)

    print('\n=== Ablation matrix (ranked by CI lower bound) ===')
    print(df.to_string(index=False))

    if not args.skip_ledger:
        cols = ['variant', 'PCC_mean', 'PCC_lo_mean', 'PCC_hi_mean', 'SRCC_mean',
                'perseq_PCC_mean', 'fps']
        lines = ['', f'### Ablation `{args.label}` — {datetime.now():%Y-%m-%d %H:%M}', '',
                 f'- Phase: {args.phase}',
                 f'- Commit: `{git_sha(short=False)}`',
                 f'- Subset: **{args.subset}**, profile `{args.profile}`, '
                 f'ranking metric `{args.metric}`',
                 f'- Axes: ' + '; '.join(f'`{k}` ∈ {{{", ".join(v)}}}' for k, v in args.axis),
                 f'- Extra args: `{" ".join(base_extra) or "(none)"}`',
                 f'- Sequences: {", ".join(s["name"] for s in cfg["sequences"])}',
                 f'- Results: `{out_dir.relative_to(SCRIPT_DIR.parent)}`',
                 '',
                 'Values are averaged over QPs 22/27/32/37. `PCC_lo_mean` is the gate '
                 'ranking key; `perseq_PCC_mean` is the mean within-sequence PCC.', '',
                 format_markdown(df[cols]), '']
        with open(RESULTS_MD, 'a') as f:
            f.write('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
