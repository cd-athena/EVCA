"""EVCA2 benchmark driver.

    python validation/run_benchmark.py --subset fast --label gate1

Runs the configured EVCA profiles over every configured sequence, builds (or reuses)
per-frame x265 ground truth, computes frame-level and sequence-mean correlations, and
writes everything to `validation/results/<label>_<sha>/` plus a summary section in
`validation/RESULTS.md`.

The ledger section leads with the **mean within-sequence** correlation, which is the
decision statistic: pooling over a handful of sequences mixes between-sequence content
ranking into what is meant to be a per-frame prediction score. Temporal metrics are
judged on `PCC_log`, since P-frame bits grow with the log of residual variance. Every
`TC_MC` figure is accompanied by `intra_frac`, which says how often the intra gate
fired and therefore how much of `TC_MC` is spatial complexity rather than motion
compensation. See `validation/report.py` for both conventions.

`--extra-args` passes ablation flags straight through to `main.py`, so a Phase 2+
matrix is driven by repeated invocations with distinct labels.
"""
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validation import ground_truth as gt  # noqa: E402
from validation.report import (format_markdown, frame_level_report,  # noqa: E402
                               headline_table, mc_health_table, mean_within_table,
                               sequence_mean_report)

RESULTS_MD = SCRIPT_DIR / 'RESULTS.md'
RESULTS_DIR = SCRIPT_DIR / 'results'
CACHE_DIR = SCRIPT_DIR / 'gt_cache'

SUBSET_FRAMES = {'fast': 120, 'full': 0}

# Profile -> extra CLI flags. `baseline` is upstream EVCA with no motion estimation.
PROFILE_FLAGS = {
    'baseline': [],
    'fast': ['-me', '-cc', '-cf', '--profile', 'fast'],
    'full': ['-me', '-cc', '-cf', '--profile', 'full'],
}

# Headline metrics quoted in the ledger summary.
HEADLINE = ['baseline_TC', 'fast_TC_SAD', 'fast_MVC', 'full_TC_MC', 'baseline_SC']

_TIME_RE = re.compile(r'completed in ([0-9.]+) seconds')


def git_sha(short: bool = True) -> str:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
        return sha[:8] if short else sha
    except Exception:
        return 'nogit'


def run_evca(seq: dict, profile: str, n_frames: int, out_csv: Path, device: str,
             extra_args: list, loader: str) -> dict:
    """Runs one EVCA extraction; returns {frames, seconds, fps, cmd}."""
    cmd = [sys.executable, str(PROJECT_ROOT / 'main.py'),
           '-i', seq['full_path'], '-r', seq['res'], '-p', seq['pix_fmt'],
           '--bit_depth', str(seq.get('bit_depth', 8)),
           '--loader', loader, '--device', device,
           '-c', str(out_csv)]
    if n_frames:
        cmd += ['-f', str(n_frames)]
    cmd += PROFILE_FLAGS[profile] + extra_args

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f'EVCA failed for {seq["name"]}/{profile}:\n{proc.stderr}')

    m = _TIME_RE.search(proc.stdout)
    compute_s = float(m.group(1)) if m else wall
    frames = len(pd.read_csv(out_csv))
    return {'frames': frames, 'seconds': compute_s, 'wall_seconds': wall,
            'fps': frames / compute_s if compute_s > 0 else float('nan'),
            'cmd': ' '.join(shlex.quote(c) for c in cmd)}


def collect_evca_frames(sequences: list, profiles: list, n_frames: int, out_dir: Path,
                        device: str, extra_args: list, loader: str) -> tuple:
    """Runs every (sequence, profile) and returns (wide per-frame df, fps df)."""
    csv_dir = out_dir / 'evca_csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    per_seq, fps_rows = [], []

    for seq in sequences:
        frames_df = None
        for profile in profiles:
            out_csv = csv_dir / f"{seq['name']}_{profile}.csv"
            print(f"  [EVCA] {seq['name']} / {profile}", flush=True)
            info = run_evca(seq, profile, n_frames, out_csv, device, extra_args, loader)
            fps_rows.append({'seq_name': seq['name'], 'profile': profile, **info})

            df = pd.read_csv(out_csv)
            df = df.rename(columns={c: f'{profile}_{c}' for c in df.columns})
            df['frame_idx'] = range(len(df))
            frames_df = df if frames_df is None else frames_df.merge(df, on='frame_idx')
        frames_df['seq_name'] = seq['name']
        per_seq.append(frames_df)

    evca = pd.concat(per_seq, ignore_index=True)
    front = ['seq_name', 'frame_idx']
    return evca[front + [c for c in evca.columns if c not in front]], pd.DataFrame(fps_rows)


def append_ledger(out_dir: Path, args, cfg: dict, sha: str, fps_df: pd.DataFrame,
                  head: pd.DataFrame, seq_mean: pd.DataFrame, notes: list,
                  within: pd.DataFrame = None, mc_health: pd.DataFrame = None) -> None:
    """Appends this run's summary section to validation/RESULTS.md."""
    lines = [
        '',
        f'### Run `{args.label}` — {datetime.now():%Y-%m-%d %H:%M}',
        '',
        f'- Phase: {args.phase}',
        f'- Commit: `{git_sha(short=False)}`',
        f'- Subset: **{args.subset}** ({"all frames" if not SUBSET_FRAMES[args.subset] else str(SUBSET_FRAMES[args.subset]) + " frames"}), '
        f'sequences: {", ".join(s["name"] for s in cfg["sequences"]) or "none"}',
        f'- Device: `{args.device}`, loader: `{args.loader}`, profiles: {", ".join(args.profiles)}',
        f'- Extra EVCA args: `{" ".join(args.extra_args_list) or "(none)"}`',
        f'- Bootstrap: {args.n_boot} resamples, seed {args.seed}',
        f'- Results: `{out_dir.relative_to(SCRIPT_DIR.parent)}`',
        '',
    ]
    for note in notes:
        lines.append(f'- {note}')
    if notes:
        lines.append('')

    if not fps_df.empty:
        fps_tbl = (fps_df.groupby('profile')
                   .agg(frames=('frames', 'sum'), seconds=('seconds', 'sum'))
                   .assign(fps=lambda d: d['frames'] / d['seconds'])
                   .reset_index())
        lines += ['**Throughput**', '', format_markdown(fps_tbl, '{:.2f}'), '']

    if within is not None and not within.empty:
        lines += ['**Mean within-sequence correlations** — the decision statistic. '
                  '`stat` is `PCC_log` for temporal metrics (bits grow with the log '
                  'of residual variance) and `PCC` for spatial ones; `primary_min`/'
                  '`primary_max` are the worst and best single sequence.', '',
                  format_markdown(within), '']
    if mc_health is not None and not mc_health.empty:
        lines += ['**Motion-search and intra-gate health**, per sequence, means over '
                  'frames ≥ 1. `intra_frac` says how much of `TC_MC` is the intra '
                  'fallback rather than motion compensation; `MV_sat_frac` says how '
                  'much of `TC_SAD` is search failure rather than content complexity.',
                  '', format_markdown(mc_health), '']
    if not head.empty:
        lines += ['**Frame-level pooled correlations** (reported, not decided on — '
                  'pooling mixes in between-sequence content ranking). `primary` is '
                  'the domain statistic named in `stat`; CI = 95 % bootstrap, '
                  '`blk` = sequence-level block bootstrap.', '',
                  format_markdown(head), '']
    if not seq_mean.empty:
        lines += ['**Sequence-mean correlations** (legacy, n = sequences)', '',
                  format_markdown(seq_mean), '']

    with open(RESULTS_MD, 'a') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> int:
    p = argparse.ArgumentParser(description='EVCA2 correlation benchmark.')
    p.add_argument('--subset', choices=['fast', 'full'], default='fast')
    p.add_argument('--label', required=True, help='short run name; names the output dir')
    p.add_argument('--phase', default='', help='phase tag recorded in the ledger')
    p.add_argument('--profiles', default='baseline,fast,full')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    p.add_argument('--loader', default='optimized', choices=['standard', 'optimized'])
    p.add_argument('--extra-args', default='', help='extra flags passed to main.py')
    p.add_argument('--config', default=str(gt.DEFAULT_CONFIG))
    p.add_argument('--sequence-root', default=None)
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--force-reencode', action='store_true')
    p.add_argument('--skip-ledger', action='store_true')
    p.add_argument('--x265-preset', default='medium')
    args = p.parse_args()

    args.profiles = [s.strip() for s in args.profiles.split(',') if s.strip()]
    args.extra_args_list = shlex.split(args.extra_args)
    for profile in args.profiles:
        if profile not in PROFILE_FLAGS:
            p.error(f'unknown profile {profile!r}; choose from {list(PROFILE_FLAGS)}')

    cfg = gt.load_config(Path(args.config), args.sequence_root)
    qps = cfg.get('qps', [22, 27, 32, 37])
    n_frames = SUBSET_FRAMES[args.subset]
    sha = git_sha()
    out_dir = RESULTS_DIR / f'{args.label}_{sha}'
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = []
    if cfg['missing']:
        notes.append('Missing sequences (skipped): '
                     + ', '.join(s['name'] for s in cfg['missing']))
        print('WARNING: missing sequences: '
              + ', '.join(s['name'] for s in cfg['missing']), file=sys.stderr)
    if not cfg['sequences']:
        notes.append('**No sequences available** — nothing was measured. '
                     'Run the synthetic-only pytest suite instead.')
        print('ERROR: no sequences found; nothing to do.', file=sys.stderr)
        if not args.skip_ledger:
            append_ledger(out_dir, args, cfg, sha, pd.DataFrame(),
                          pd.DataFrame(), pd.DataFrame(), notes)
        return 1

    print(f'Stage 1/3: EVCA extraction ({args.subset} subset)...', flush=True)
    evca, fps_df = collect_evca_frames(cfg['sequences'], args.profiles, n_frames,
                                       out_dir, args.device, args.extra_args_list,
                                       args.loader)
    evca.to_csv(out_dir / 'evca_frames.csv', index=False)
    fps_df.to_csv(out_dir / 'fps.csv', index=False)

    mc_health = mc_health_table(evca)
    if not mc_health.empty:
        mc_health.to_csv(out_dir / 'mc_health.csv', index=False)

    has_ffmpeg = gt.ffmpeg_available()
    frame_corr = seq_corr = head = within = pd.DataFrame()
    if not has_ffmpeg:
        notes.append('**ffmpeg/libx265 unavailable** — EVCA features and fps were '
                     'measured, but no ground truth or correlations were computed.')
        print('WARNING: ffmpeg with libx265 not available; skipping ground truth.',
              file=sys.stderr)
    else:
        print('Stage 2/3: ground truth (x265 AI + LDP)...', flush=True)
        gt_frames = gt.build_ground_truth(cfg['sequences'], qps, n_frames, CACHE_DIR,
                                          force=args.force_reencode,
                                          preset=args.x265_preset)
        gt_frames.to_csv(out_dir / 'ground_truth_frames.csv', index=False)
        gt_means = gt.sequence_mean_table(gt_frames)
        gt_means.to_csv(out_dir / 'ground_truth_results.csv', index=False)

        print('Stage 3/3: correlations...', flush=True)
        frame_corr = frame_level_report(evca, gt_frames, qps,
                                        n_boot=args.n_boot, seed=args.seed)
        frame_corr.to_csv(out_dir / 'frame_level_correlations.csv', index=False)
        seq_corr = sequence_mean_report(evca, gt_means, qps)
        seq_corr.to_csv(out_dir / 'sequence_mean_correlations.csv', index=False)
        headline = [m for m in HEADLINE if m in evca.columns]
        within = mean_within_table(frame_corr, headline)
        within.to_csv(out_dir / 'mean_within_correlations.csv', index=False)
        head = headline_table(frame_corr, headline, df_evca=evca)

    meta = {
        'label': args.label, 'phase': args.phase, 'subset': args.subset,
        'git_sha': git_sha(short=False), 'device': args.device,
        'loader': args.loader, 'profiles': args.profiles,
        'extra_args': args.extra_args_list, 'qps': qps, 'n_frames': n_frames,
        'n_boot': args.n_boot, 'seed': args.seed,
        'sequences': [s['name'] for s in cfg['sequences']],
        'missing_sequences': [s['name'] for s in cfg['missing']],
        'ffmpeg_available': has_ffmpeg,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'evca_commands': fps_df['cmd'].tolist() if not fps_df.empty else [],
    }
    with open(out_dir / 'run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    if not args.skip_ledger:
        append_ledger(out_dir, args, cfg, sha, fps_df, head, seq_corr, notes,
                      within=within, mc_health=mc_health)

    print(f'\nResults written to {out_dir}')
    if not within.empty:
        print('\n=== Mean within-sequence correlations (decision statistic) ===')
        print(within.to_string(index=False))
    if not mc_health.empty:
        print('\n=== Motion-search / intra-gate health ===')
        print(mc_health.to_string(index=False))
    if not head.empty:
        print('\n=== Frame-level pooled correlations (reported) ===')
        print(head.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
