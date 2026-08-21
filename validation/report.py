"""Frame-level and sequence-mean correlation reports.

Frame alignment: EVCA row `f` carries temporal metrics describing the transition
`f-1 -> f`, which is exactly what the Low-Delay-P encoder spends bits on when
coding P-frame `f`. Frame 0 is the I-frame in the LDP stream and is excluded from
every temporal comparison. Spatial metrics are compared against the All-Intra bits
of the same frame index, with all frames retained.
"""
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from validation.stats import DEFAULT_N_BOOT, DEFAULT_SEED, correlation_record

# Metric -> ground-truth column. Temporal metrics are compared against the LDP
# P-frame bits, spatial metrics against the All-Intra bits.
TEMPORAL_SUFFIXES = ['TC', 'TC2', 'TC_SAD', 'MVC', 'TC_MC', 'TC_SAD_full',
                     'mean_mv_mag', 'MV_sat_frac', 'intra_frac',
                     'MV_coherence', 'MV_div', 'MV_curl', 'MVD_cost',
                     'skip_frac', 'GMV_mag']
SPATIAL_SUFFIXES = ['SC', 'SC_u', 'SC_v', 'Colorfulness', 'B']


def classify(metric_col: str) -> str:
    """'Temporal', 'Spatial', or '' for a `<profile>_<metric>` column name."""
    suffix = metric_col.split('_', 1)[1] if '_' in metric_col else metric_col
    if suffix in TEMPORAL_SUFFIXES:
        return 'Temporal'
    if suffix in SPATIAL_SUFFIXES:
        return 'Spatial'
    return ''


def frame_level_report(df_evca: pd.DataFrame, df_gt: pd.DataFrame, qps: Sequence[int],
                       n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                       metrics: List[str] = None) -> pd.DataFrame:
    """Per-QP, per-metric frame-level correlations with bootstrap CIs.

    `df_evca` is wide: seq_name, frame_idx, and one column per `<profile>_<metric>`.
    `df_gt` is the per-frame ground truth from `ground_truth.build_ground_truth`.
    Emits one row per (QP, metric, scope) where scope is 'pooled' or a sequence name.
    """
    id_cols = {'seq_name', 'frame_idx'}
    if metrics is None:
        metrics = [c for c in df_evca.columns if c not in id_cols and classify(c)]

    records = []
    for qp in qps:
        gt_qp = df_gt[df_gt['qp'] == qp]
        if gt_qp.empty:
            continue
        merged = df_evca.merge(gt_qp, on=['seq_name', 'frame_idx'], how='inner')
        for metric in metrics:
            domain = classify(metric)
            if metric not in merged.columns:
                continue
            if domain == 'Temporal':
                sub = merged[(merged['frame_idx'] > 0) & (merged['pict_type'] == 'P')]
                gt_col = 'bits_ldp'
            else:
                sub = merged
                gt_col = 'bits_ai'
            sub = sub[np.isfinite(sub[metric]) & np.isfinite(sub[gt_col])]
            if len(sub) < 3:
                continue

            groups = [(g[metric].to_numpy(float), g[gt_col].to_numpy(float))
                      for _, g in sub.groupby('seq_name') if len(g) >= 3]
            rec = correlation_record(metric, 'pooled',
                                     sub[metric].to_numpy(float),
                                     sub[gt_col].to_numpy(float),
                                     groups=groups, n_boot=n_boot, seed=seed)
            records.append({'Domain': domain, 'QP': qp, **rec})

            for seq_name, g in sub.groupby('seq_name'):
                if len(g) < 3:
                    continue
                rec = correlation_record(metric, seq_name,
                                         g[metric].to_numpy(float),
                                         g[gt_col].to_numpy(float),
                                         n_boot=n_boot, seed=seed)
                records.append({'Domain': domain, 'QP': qp, **rec})

    cols = ['Domain', 'QP', 'metric', 'scope', 'n', 'PCC', 'PCC_lo', 'PCC_hi',
            'SRCC', 'SRCC_lo', 'SRCC_hi', 'PCC_log', 'PCC_log_lo', 'PCC_log_hi',
            'PCC_blk_lo', 'PCC_blk_hi', 'SRCC_blk_lo', 'SRCC_blk_hi']
    df = pd.DataFrame(records)
    return df[cols] if not df.empty else pd.DataFrame(columns=cols)


def sequence_mean_report(df_evca: pd.DataFrame, df_gt_means: pd.DataFrame,
                         qps: Sequence[int]) -> pd.DataFrame:
    """Legacy sequence-mean correlations (n = number of sequences per QP).

    Kept for continuity with the pre-Phase-1 ledger. EVCA metrics are averaged over
    frames >= 1, matching the original pipeline.
    """
    metrics = [c for c in df_evca.columns
               if c not in {'seq_name', 'frame_idx'} and classify(c)]
    means = (df_evca[df_evca['frame_idx'] >= 1]
             .groupby('seq_name')[metrics].mean().reset_index())

    records = []
    for qp in qps:
        merged = df_gt_means[df_gt_means['qp'] == qp].merge(means, on='seq_name')
        if len(merged) < 3:
            continue
        for metric in metrics:
            domain = classify(metric)
            gt_col = 'TC_gt' if domain == 'Temporal' else 'SC_gt'
            x = merged[metric].to_numpy(float)
            y = merged[gt_col].to_numpy(float)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                continue
            from validation.stats import pcc, srcc
            records.append({'Domain': domain, 'QP': qp, 'Metric': metric,
                            'PCC': pcc(x, y), 'SRCC': srcc(x, y), 'n': len(merged)})
    return pd.DataFrame(records)


def headline_table(df_frame: pd.DataFrame, metrics: Sequence[str] = None) -> pd.DataFrame:
    """Compact pooled view: one row per (metric, QP) with PCC and its frame CI."""
    pooled = df_frame[df_frame['scope'] == 'pooled']
    if metrics is not None:
        pooled = pooled[pooled['metric'].isin(metrics)]
    return pooled[['Domain', 'QP', 'metric', 'n', 'PCC', 'PCC_lo', 'PCC_hi',
                   'PCC_blk_lo', 'PCC_blk_hi', 'SRCC', 'PCC_log']].reset_index(drop=True)


def format_markdown(df: pd.DataFrame, floatfmt: str = '{:.4f}') -> str:
    """Renders a DataFrame as a GitHub-flavored markdown table."""
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
