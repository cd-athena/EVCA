"""Frame-level and sequence-mean correlation reports.

Frame alignment: EVCA row `f` carries temporal metrics describing the transition
`f-1 -> f`, which is exactly what the Low-Delay-P encoder spends bits on when
coding P-frame `f`. Frame 0 is the I-frame in the LDP stream and is excluded from
every temporal comparison. Spatial metrics are compared against the All-Intra bits
of the same frame index, with all frames retained.

Two conventions decide which number is the *decision* statistic:

* **Scope.** The pooled statistic mixes between-sequence content ranking into what
  is meant to be a per-frame prediction score, and with a handful of sequences the
  between-sequence term dominates. Measured on the macOS re-baseline, ranking by
  pooled PCC puts `mean_mv_mag` (average motion-vector length, a search diagnostic
  with *negative* within-sequence correlation) above `TC_MC`. `mean_within_table`
  therefore carries the ranking statistic; the pooled table stays as a report.
* **Transform.** Bits grow with the log of residual variance, so temporal metrics
  correlate better against `log(bits)`: `PCC_log` beats `PCC` for every temporal
  metric measured, by 0.15-0.25. `primary_stat` selects `PCC_log` for temporal
  metrics and plain `PCC` for spatial ones, where the two are equivalent.
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


def suffix_of(metric_col: str) -> str:
    """'full_TC_MC' -> 'TC_MC'; 'TC_MC' -> 'TC_MC'."""
    return metric_col.split('_', 1)[1] if '_' in metric_col else metric_col


def primary_stat(domain: str) -> str:
    """Correlation statistic a domain is judged on: 'PCC_log' or 'PCC'.

    Temporal metrics predict P-frame bits, which grow with the log of residual
    variance, so the log transform is the correctly specified one. Spatial metrics
    show no such gap and keep plain PCC.
    """
    return 'PCC_log' if domain == 'Temporal' else 'PCC'


def companion_column(metric_col: str) -> str:
    """EVCA column that must be read alongside `metric_col`, or '' if none.

    `TC_MC` is intra-gated (`min(SC_MC, SC)`), so where the gate fires often it is
    reporting spatial complexity rather than motion-compensation quality.
    `intra_frac` is the fraction of blocks where the gate fired and is the only way
    to tell the two cases apart, so it travels with every `TC_MC` figure.
    """
    if suffix_of(metric_col) != 'TC_MC':
        return ''
    profile = metric_col.rsplit('_TC_MC', 1)[0]
    return f'{profile}_intra_frac' if profile else 'intra_frac'


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


def headline_table(df_frame: pd.DataFrame, metrics: Sequence[str] = None,
                   df_evca: pd.DataFrame = None) -> pd.DataFrame:
    """Compact pooled view: one row per (metric, QP).

    `primary` carries the domain's decision statistic (`PCC_log` for temporal,
    `PCC` for spatial) with its bootstrap CI, so the leading number is the correctly
    specified one; plain PCC is retained beside it for continuity with earlier runs.
    When `df_evca` is supplied, `intra_frac` is attached to every `TC_MC` row.

    This is a *reported* table, not a decision table — it is pooled over sequences.
    Ranking decisions use `mean_within_table`.
    """
    pooled = df_frame[df_frame['scope'] == 'pooled'].copy()
    if metrics is not None:
        pooled = pooled[pooled['metric'].isin(metrics)]
    if pooled.empty:
        return pooled

    stat = pooled['Domain'].map(primary_stat)
    pooled['stat'] = stat
    for out, suffix in (('primary', ''), ('primary_lo', '_lo'), ('primary_hi', '_hi')):
        pooled[out] = [row[row['stat'] + suffix] for _, row in pooled.iterrows()]

    cols = ['Domain', 'QP', 'metric', 'n', 'stat', 'primary', 'primary_lo',
            'primary_hi', 'PCC', 'PCC_lo', 'PCC_hi', 'PCC_blk_lo', 'PCC_blk_hi', 'SRCC']
    if df_evca is not None:
        pooled['intra_frac'] = pooled['metric'].map(
            lambda m: _companion_mean(df_evca, m))
        if pooled['intra_frac'].notna().any():
            cols.append('intra_frac')
        else:
            pooled = pooled.drop(columns='intra_frac')
    return pooled[cols].reset_index(drop=True)


def _companion_mean(df_evca: pd.DataFrame, metric_col: str) -> float:
    """Corpus mean of `metric_col`'s companion column over frames >= 1, or NaN."""
    companion = companion_column(metric_col)
    if not companion or companion not in df_evca.columns:
        return float('nan')
    vals = df_evca.loc[df_evca['frame_idx'] >= 1, companion]
    return float(vals.mean()) if len(vals) else float('nan')


def mean_within_table(df_frame: pd.DataFrame, metrics: Sequence[str] = None) -> pd.DataFrame:
    """Mean within-sequence correlation per (metric, QP) — the decision statistic.

    Averages the per-sequence rows of `frame_level_report`, which removes the
    between-sequence term that dominates the pooled figure. `primary` is `PCC_log`
    for temporal metrics and `PCC` for spatial ones; `primary_min`/`primary_max`
    expose the spread across sequences, since a high mean built from one strong and
    one negative sequence is not the same result as a uniformly moderate one.
    """
    per = df_frame[df_frame['scope'] != 'pooled']
    if metrics is not None:
        per = per[per['metric'].isin(metrics)]
    if per.empty:
        return pd.DataFrame(columns=['Domain', 'QP', 'metric', 'n_seq', 'stat',
                                     'primary', 'primary_min', 'primary_max', 'PCC'])

    records = []
    for (domain, qp, metric), g in per.groupby(['Domain', 'QP', 'metric']):
        stat = primary_stat(domain)
        vals = g[stat].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        records.append({
            'Domain': domain, 'QP': qp, 'metric': metric, 'n_seq': len(g),
            'stat': stat, 'primary': float(vals.mean()),
            'primary_min': float(vals.min()), 'primary_max': float(vals.max()),
            'PCC': float(g['PCC'].mean()),
        })
    return pd.DataFrame(records).sort_values(['Domain', 'QP', 'metric']).reset_index(drop=True)


def mc_health_table(df_evca: pd.DataFrame) -> pd.DataFrame:
    """Per-sequence motion-search and intra-gate diagnostics, means over frames >= 1.

    Reported beside every `TC_MC` figure: `intra_frac` says how much of `TC_MC` is
    the intra fallback rather than motion compensation, and `MV_sat_frac` says how
    much of `TC_SAD` is search failure rather than content complexity.
    """
    wanted = ['MV_sat_frac', 'mean_mv_mag', 'intra_frac']
    cols = {}
    for suffix in wanted:
        hits = [c for c in df_evca.columns if suffix_of(c) == suffix]
        if hits:
            # Identical across profiles; prefer the richest one that carries it.
            cols[suffix] = sorted(hits)[-1]
    if not cols or 'seq_name' not in df_evca.columns:
        return pd.DataFrame()
    sub = df_evca[df_evca['frame_idx'] >= 1]
    out = (sub.groupby('seq_name')[list(cols.values())].mean()
           .rename(columns={v: k for k, v in cols.items()})
           .reset_index().rename(columns={'seq_name': 'sequence'}))
    return out.sort_values('sequence').reset_index(drop=True)


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
