"""Correlation statistics with bootstrap confidence intervals.

Two resampling schemes are provided:

* `bootstrap_ci`   - i.i.d. resampling over frames. Treats every frame as an
  independent observation; appropriate for a per-sequence CI.
* `block_bootstrap_ci` - resampling whole sequences with replacement (each drawn
  sequence contributes all of its frames). For pooled statistics this is the
  honest interval: frames within a sequence are strongly dependent, so the
  frame-level CI understates uncertainty about generalization to new content.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 12345


def _pcc_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of two [n_rows, n_obs] arrays."""
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    num = (xc * yc).sum(axis=-1)
    den = np.sqrt((xc ** 2).sum(axis=-1) * (yc ** 2).sum(axis=-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient (NaN if either input is constant)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        return float('nan')
    return float(_pcc_rows(x[None, :], y[None, :])[0])


def srcc(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        return float('nan')
    return pcc(rankdata(x), rankdata(y))


def pcc_log(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of x against log(y); y must be positive."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = y > 0
    if mask.sum() < 3:
        return float('nan')
    return pcc(x[mask], np.log(y[mask]))


def _resample_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_ci(x: Sequence[float], y: Sequence[float], kind: str = 'pcc',
                 n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI over i.i.d. frame resamples. Returns (lo, hi)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = _resample_indices(n, n_boot, rng)
    xb, yb = x[idx], y[idx]
    if kind == 'srcc':
        xb, yb = rankdata(xb, axis=1), rankdata(yb, axis=1)
    elif kind == 'pcc_log':
        keep = y > 0
        if keep.sum() < 3:
            return float('nan'), float('nan')
        xk, yk = x[keep], np.log(y[keep])
        idx = _resample_indices(xk.size, n_boot, rng)
        xb, yb = xk[idx], yk[idx]
    stats = _pcc_rows(xb, yb)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return float('nan'), float('nan')
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def block_bootstrap_ci(groups: List[Tuple[np.ndarray, np.ndarray]], kind: str = 'pcc',
                       n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                       alpha: float = 0.05) -> Tuple[float, float]:
    """Sequence-level block bootstrap: resample whole (x, y) groups with replacement.

    `groups` is a list of per-sequence (metric_values, gt_values) arrays.
    """
    if len(groups) < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        xb = np.concatenate([groups[i][0] for i in pick])
        yb = np.concatenate([groups[i][1] for i in pick])
        if kind == 'srcc':
            val = pcc(rankdata(xb), rankdata(yb))
        elif kind == 'pcc_log':
            val = pcc_log(xb, yb)
        else:
            val = pcc(xb, yb)
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return float('nan'), float('nan')
    stats = np.asarray(stats)
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def correlation_record(metric_name: str, scope: str, x: np.ndarray, y: np.ndarray,
                       groups: List[Tuple[np.ndarray, np.ndarray]] = None,
                       n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> Dict:
    """One row of the frame-level correlation report.

    When `groups` is given (pooled scope), the CI columns carry the sequence-level
    block bootstrap in addition to the frame-level one.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rec = {
        'metric': metric_name,
        'scope': scope,
        'n': int(x.size),
        'PCC': pcc(x, y),
        'SRCC': srcc(x, y),
        'PCC_log': pcc_log(x, y),
    }
    rec['PCC_lo'], rec['PCC_hi'] = bootstrap_ci(x, y, 'pcc', n_boot, seed)
    rec['SRCC_lo'], rec['SRCC_hi'] = bootstrap_ci(x, y, 'srcc', n_boot, seed)
    rec['PCC_log_lo'], rec['PCC_log_hi'] = bootstrap_ci(x, y, 'pcc_log', n_boot, seed)
    if groups is not None:
        rec['PCC_blk_lo'], rec['PCC_blk_hi'] = block_bootstrap_ci(groups, 'pcc', n_boot, seed)
        rec['SRCC_blk_lo'], rec['SRCC_blk_hi'] = block_bootstrap_ci(groups, 'srcc', n_boot, seed)
    else:
        rec['PCC_blk_lo'] = rec['PCC_blk_hi'] = float('nan')
        rec['SRCC_blk_lo'] = rec['SRCC_blk_hi'] = float('nan')
    return rec
