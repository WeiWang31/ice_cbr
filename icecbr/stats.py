from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import stats


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1):
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return {"dm_stat": np.nan, "p_value": np.nan}
    mean_d = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    var_d = gamma0
    for lag in range(1, min(h, d.size - 1)):
        cov = np.cov(d[:-lag], d[lag:], ddof=1)[0, 1]
        var_d += 2 * cov
    denom = math.sqrt(max(var_d / d.size, 1e-12))
    dm_stat = mean_d / denom
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}


def block_bootstrap_ci(values: Iterable[float], block_size: int = 12, n_boot: int = 1000, alpha: float = 0.05, random_state: int = 42):
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "lower": np.nan, "upper": np.nan}
    rng = np.random.default_rng(random_state)
    means = []
    n = arr.size
    starts = np.arange(max(1, n - block_size + 1))
    for _ in range(n_boot):
        sample = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            sample.extend(arr[start : start + block_size].tolist())
        means.append(np.mean(sample[:n]))
    lower = np.quantile(means, alpha / 2)
    upper = np.quantile(means, 1 - alpha / 2)
    return {"mean": float(np.mean(arr)), "lower": float(lower), "upper": float(upper)}


def paired_significance(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 3:
        return {"test": "none", "stat": np.nan, "p_value": np.nan}
    diff = a - b
    if stats.shapiro(diff).pvalue > 0.05:
        stat, pvalue = stats.ttest_rel(a, b)
        return {"test": "paired_t", "stat": float(stat), "p_value": float(pvalue)}
    stat, pvalue = stats.wilcoxon(a, b)
    return {"test": "wilcoxon", "stat": float(stat), "p_value": float(pvalue)}

