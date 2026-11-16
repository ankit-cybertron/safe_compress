"""Computation of refusal metrics, bootstrap confidence intervals, and composite ranks."""

from __future__ import annotations

import random
from typing import Sequence, Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def refusal_gap(refusal_rate: float, baseline_fp16: float = 0.655) -> float:
    """Calculate the difference between an arm's refusal rate and the FP16 baseline.
    
    Positive indicates higher refusal than FP16; negative indicates refusal degradation.
    """
    return refusal_rate - baseline_fp16


def compute_bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for the mean.
    
    Args:
        values: Sequence of numerical values (e.g. binary indicators 0/1).
        n_bootstrap: Number of resamples (default 10,000 as in published stats).
        alpha: Two-tailed significance level (0.05 corresponds to 95% CI).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    if len(values) == 0:
        return (0.0, 0.0)
    
    vals = list(values)
    n = len(vals)
    
    if _HAS_NUMPY:
        rng = np.random.default_rng(seed)
        arr = np.asarray(vals, dtype=float)
        resamples = rng.choice(arr, size=(n_bootstrap, n), replace=True)
        resample_means = np.mean(resamples, axis=1)
        lower = float(np.percentile(resample_means, 100 * (alpha / 2)))
        upper = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
        return (lower, upper)
    else:
        rng = random.Random(seed)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = [vals[rng.randrange(n)] for _ in range(n)]
            boot_means.append(sum(sample) / n)
        boot_means.sort()
        idx_lo = int((alpha / 2) * n_bootstrap)
        idx_hi = int((1 - alpha / 2) * n_bootstrap)
        idx_hi = min(idx_hi, n_bootstrap - 1)
        return (float(boot_means[idx_lo]), float(boot_means[idx_hi]))


def paired_difference_ci(
    arm_values: Sequence[float],
    control_values: Sequence[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Compute mean paired difference (arm - control) and 95% bootstrap CI.
    
    Args:
        arm_values: Paired evaluation scores for test arm.
        control_values: Paired evaluation scores for control arm (e.g., E1_uniform_W4).
        n_bootstrap: Number of resamples (10,000).
        alpha: Significance level.
        seed: Random seed.
        
    Returns:
        (mean_diff, (ci_lower, ci_upper))
    """
    if len(arm_values) != len(control_values):
        raise ValueError(f"Lengths must match: {len(arm_values)} vs {len(control_values)}")
    
    diffs = [float(a) - float(c) for a, c in zip(arm_values, control_values)]
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    ci = compute_bootstrap_ci(diffs, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
    return mean_diff, ci


def rank_layers_by_composite(
    importance_scores: Sequence[float],
    fragility_scores: Sequence[float],
) -> list[int]:
    """Rank layers by normalized composite score: normalized(importance) * normalized(fragility).
    
    Returns:
        Layer indices sorted descending by composite score.
    """
    imp = list(importance_scores)
    frg = list(fragility_scores)
    min_i, max_i = min(imp), max(imp)
    min_f, max_f = min(frg), max(frg)
    
    range_i = (max_i - min_i) or 1e-12
    range_f = (max_f - min_f) or 1e-12
    
    scores = []
    for idx, (i_val, f_val) in enumerate(zip(imp, frg)):
        norm_i = (i_val - min_i) / range_i
        norm_f = (f_val - min_f) / range_f
        scores.append((norm_i * norm_f, idx))
        
    scores.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in scores]
