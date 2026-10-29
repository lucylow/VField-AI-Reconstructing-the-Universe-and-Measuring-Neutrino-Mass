"""Bootstrap helpers for uncertainty on finite benchmark metrics."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_interval(
    values: np.ndarray, *, n_bootstrap: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for an independent-row mean."""
    x = np.asarray(values, dtype=float).ravel()
    if len(x) < 2 or not np.isfinite(x).all() or n_bootstrap < 100 or not 0 < level < 1:
        raise ValueError("need >=2 finite values, >=100 replicates, and level in (0,1)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_bootstrap, len(x)))
    means = x[idx].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap_difference(
    a: np.ndarray, b: np.ndarray, *, n_bootstrap: int = 2000, level: float = 0.95, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap for same-row metric arrays; preserves pairing."""
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if x.shape != y.shape or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("a and b must be same-shaped finite vectors")
    diff = x - y
    mean, lo, hi = bootstrap_mean_interval(diff, n_bootstrap=n_bootstrap, level=level, seed=seed)
    return {
        "mean_difference_a_minus_b": mean,
        "lower": lo,
        "upper": hi,
        "level": level,
        "n_pairs": int(len(diff)),
    }
