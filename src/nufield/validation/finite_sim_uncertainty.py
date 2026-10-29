"""Finite-simulation uncertainty approximations for reporting."""

from __future__ import annotations

import numpy as np


def binomial_coverage_interval(covered: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for empirical coverage proportion."""
    if total <= 0 or covered < 0 or covered > total or z <= 0:
        raise ValueError("invalid binomial counts or z")
    p = covered / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def standard_error_of_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).ravel()
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("need at least two finite values")
    return float(x.std(ddof=1) / np.sqrt(len(x)))
