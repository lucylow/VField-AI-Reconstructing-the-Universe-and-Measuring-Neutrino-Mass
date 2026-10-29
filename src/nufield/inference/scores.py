"""Proper scoring rules and scalar uncertainty metrics."""

from __future__ import annotations

import numpy as np


def empirical_crps(samples: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Compute per-dimension empirical CRPS with O(N log N) pairwise term."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(observation, dtype=float)
    if (
        x.ndim != 2
        or y.shape != (x.shape[1],)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("samples must be (N,D), observation (D,), all finite")
    first = np.mean(np.abs(x - y[None, :]), axis=0)
    pair = np.empty(x.shape[1], dtype=float)
    n = len(x)
    for j in range(x.shape[1]):
        ordered = np.sort(x[:, j])
        coeff = 2 * np.arange(1, n + 1) - n - 1
        half_pair_expectation = np.sum(coeff * ordered) / (n * n)
        pair[j] = half_pair_expectation
    return first - pair


def interval_score(
    lower: np.ndarray, upper: np.ndarray, truth: np.ndarray, alpha: float
) -> np.ndarray:
    """Winkler-style interval score; lower is better for a calibrated forecast."""
    lo, hi, y = np.asarray(lower), np.asarray(upper), np.asarray(truth)
    if lo.shape != hi.shape or lo.shape != y.shape or not 0 < alpha < 1 or np.any(lo > hi):
        raise ValueError("invalid interval inputs")
    width = hi - lo
    below = (2 / alpha) * (lo - y) * (y < lo)
    above = (2 / alpha) * (y - hi) * (y > hi)
    return width + below + above


def z_score(
    prediction: np.ndarray, truth: np.ndarray, standard_deviation: np.ndarray
) -> np.ndarray:
    p, t, s = np.asarray(prediction), np.asarray(truth), np.asarray(standard_deviation)
    if p.shape != t.shape or p.shape != s.shape or np.any(s <= 0):
        raise ValueError("shapes must match and standard deviations be positive")
    return (p - t) / s
