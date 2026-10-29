"""Operations on weighted posterior particles."""

from __future__ import annotations

import numpy as np


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite non-negative 1-D with positive sum")
    return w / w.sum()


def systematic_resample(
    weights: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Return particle indices using systematic resampling."""
    w = normalize_weights(weights)
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    positions = (rng.random() + np.arange(n_samples)) / n_samples
    cumulative = np.cumsum(w)
    return np.searchsorted(cumulative, positions, side="left").clip(max=len(w) - 1)


def weighted_mean(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=float)
    w = normalize_weights(weights)
    if x.ndim != 2 or len(x) != len(w):
        raise ValueError("samples must be (N,D), weights (N,)")
    return np.sum(x * w[:, None], axis=0)


def weighted_quantile(
    samples: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Calculate per-parameter weighted quantiles by stable sorting."""
    x = np.asarray(samples, dtype=float)
    w = normalize_weights(weights)
    q = np.asarray(quantiles, dtype=float)
    if x.ndim != 2 or len(x) != len(w) or np.any((q < 0) | (q > 1)):
        raise ValueError("invalid inputs")
    out = np.empty((len(q), x.shape[1]), dtype=float)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="mergesort")
        xs, ws = x[order, j], w[order]
        cdf = np.cumsum(ws) - 0.5 * ws
        cdf = cdf / (ws.sum())
        out[:, j] = np.interp(q, cdf, xs, left=xs[0], right=xs[-1])
    return out
