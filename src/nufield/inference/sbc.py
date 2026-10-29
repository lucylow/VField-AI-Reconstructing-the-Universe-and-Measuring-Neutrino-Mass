"""Simulation-based calibration utilities for ensembles of posterior samples."""

from __future__ import annotations

import numpy as np


def posterior_rank(samples: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compute randomized ranks (one rank per parameter) for discrete posterior draws."""
    s = np.asarray(samples)
    t = np.asarray(truth)
    if s.ndim != 2 or t.shape != (s.shape[1],):
        raise ValueError("samples must be (n_draws, n_params), truth must be (n_params,)")
    ranks = np.empty(s.shape[1], dtype=np.int64)
    for j in range(s.shape[1]):
        below = int(np.sum(s[:, j] < t[j]))
        equal = int(np.sum(s[:, j] == t[j]))
        ranks[j] = below + int(rng.integers(0, equal + 1))
    return ranks


def rank_histogram(ranks: np.ndarray, n_draws: int, n_bins: int = 20) -> np.ndarray:
    """Histogram randomized ranks for a single parameter across repeated simulations."""
    arr = np.asarray(ranks, dtype=np.float64)
    if arr.ndim != 1 or n_draws < 1 or n_bins < 2:
        raise ValueError("invalid ranks or histogram settings")
    if np.any(arr < 0) or np.any(arr > n_draws):
        raise ValueError("rank outside [0, n_draws]")
    return np.histogram(arr, bins=np.linspace(0, n_draws, n_bins + 1))[0]


def coverage_table(
    posterior_samples: list[np.ndarray], truths: np.ndarray, levels=(0.5, 0.68, 0.95)
) -> dict[str, list[float]]:
    """Compute per-parameter empirical central-interval coverage.

    This is a descriptive diagnostic; compute binomial uncertainty before interpreting results.
    """
    truths = np.asarray(truths, dtype=np.float64)
    if len(posterior_samples) != len(truths):
        raise ValueError("one posterior sample matrix is required per truth row")
    if not posterior_samples:
        raise ValueError("posterior_samples cannot be empty")
    n_params = truths.shape[1]
    output = {}
    for level in levels:
        covered = np.zeros(n_params, dtype=np.float64)
        for i, samples in enumerate(posterior_samples):
            s = np.asarray(samples)
            alpha = (1 - level) / 2
            lo, hi = np.quantile(s, [alpha, 1 - alpha], axis=0)
            covered += ((truths[i] >= lo) & (truths[i] <= hi)).astype(float)
        output[f"coverage_{level:.2f}"] = (covered / len(truths)).tolist()
    return output
