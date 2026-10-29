"""Posterior predictive comparison primitives."""

from __future__ import annotations

import numpy as np


def empirical_tail_probability(
    observed: float, replicated: np.ndarray, two_sided: bool = True
) -> float:
    """Compute a simple finite-sample tail probability for a pre-registered scalar summary."""
    sims = np.asarray(replicated, dtype=float)
    if sims.ndim != 1 or len(sims) < 1 or not np.isfinite(sims).all() or not np.isfinite(observed):
        raise ValueError("observed and replicated values must be finite")
    if two_sided:
        center = float(np.median(sims))
        exceed = np.sum(np.abs(sims - center) >= abs(observed - center))
    else:
        exceed = np.sum(sims >= observed)
    return float((exceed + 1) / (len(sims) + 1))


def compare_moments(observed: np.ndarray, replicated: np.ndarray) -> dict[str, float]:
    """Compare array-level mean and variance; designed for small smoke tests."""
    obs, rep = np.asarray(observed, dtype=float), np.asarray(replicated, dtype=float)
    if not np.isfinite(obs).all() or not np.isfinite(rep).all():
        raise ValueError("inputs must be finite")
    return {
        "observed_mean": float(obs.mean()),
        "replicated_mean": float(rep.mean()),
        "observed_variance": float(obs.var()),
        "replicated_variance": float(rep.var()),
        "mean_difference": float(obs.mean() - rep.mean()),
        "variance_difference": float(obs.var() - rep.var()),
    }
