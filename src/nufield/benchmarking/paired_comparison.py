"""Paired bootstrap model comparisons on the same held-out simulation IDs."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_metric_difference(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Estimate metric(B)-metric(A) with paired bootstrap across the same test simulations.

    Use per-simulation errors/score values, not a single aggregate metric. For RMSE comparisons,
    pass each simulation's squared error and state that the difference is in MSE; alternatively
    bootstrap a precomputed per-row loss. Positive differences mean B has larger loss.
    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) < 2:
        raise ValueError("paired metric arrays must be equal-length 1-D arrays with N>=2")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("metrics must be finite")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >=100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_bootstrap, len(d)))
    boot = d[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "n_paired_test_simulations": int(len(d)),
        "metric_difference_mean_b_minus_a": float(d.mean()),
        "confidence": confidence,
        "bootstrap_interval": [
            float(np.quantile(boot, alpha)),
            float(np.quantile(boot, 1 - alpha)),
        ],
        "bootstrap_replicates": n_bootstrap,
        "seed": seed,
        "interpretation": "Positive means model B has a larger value of the supplied metric; lower-is-better loss metrics favor negative differences.",
        "warning": "Paired bootstrap assumes the resampling units are independent; use parent simulation groups as units, not correlated voxels/crops.",
    }
