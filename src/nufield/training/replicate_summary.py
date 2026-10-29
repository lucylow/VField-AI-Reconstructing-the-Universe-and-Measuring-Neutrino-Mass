"""Summarize independent-seed experiment replicates without hiding uncertainty."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_replicates(
    values: np.ndarray,
    *,
    metric_name: str,
    direction: str,
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap replicate-level metric summaries. Input rows must be independent run seeds.

    This does not substitute for per-test-simulation paired intervals: it quantifies run-to-run
    variability, while object-level uncertainty is a separate experimental dimension.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("values must be finite 1-D array with at least two independent seeds")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be minimize or maximize")
    if not 0 < confidence < 1 or n_bootstrap < 500:
        raise ValueError("confidence must lie in (0,1), n_bootstrap >=500")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), size=(n_bootstrap, len(x)))
    means = x[ix].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "metric_name": metric_name,
        "direction": direction,
        "n_independent_seeds": len(x),
        "values": x.tolist(),
        "mean": float(x.mean()),
        "std_between_seeds": float(x.std(ddof=1)),
        "median": float(np.median(x)),
        "confidence": confidence,
        "bootstrap_mean_interval": [
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1 - alpha)),
        ],
        "best_seed_value": float(x.min() if direction == "minimize" else x.max()),
        "seed": seed,
        "warning": "Do not report the best seed as the expected model performance; report mean/uncertainty and predeclared selection procedure.",
    }
