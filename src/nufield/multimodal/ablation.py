"""Matched ablation result comparison utilities."""

from __future__ import annotations

import numpy as np


def paired_metric_delta(
    full_model: np.ndarray,
    ablated_model: np.ndarray,
    metric_name: str,
    direction: str = "lower_is_better",
) -> dict[str, float | str]:
    """Compare per-observation metrics only when both methods were tested on same rows."""
    full, abl = np.asarray(full_model, float), np.asarray(ablated_model, float)
    if (
        full.shape != abl.shape
        or full.ndim != 1
        or len(full) < 2
        or not np.isfinite(full).all()
        or not np.isfinite(abl).all()
    ):
        raise ValueError("metric arrays must be same-shaped finite vectors with at least two rows")
    if direction not in {"lower_is_better", "higher_is_better"}:
        raise ValueError("unknown metric direction")
    delta = full - abl
    if direction == "higher_is_better":
        interpretation = "positive favors full model"
    else:
        interpretation = "negative favors full model"
    return {
        "metric": metric_name,
        "mean_full": float(full.mean()),
        "mean_ablated": float(abl.mean()),
        "mean_delta_full_minus_ablated": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "paired_delta_std": float(delta.std(ddof=1)),
        "interpretation": interpretation,
        "n_paired_observations": int(len(delta)),
    }


def require_matched_priors(prior_hash_full: str, prior_hash_ablated: str) -> None:
    if not prior_hash_full or prior_hash_full != prior_hash_ablated:
        raise ValueError("ablation comparison requires matched, explicitly hashed priors")
