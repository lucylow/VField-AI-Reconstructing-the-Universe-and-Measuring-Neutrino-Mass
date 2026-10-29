"""Model committee utilities for ensemble diversity and seed-stability auditing."""

from __future__ import annotations

from typing import Any

import numpy as np


def pairwise_prediction_correlation(
    member_predictions: Any, *, target_axis: int = -1
) -> dict[str, Any]:
    """Compute member prediction correlations over samples and targets.

    High correlation is not inherently bad when the target is easy; very high correlation plus low
    ensemble spread means the committee may understate epistemic uncertainty. Very low correlation
    can also signal unstable models, so compare with accuracy and calibration.
    """
    p = np.asarray(member_predictions, float)
    if p.ndim < 3 or p.shape[0] < 2 or not np.isfinite(p).all():
        raise ValueError("member_predictions must be finite [M,N,...] with M>=2")
    flat = p.reshape(p.shape[0], -1)
    corr = np.corrcoef(flat)
    upper = corr[np.triu_indices(len(corr), k=1)]
    return {
        "correlation_matrix": corr.tolist(),
        "mean_pairwise_correlation": float(np.nanmean(upper)),
        "min_pairwise_correlation": float(np.nanmin(upper)),
        "max_pairwise_correlation": float(np.nanmax(upper)),
        "warning": "Correlation alone is not a selection criterion; evaluate predictive skill and coverage.",
    }


def ensemble_diversity_report(member_predictions: Any, targets: Any) -> dict[str, Any]:
    """Compare individual-member and mean-ensemble RMSE, spread and pairwise correlation."""
    p = np.asarray(member_predictions, float)
    y = np.asarray(targets, float)
    if y.ndim == 1:
        y = y[:, None]
    if p.ndim == 2:
        p = p[:, :, None]
    if (
        p.ndim != 3
        or p.shape[1:] != y.shape
        or p.shape[0] < 2
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("member_predictions must be [M,N,P] aligned with targets [N,P]")
    member_rmse = np.sqrt(np.mean((p - y[None, :, :]) ** 2, axis=(1, 2)))
    mean_pred = p.mean(axis=0)
    return {
        "member_rmse": member_rmse.tolist(),
        "member_rmse_mean": float(member_rmse.mean()),
        "ensemble_mean_rmse": float(np.sqrt(np.mean((mean_pred - y) ** 2))),
        "committee": pairwise_prediction_correlation(p),
        "epistemic_variance_mean": float(p.var(axis=0, ddof=1).mean()),
        "warning": "The ensemble score is valid only on a held-out partition not used to pick members.",
    }
