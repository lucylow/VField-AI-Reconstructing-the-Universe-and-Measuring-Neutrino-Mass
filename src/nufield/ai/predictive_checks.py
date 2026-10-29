"""Posterior-predictive and simulation-to-observation discrepancy diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def standardized_discrepancy(
    observed: Any, replicated: Any, *, axis: int = 0, eps: float = 1e-12
) -> dict[str, Any]:
    """Compare observed summaries to an ensemble of replicated summaries.

    ``replicated`` has a simulation/replicate axis ``axis``. This calculation assumes the observed
    and replicated summaries share an exact definition and units; it does not fix survey selection or
    latent-model mismatch.
    """
    obs, rep = np.asarray(observed, float), np.asarray(replicated, float)
    if not np.isfinite(obs).all() or not np.isfinite(rep).all():
        raise ValueError("observed and replicated summaries must be finite")
    if rep.ndim != obs.ndim + 1:
        raise ValueError("replicated array must have exactly one extra replicate axis")
    axis = axis % rep.ndim
    if rep.shape[:axis] + rep.shape[axis + 1 :] != obs.shape:
        raise ValueError("observed shape must equal replicated shape with replicate axis removed")
    mean = rep.mean(axis=axis)
    std = rep.std(axis=axis, ddof=1)
    z = (obs - mean) / np.maximum(std, eps)
    # A descriptive Gaussian-tail discrepancy score, not a posterior-predictive p-value.
    return {
        "observed": obs.tolist(),
        "replicated_mean": mean.tolist(),
        "replicated_std": std.tolist(),
        "standardized_residual": z.tolist(),
        "max_abs_z": float(np.max(np.abs(z))),
        "fraction_abs_z_gt_2": float(np.mean(np.abs(z) > 2)),
        "warning": "Large discrepancies suggest model mismatch; low discrepancies do not prove adequacy.",
    }


def posterior_predictive_rank(
    observed_statistic: Any, replicated_statistics: Any, *, two_sided: bool = True
) -> dict[str, Any]:
    """Compute a simple posterior-predictive rank for a predeclared scalar discrepancy statistic."""
    obs = np.asarray(observed_statistic, float)
    rep = np.asarray(replicated_statistics, float).reshape(-1)
    if obs.size != 1 or len(rep) < 2 or not np.isfinite(obs).all() or not np.isfinite(rep).all():
        raise ValueError("observed statistic must be scalar and have >=2 finite replications")
    low = (np.sum(rep <= float(obs)) + 1) / (len(rep) + 1)
    p = min(1.0, 2 * min(low, 1 - low)) if two_sided else low
    return {
        "empirical_lower_tail": float(low),
        "empirical_two_sided_tail": float(p) if two_sided else None,
        "n_replications": int(len(rep)),
        "warning": "This is a diagnostic tail frequency, not a frequentist goodness-of-fit guarantee.",
    }


def featurewise_predictive_checks(
    observed_features: Any, replicated_features: Any, *, feature_names: list[str] | None = None
) -> list[dict[str, Any]]:
    """Build a named discrepancy table from replicated [R,F] features and observation [F]."""
    obs, rep = (
        np.asarray(observed_features, float).reshape(-1),
        np.asarray(replicated_features, float),
    )
    if rep.ndim != 2 or rep.shape[1] != len(obs) or len(rep) < 2:
        raise ValueError("replicated_features must be [R>=2,F] and align with observations")
    if not np.isfinite(obs).all() or not np.isfinite(rep).all():
        raise ValueError("features must be finite")
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(obs))]
    if len(feature_names) != len(obs) or len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique and match feature count")
    mean = rep.mean(axis=0)
    std = rep.std(axis=0, ddof=1)
    z = (obs - mean) / np.maximum(std, 1e-12)
    return [
        {
            "name": name,
            "observed": float(obs[i]),
            "replicated_mean": float(mean[i]),
            "replicated_std": float(std[i]),
            "z_score": float(z[i]),
            "flag_gt_3_sigma": bool(abs(z[i]) > 3),
        }
        for i, name in enumerate(feature_names)
    ]
