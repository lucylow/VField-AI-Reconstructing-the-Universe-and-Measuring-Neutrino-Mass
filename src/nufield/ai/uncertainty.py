"""Predictive uncertainty decomposition and stable ensemble summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnsembleUncertainty:
    mean: np.ndarray
    total_variance: np.ndarray
    epistemic_variance: np.ndarray
    aleatoric_variance: np.ndarray
    member_means: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "total_variance": self.total_variance.tolist(),
            "epistemic_variance": self.epistemic_variance.tolist(),
            "aleatoric_variance": self.aleatoric_variance.tolist(),
            "n_members": int(self.member_means.shape[0]),
            "warning": "Empirical ensemble spread is a heuristic epistemic measure; validate calibration.",
        }


def decompose_ensemble(
    member_means: Any, member_variances: Any | None = None, *, ddof: int = 1
) -> EnsembleUncertainty:
    """Apply law of total variance to a stack of independently trained model predictions.

    ``member_means`` shape is [M,...,P], where M>=2. If member variances are given, they must
    match the same shape and represent conditional aleatoric variances in the same target units.
    """
    means = np.asarray(member_means, dtype=np.float64)
    if means.ndim < 2 or means.shape[0] < 2:
        raise ValueError("member_means must have shape [M,...,P] and M>=2")
    if not np.isfinite(means).all():
        raise ValueError("member means must be finite")
    epistemic = means.var(axis=0, ddof=ddof)
    if member_variances is None:
        aleatoric = np.zeros_like(epistemic)
    else:
        variances = np.asarray(member_variances, dtype=np.float64)
        if (
            variances.shape != means.shape
            or not np.isfinite(variances).all()
            or np.any(variances < 0)
        ):
            raise ValueError("member variances must be finite, nonnegative and match member means")
        aleatoric = variances.mean(axis=0)
    return EnsembleUncertainty(
        means.mean(axis=0), epistemic + aleatoric, epistemic, aleatoric, means
    )


def gaussian_entropy(variance: Any, *, axis: int = -1) -> np.ndarray:
    """Entropy of independent Gaussian dimensions, up to an additive unit convention."""
    var = np.asarray(variance, float)
    if not np.isfinite(var).all() or np.any(var <= 0):
        raise ValueError("variance must be finite and positive")
    return 0.5 * np.sum(np.log(2 * np.pi * np.e * var), axis=axis)


def normalized_ensemble_disagreement(
    member_means: Any, prior_widths: Any, *, eps: float = 1e-12
) -> np.ndarray:
    """Dimensionless spread by scaling each parameter with its prior width."""
    means = np.asarray(member_means, float)
    widths = np.asarray(prior_widths, float)
    if means.ndim != 3 or means.shape[0] < 2:
        raise ValueError("member_means must be [M,N,P] with M>=2")
    if widths.shape != (means.shape[-1],) or np.any(widths <= 0) or not np.isfinite(widths).all():
        raise ValueError("prior_widths must be positive [P]")
    if not np.isfinite(means).all():
        raise ValueError("predictions must be finite")
    scaled = means / widths[None, None, :]
    return np.sqrt(scaled.var(axis=0, ddof=1).mean(axis=-1) + eps)


def predictive_interval(mean: Any, variance: Any, *, z_value: float = 1.959963984540054):
    """Return central Gaussian intervals; appropriate only when Gaussian calibration is supported."""
    m, v = np.asarray(mean, float), np.asarray(variance, float)
    if m.shape != v.shape or not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError("mean and variance must be finite, same-shaped, with nonnegative variance")
    if z_value <= 0:
        raise ValueError("z_value must be positive")
    radius = z_value * np.sqrt(v)
    return m - radius, m + radius
