"""Real-space density-field diagnostics for initial-condition and reconstruction tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, normaltest, skew


def field_moments(field: np.ndarray) -> dict[str, float]:
    """Return mean, variance, skewness and excess kurtosis from a finite array."""
    x = np.asarray(field, float)
    if x.size < 8 or np.any(~np.isfinite(x)):
        raise ValueError("field must contain at least eight finite values")
    flat = x.ravel()
    return {
        "mean": float(flat.mean()),
        "variance": float(flat.var()),
        "skewness": float(skew(flat, bias=False)),
        "excess_kurtosis": float(kurtosis(flat, fisher=True, bias=False)),
    }


def gaussianity_diagnostic(
    field: np.ndarray, *, max_samples: int = 20_000, seed: int = 0
) -> dict[str, object]:
    """Run a sampled D'Agostino-Pearson normality test and report its sampling limitation."""
    x = np.asarray(field, float)
    if x.size < 20 or max_samples < 20 or seed < 0 or np.any(~np.isfinite(x)):
        raise ValueError("field/max_samples/seed invalid")
    flat = x.ravel()
    if flat.size > max_samples:
        indices = np.random.default_rng(seed).choice(flat.size, max_samples, replace=False)
        flat = flat[indices]
    stat, pval = normaltest(flat)
    return {
        "test": "D'Agostino-Pearson K²",
        "n_samples": int(flat.size),
        "statistic": float(stat),
        "p_value": float(pval),
        "warning": "Grid cells are spatially correlated; this p-value is not a formal field-level test.",
    }


def periodic_mean_mode(field: np.ndarray) -> float:
    """Return mean/zero-mode diagnostic for a periodic field."""
    x = np.asarray(field, float)
    if x.ndim != 3 or np.any(~np.isfinite(x)):
        raise ValueError("field must be a finite 3-D array")
    return float(np.mean(x))


def reconstruction_residual_metrics(
    truth: np.ndarray, reconstruction: np.ndarray
) -> dict[str, float]:
    """Return RMSE, correlation coefficient and residual mean for matched fields."""
    t, r = np.asarray(truth, float), np.asarray(reconstruction, float)
    if t.shape != r.shape or t.size < 2 or np.any(~np.isfinite(t + r)):
        raise ValueError("truth/reconstruction must be matching finite arrays")
    residual = r - t
    corr = (
        float(np.corrcoef(t.ravel(), r.ravel())[0, 1])
        if t.std() > 0 and r.std() > 0
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "residual_mean": float(np.mean(residual)),
        "correlation": corr,
        "truth_variance": float(np.var(t)),
        "reconstruction_variance": float(np.var(r)),
    }


__all__ = [
    "field_moments",
    "gaussianity_diagnostic",
    "periodic_mean_mode",
    "reconstruction_residual_metrics",
]
