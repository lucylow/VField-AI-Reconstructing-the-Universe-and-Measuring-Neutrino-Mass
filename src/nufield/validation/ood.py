"""Training-support diagnostics for detecting observations outside simulator feature support."""

from __future__ import annotations

import numpy as np


def fit_support_envelope(
    training_features: np.ndarray,
    *,
    quantile: float = 0.99,
    ridge: float = 1e-5,
    max_reference_rows: int = 5000,
    seed: int = 11,
) -> dict:
    """Fit a robust-ish Mahalanobis support envelope on training features only.

    This is a heuristic OOD flag, not a p-value. It must be fitted after a train-only transform and
    cannot detect all simulator misspecification or physical model errors.
    """
    x = np.asarray(training_features, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("training_features must be finite [N,F] with N>=3")
    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be in (0.5,1)")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    if len(x) > max_reference_rows:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), max_reference_rows, replace=False)]
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.asarray([[float(cov)]])
    cov = np.atleast_2d(cov)
    scale = max(float(np.trace(cov) / max(1, cov.shape[0])), 1e-12)
    cov_reg = cov + np.eye(cov.shape[0]) * ridge * scale
    precision = np.linalg.pinv(cov_reg, hermitian=True)
    delta = x - mean
    d2 = np.einsum("ni,ij,nj->n", delta, precision, delta)
    threshold = float(np.quantile(d2, quantile))
    return {
        "mean": mean.tolist(),
        "precision": precision.tolist(),
        "distance_squared_threshold": threshold,
        "threshold_quantile": quantile,
        "n_reference_used": len(x),
        "n_features": x.shape[1],
        "seed": seed,
        "warning": "A support envelope is only a warning system; thresholds need held-out calibration under the training simulator.",
    }


def score_support_envelope(model: dict, features: np.ndarray) -> dict:
    x = np.asarray(features, dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    precision = np.asarray(model["precision"], dtype=float)
    if x.ndim != 2 or x.shape[1] != len(mean):
        raise ValueError("features dimension mismatch")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    d = x - mean
    d2 = np.einsum("ni,ij,nj->n", d, precision, d)
    threshold = float(model["distance_squared_threshold"])
    return {
        "distance_squared": d2.tolist(),
        "threshold": threshold,
        "out_of_support": (d2 > threshold).tolist(),
        "fraction_flagged": float(np.mean(d2 > threshold)),
        "n_rows": len(x),
        "warning": "Flagged observations may reflect covariate shift, outliers, or physical misspecification; not an automatic rejection criterion.",
    }
