"""Classical low-dimensional baseline helpers."""

from __future__ import annotations

import numpy as np


def nearest_training_particle(
    features_train: np.ndarray, theta_train: np.ndarray, features_test: np.ndarray
) -> np.ndarray:
    """Return the parameter vector associated with the nearest standardized feature vector.

    The function fits scaling to the training set only. It is a deterministic non-probabilistic
    baseline, useful to sanity-check whether a feature vector contains toy-model signal.
    """
    x = np.asarray(features_train, dtype=float)
    y = np.asarray(theta_train, dtype=float)
    xt = np.asarray(features_test, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or xt.ndim != 2 or len(x) != len(y):
        raise ValueError("invalid feature/parameter shapes")
    mu, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    xs, ts = (x - mu) / scale, (xt - mu) / scale
    distances = np.sum((ts[:, None, :] - xs[None, :, :]) ** 2, axis=-1)
    return y[np.argmin(distances, axis=1)]


def ridge_regression_fit(
    x: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> dict[str, np.ndarray | float]:
    """Fit a closed-form ridge baseline with an unpenalized intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or alpha < 0:
        raise ValueError("invalid inputs or alpha")
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return {"coef": coef, "x_mean": xm, "y_mean": ym, "alpha": float(alpha)}


def ridge_regression_predict(model: dict, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return (arr - model["x_mean"]) @ model["coef"] + model["y_mean"]
