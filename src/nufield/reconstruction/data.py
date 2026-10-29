"""Array dataset helpers for field reconstruction experiments."""

from __future__ import annotations

import numpy as np


def validate_paired_fields(
    inputs: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate `(N,X,Y,Z)` pairs without copying unless needed."""
    x, y = np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)
    if x.ndim != 4 or y.ndim != 4 or x.shape != y.shape:
        raise ValueError("inputs and targets must have identical (N,X,Y,Z) shapes")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("need at least two paired finite samples")
    if len(set(x.shape[1:])) != 1:
        raise ValueError("fields must be cubic grids")
    return x, y


def train_only_normalize(
    inputs: np.ndarray, targets: np.ndarray, train_indices: np.ndarray
) -> dict:
    """Fit scalar normalization on training fields only and apply to all pairs."""
    x, y = validate_paired_fields(inputs, targets)
    idx = np.asarray(train_indices, dtype=int)
    if idx.ndim != 1 or len(idx) == 0 or np.any(idx < 0) or np.any(idx >= len(x)):
        raise ValueError("invalid train_indices")
    x_mean, x_std = float(x[idx].mean()), float(x[idx].std())
    y_mean, y_std = float(y[idx].mean()), float(y[idx].std())
    x_std = max(x_std, 1e-8)
    y_std = max(y_std, 1e-8)
    return {
        "inputs": (x - x_mean) / x_std,
        "targets": (y - y_mean) / y_std,
        "normalization": {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
    }
