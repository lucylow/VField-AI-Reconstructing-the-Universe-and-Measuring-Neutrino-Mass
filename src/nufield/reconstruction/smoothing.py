"""Simple classical smoothing baseline."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_smooth(field: np.ndarray, sigma_voxels: float = 1.0) -> np.ndarray:
    """Smooth a field with periodic boundaries. This is not a BAO reconstruction method."""
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 3 or sigma_voxels < 0:
        raise ValueError("field must be 3-D and sigma non-negative")
    return gaussian_filter(arr, sigma=sigma_voxels, mode="wrap").astype(np.float32)


def reconstruction_metrics(reconstructed: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return basic map-space diagnostics; these alone do not establish cosmological usefulness."""
    a, b = np.asarray(reconstructed, dtype=float), np.asarray(target, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("reconstructed and target must be same-shaped finite arrays")
    error = a - b
    denom = np.sqrt(np.mean(b * b)) + 1e-12
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_rmse": float(np.sqrt(np.mean(error * error)) / denom),
        "correlation": float(corr),
    }
