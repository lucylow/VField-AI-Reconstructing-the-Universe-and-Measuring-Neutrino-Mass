"""Post-processing checks for field predictions."""

from __future__ import annotations

import numpy as np


def remove_predicted_dc(field: np.ndarray) -> np.ndarray:
    x = np.asarray(field, float)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("field must be finite 3-D")
    return (x - x.mean()).astype(np.float32)


def normalize_to_reference_rms(prediction: np.ndarray, reference_rms: float) -> np.ndarray:
    x = np.asarray(prediction, float)
    if x.ndim != 3 or not np.isfinite(x).all() or reference_rms <= 0:
        raise ValueError("prediction must be finite 3-D and reference_rms positive")
    rms = np.sqrt(np.mean((x - x.mean()) ** 2))
    if rms < 1e-12:
        raise ValueError("prediction has zero RMS")
    return ((x - x.mean()) / rms * reference_rms).astype(np.float32)


def clip_extremes(field: np.ndarray, quantile: float = 0.999) -> np.ndarray:
    x = np.asarray(field, float)
    if x.ndim != 3 or not np.isfinite(x).all() or not 0.5 < quantile < 1:
        raise ValueError("invalid field or quantile")
    bound = np.quantile(np.abs(x), quantile)
    return np.clip(x, -bound, bound).astype(np.float32)
