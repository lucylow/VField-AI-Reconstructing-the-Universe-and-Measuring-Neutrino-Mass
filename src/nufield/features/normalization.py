"""Feature transforms with explicit fit state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RobustScaler:
    median_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> RobustScaler:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError("x must be finite and 2-D")
        self.median_ = np.median(arr, axis=0)
        q75, q25 = np.quantile(arr, [0.75, 0.25], axis=0)
        spread = q75 - q25
        self.scale_ = np.where(spread > 1e-10, spread, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.scale_ is None:
            raise RuntimeError("fit scaler using training data before transform")
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1] != len(self.median_):
            raise ValueError("feature width mismatch")
        return ((arr - self.median_) / self.scale_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def winsorize_by_training_quantiles(
    train: np.ndarray, values: np.ndarray, lower: float = 0.005, upper: float = 0.995
) -> np.ndarray:
    """Clip values to quantiles estimated from training data only."""
    tr, val = np.asarray(train, float), np.asarray(values, float)
    if tr.ndim != 2 or val.shape[-1] != tr.shape[1] or not 0 <= lower < upper <= 1:
        raise ValueError("invalid shapes or quantile bounds")
    lo, hi = np.quantile(tr, [lower, upper], axis=0)
    return np.clip(val, lo, hi).astype(np.float32)
