"""Train-only feature standardization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> Standardizer:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError("x must be finite 2-D array")
        self.mean_ = arr.mean(axis=0)
        scale = arr.std(axis=0)
        self.scale_ = np.where(scale > 1e-10, scale, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer must be fit before transform")
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.mean_.shape[0]:
            raise ValueError("feature dimension mismatch")
        return ((arr - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
