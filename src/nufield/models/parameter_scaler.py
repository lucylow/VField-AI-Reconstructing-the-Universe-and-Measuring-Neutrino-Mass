"""Train-only coordinate scaling for supervised neural baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParameterScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, theta: np.ndarray) -> ParameterScaler:
        x = np.asarray(theta, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("theta must be finite [N,P] with N>=2")
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.scale_ = np.maximum(x.std(axis=0, keepdims=True), self.eps)
        return self

    def transform(self, theta: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must precede transform")
        x = np.asarray(theta, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.mean_.shape[1]:
            raise ValueError("theta shape mismatch")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def inverse_mean(self, scaled: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler is not fitted")
        return np.asarray(scaled) * self.scale_ + self.mean_

    def inverse_variance(self, scaled_variance: np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("scaler is not fitted")
        return np.asarray(scaled_variance) * np.square(self.scale_)

    def to_dict(self) -> dict:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler is not fitted")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist(), "eps": self.eps}
