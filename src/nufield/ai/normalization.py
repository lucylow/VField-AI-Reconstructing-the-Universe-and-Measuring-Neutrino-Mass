"""Training-only feature scaling and invertible parameter transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Standardizer:
    """Column standardizer fitted only on a declared training array."""

    eps: float = 1e-8
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    n_fit_rows_: int = 0

    def fit(self, x: Any) -> Standardizer:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2:
            raise ValueError("Standardizer.fit expects a matrix [N,F], N>=2")
        if not np.isfinite(arr).all():
            raise ValueError("training features must be finite")
        self.mean_ = arr.mean(axis=0)
        raw_scale = arr.std(axis=0, ddof=0)
        self.scale_ = np.where(raw_scale > self.eps, raw_scale, 1.0)
        self.n_fit_rows_ = int(len(arr))
        return self

    def transform(self, x: Any) -> np.ndarray:
        self._check_fitted()
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.mean_.shape[0]:
            raise ValueError("feature width mismatch")
        if not np.isfinite(arr).all():
            raise ValueError("features must be finite")
        return ((arr - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, x: Any) -> np.ndarray:
        self._check_fitted()
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != self.mean_.shape[0]:
            raise ValueError("feature width mismatch")
        return arr * self.scale_ + self.mean_

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "kind": "standard",
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "n_fit_rows": self.n_fit_rows_,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> Standardizer:
        obj = cls(eps=float(state.get("eps", 1e-8)))
        obj.mean_ = np.asarray(state["mean"], dtype=float)
        obj.scale_ = np.asarray(state["scale"], dtype=float)
        obj.n_fit_rows_ = int(state.get("n_fit_rows", 0))
        if obj.mean_.ndim != 1 or obj.scale_.shape != obj.mean_.shape or np.any(obj.scale_ <= 0):
            raise ValueError("invalid standardizer state")
        return obj


@dataclass
class RobustStandardizer:
    """Median/IQR scaling for heavy-tailed summary features.

    This should be selected before evaluation, not chosen after inspecting test results.
    """

    eps: float = 1e-8
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    n_fit_rows_: int = 0

    def fit(self, x: Any) -> RobustStandardizer:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or len(arr) < 2 or not np.isfinite(arr).all():
            raise ValueError("fit expects a finite [N,F] matrix with N>=2")
        self.center_ = np.median(arr, axis=0)
        q25, q75 = np.quantile(arr, [0.25, 0.75], axis=0)
        spread = (q75 - q25) / 1.349
        self.scale_ = np.where(spread > self.eps, spread, 1.0)
        self.n_fit_rows_ = int(len(arr))
        return self

    def transform(self, x: Any) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called first")
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1] != len(self.center_) or not np.isfinite(arr).all():
            raise ValueError("invalid feature matrix")
        return ((arr - self.center_) / self.scale_).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called first")
        return {
            "kind": "robust",
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
            "n_fit_rows": self.n_fit_rows_,
            "eps": self.eps,
        }


@dataclass(frozen=True)
class BoxTransform:
    """Map physical parameters in an open box to normalized coordinates and back."""

    low: tuple[float, ...]
    high: tuple[float, ...]
    eps: float = 1e-6

    def __post_init__(self) -> None:
        lo, hi = np.asarray(self.low, float), np.asarray(self.high, float)
        if lo.ndim != 1 or lo.shape != hi.shape or len(lo) == 0:
            raise ValueError("low and high must be same-length vectors")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
            raise ValueError("bounds must be finite and high > low")
        if not 0 < self.eps < 0.1:
            raise ValueError("eps must be in (0, 0.1)")

    def forward(self, theta: Any) -> np.ndarray:
        x = np.asarray(theta, float)
        lo, hi = np.asarray(self.low), np.asarray(self.high)
        if x.shape[-1] != len(lo) or not np.isfinite(x).all():
            raise ValueError("theta shape or values invalid")
        if np.any(x < lo) or np.any(x > hi):
            raise ValueError("theta lies outside declared bounds")
        y = (x - lo) / (hi - lo)
        y = np.clip(y, self.eps, 1 - self.eps)
        return np.log(y) - np.log1p(-y)

    def inverse(self, z: Any) -> np.ndarray:
        arr = np.asarray(z, float)
        lo, hi = np.asarray(self.low), np.asarray(self.high)
        if arr.shape[-1] != len(lo) or not np.isfinite(arr).all():
            raise ValueError("transformed parameters shape or values invalid")
        y = np.where(arr >= 0, 1 / (1 + np.exp(-arr)), np.exp(arr) / (1 + np.exp(arr)))
        return lo + y * (hi - lo)
