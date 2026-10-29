"""Leakage-aware preprocessing for fields and hand-crafted spectral summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, x: np.ndarray) -> Standardizer:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim < 2 or len(arr) < 2:
            raise ValueError("fit expects [N,...] with N>=2")
        if not np.isfinite(arr).all():
            raise ValueError("training data must be finite")
        self.mean_ = arr.mean(axis=0, keepdims=True)
        self.scale_ = arr.std(axis=0, keepdims=True)
        self.scale_ = np.maximum(self.scale_, self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must precede transform")
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[1:] != self.mean_.shape[1:]:
            raise ValueError("feature shape differs from fitted shape")
        return ((arr - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Map standardized values back to the feature coordinates used during fit."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must precede inverse_transform")
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim < 2 or arr.shape[1:] != self.mean_.shape[1:]:
            raise ValueError("standardized feature shape differs from fitted shape")
        if not np.isfinite(arr).all():
            raise ValueError("standardized values must be finite")
        return (arr * self.scale_ + self.mean_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state_dict(self) -> dict:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("standardizer is not fitted")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict) -> Standardizer:
        obj = cls(eps=float(state.get("eps", 1e-6)))
        obj.mean_ = np.asarray(state["mean"], dtype=np.float64)
        obj.scale_ = np.asarray(state["scale"], dtype=np.float64)
        if (obj.scale_ <= 0).any():
            raise ValueError("stored scales must be positive")
        return obj


def normalize_overdensity(field: np.ndarray, *, clip: float | None = 8.0) -> np.ndarray:
    """Convert a count/density field to delta=(rho-mean(rho))/mean(rho), preserving shape."""
    arr = np.asarray(field, dtype=np.float32)
    if arr.ndim < 2 or not np.isfinite(arr).all():
        raise ValueError("field must be finite with sample plus spatial axes")
    axes = tuple(range(1, arr.ndim))
    mean = arr.mean(axis=axes, keepdims=True)
    if np.any(mean <= 0):
        raise ValueError("field mean must be positive for overdensity normalization")
    delta = arr / mean - 1.0
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive or None")
        delta = np.clip(delta, -clip, clip)
    return delta.astype(np.float32)


def log_counts(counts: np.ndarray) -> np.ndarray:
    """Variance-stabilizing transform for nonnegative count maps; not a substitute for a likelihood."""
    arr = np.asarray(counts)
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    return np.log1p(arr).astype(np.float32)


def crop_cube(
    field: np.ndarray, size: int, *, origin: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Deterministic cubic crop. If used, all crops from one parent stay in the same split."""
    arr = np.asarray(field)
    if arr.ndim != 3:
        raise ValueError("field must be a 3-D cube")
    if size <= 0 or size > min(arr.shape):
        raise ValueError("crop size must fit the field")
    if len(origin) != 3:
        raise ValueError("origin must contain three indices")
    end = tuple(origin[i] + size for i in range(3))
    if any(origin[i] < 0 or end[i] > arr.shape[i] for i in range(3)):
        raise ValueError("crop origin outside field")
    return arr[origin[0] : end[0], origin[1] : end[1], origin[2] : end[2]].copy()


@dataclass
class ChannelStandardizer:
    """Train-only per-channel normalization for [N,C,D,H,W] fields.

    Spatial dimensions are pooled when estimating channel statistics, preserving common treatment
    across positions. This is distinct from per-voxel normalization and should be ablated against a
    physically motivated density-contrast transform where appropriate.
    """

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, x: np.ndarray) -> ChannelStandardizer:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 5 or len(arr) < 2:
            raise ValueError("fit expects [N,C,D,H,W] with N>=2")
        if not np.isfinite(arr).all():
            raise ValueError("training fields must be finite")
        self.mean_ = arr.mean(axis=(0, 2, 3, 4), keepdims=True)
        self.scale_ = np.maximum(arr.std(axis=(0, 2, 3, 4), keepdims=True), self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must precede transform")
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 5 or arr.shape[1] != self.mean_.shape[1]:
            raise ValueError("field channel shape differs from fitted shape")
        if not np.isfinite(arr).all():
            raise ValueError("fields must be finite")
        return ((arr - self.mean_) / self.scale_).astype(np.float32)

    def state_dict(self) -> dict:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("standardizer is not fitted")
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "eps": self.eps,
            "kind": "channel",
        }
