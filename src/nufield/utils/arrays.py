"""Array validation helpers used at model boundaries."""

from __future__ import annotations

import numpy as np


def require_finite(name: str, value: np.ndarray, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def require_shape(name: str, value: np.ndarray, shape: tuple[int | None, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != len(shape):
        raise ValueError(f"{name} expected {len(shape)} dimensions")
    for i, (actual, expected) in enumerate(zip(arr.shape, shape)):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} dimension {i} expected {expected}, got {actual}")
    return arr


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill: float = np.nan) -> np.ndarray:
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    return np.divide(
        a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), fill, dtype=float), where=b != 0
    )
