"""Small utilities for deterministic toy initial conditions."""

from __future__ import annotations

import numpy as np


def gaussian_white_noise(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Draw a zero-mean unit-variance cubic Gaussian field."""
    if len(shape) != 3 or len(set(shape)) != 1 or any(n < 4 for n in shape):
        raise ValueError("shape must be cubic with each side >=4")
    return rng.normal(size=shape).astype(np.float32)


def remove_dc_mode(field: np.ndarray) -> np.ndarray:
    """Subtract the mean/DC component from a field."""
    x = np.asarray(field, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("field must be a finite 3-D array")
    return (x - x.mean()).astype(np.float32)


def normalize_rms(field: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    """Normalize RMS while guarding against a zero-variance field."""
    x = remove_dc_mode(field).astype(np.float64)
    if target_rms <= 0 or not np.isfinite(target_rms):
        raise ValueError("target_rms must be positive and finite")
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-12:
        raise ValueError("cannot normalize a degenerate field")
    return (x * (target_rms / rms)).astype(np.float32)
