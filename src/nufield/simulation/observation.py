"""Composable observation effects for controlled software experiments."""

from __future__ import annotations

import numpy as np


def apply_voxel_mask(field: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Apply a broadcastable mask without changing the mask's meaning silently."""
    x, m = np.asarray(field), np.asarray(mask, dtype=bool)
    if x.shape != m.shape:
        raise ValueError("field and mask must have identical shape; broadcast masks explicitly")
    if not np.isfinite(x).all():
        raise ValueError("field must be finite")
    return np.where(m, x, fill_value).astype(np.float32)


def inject_additive_noise(
    field: np.ndarray, standard_deviation: float, rng: np.random.Generator
) -> np.ndarray:
    """Add independent Gaussian noise for synthetic stress testing only."""
    x = np.asarray(field, dtype=np.float32)
    if standard_deviation < 0 or not np.isfinite(x).all():
        raise ValueError("noise scale must be non-negative and field finite")
    if standard_deviation == 0:
        return x.copy()
    return (x + rng.normal(0, standard_deviation, size=x.shape)).astype(np.float32)


def random_dropout_mask(
    shape: tuple[int, ...], keep_probability: float, rng: np.random.Generator
) -> np.ndarray:
    """Create a Bernoulli visibility mask; not a realistic angular survey mask."""
    if not 0 < keep_probability <= 1 or not shape or any(n < 1 for n in shape):
        raise ValueError("invalid shape or keep_probability")
    return rng.random(shape) < keep_probability


def mask_fraction(mask: np.ndarray) -> float:
    """Return the fraction of visible elements."""
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        raise ValueError("mask cannot be empty")
    return float(arr.mean())
