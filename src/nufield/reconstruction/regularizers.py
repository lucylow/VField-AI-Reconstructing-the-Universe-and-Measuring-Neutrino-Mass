"""Numerical regularizers for reconstruction experiments."""

from __future__ import annotations

import numpy as np


def total_variation_3d(field: np.ndarray, epsilon: float = 1e-8) -> float:
    x = np.asarray(field, float)
    if x.ndim != 3 or not np.isfinite(x).all() or epsilon < 0:
        raise ValueError("field must be finite 3-D and epsilon non-negative")
    dx = np.roll(x, -1, 0) - x
    dy = np.roll(x, -1, 1) - x
    dz = np.roll(x, -1, 2) - x
    return float(np.mean(np.sqrt(dx * dx + dy * dy + dz * dz + epsilon)))


def mean_constraint_penalty(field: np.ndarray, target_mean: float = 0.0) -> float:
    x = np.asarray(field, float)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("field must be finite 3-D")
    return float((x.mean() - target_mean) ** 2)


def periodic_gradient_penalty(field: np.ndarray) -> float:
    x = np.asarray(field, float)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("field must be finite 3-D")
    grad = [(np.roll(x, -1, axis=i) - np.roll(x, 1, axis=i)) * 0.5 for i in range(3)]
    return float(np.mean(sum(g * g for g in grad)))
