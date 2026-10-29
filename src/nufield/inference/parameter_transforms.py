"""Constrained parameter transforms for optimization and neural models."""

from __future__ import annotations

import numpy as np


def to_unit_interval(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    a, lo, hi = np.asarray(x, float), np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != hi.shape or a.shape[-1] != len(lo) or np.any(hi <= lo):
        raise ValueError("bounds or parameter shape invalid")
    result = (a - lo) / (hi - lo)
    if np.any((result < -1e-8) | (result > 1 + 1e-8)):
        raise ValueError("parameters outside bounds")
    return np.clip(result, 0, 1)


def from_unit_interval(u: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    unit, lo, hi = np.asarray(u, float), np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != hi.shape or unit.shape[-1] != len(lo) or np.any(hi <= lo):
        raise ValueError("bounds or parameter shape invalid")
    if np.any((unit < 0) | (unit > 1)):
        raise ValueError("unit interval values must lie in [0,1]")
    return lo + unit * (hi - lo)


def logit_transform(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map (0,1) values to the real line with clipped numerical endpoints."""
    x = np.asarray(u, float)
    if not 0 < eps < 0.5 or np.any((x < 0) | (x > 1)):
        raise ValueError("values must lie in [0,1] and eps in (0,0.5)")
    xc = np.clip(x, eps, 1 - eps)
    return np.log(xc) - np.log1p(-xc)


def sigmoid_transform(z: np.ndarray) -> np.ndarray:
    x = np.asarray(z, float)
    # Stable sigmoid without overflow for large negative values.
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1 + exp_x)
    return out
