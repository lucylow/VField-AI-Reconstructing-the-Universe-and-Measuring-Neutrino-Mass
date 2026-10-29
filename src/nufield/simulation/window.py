"""Toy observation windows and completeness weights."""

from __future__ import annotations

import numpy as np


def apply_selection_probability(
    counts: np.ndarray, probability: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Binomially thin counts with a per-cell selection probability."""
    c, p = np.asarray(counts), np.asarray(probability, dtype=float)
    if c.shape != p.shape or c.ndim != 3 or np.any(c < 0) or np.any(c != np.floor(c)):
        raise ValueError("counts and selection probability must be same-shaped 3-D arrays")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("selection probabilities must lie in [0,1]")
    return rng.binomial(c.astype(np.int64), p).astype(np.int32)


def inverse_probability_weights(probability: np.ndarray, max_weight: float = 10.0) -> np.ndarray:
    """Compute capped inverse-probability weights, retaining zero for unobserved cells."""
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or max_weight < 1:
        raise ValueError("invalid selection probabilities or max_weight")
    out = np.zeros_like(p)
    visible = p > 0
    out[visible] = np.minimum(1.0 / p[visible], max_weight)
    return out.astype(np.float32)


def make_rectangular_window(shape: tuple[int, int, int], keep_fraction: float = 0.8) -> np.ndarray:
    """Create a simple centered box window for algorithm smoke tests."""
    if len(shape) != 3 or any(n < 2 for n in shape) or not 0 < keep_fraction <= 1:
        raise ValueError("invalid shape or keep_fraction")
    mask = np.zeros(shape, dtype=bool)
    widths = [max(1, int(round(n * keep_fraction))) for n in shape]
    starts = [(n - w) // 2 for n, w in zip(shape, widths)]
    slices = tuple(slice(s, s + w) for s, w in zip(starts, widths))
    mask[slices] = True
    return mask
