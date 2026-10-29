"""Cross-modality diagnostics for paired toy maps."""

from __future__ import annotations

import numpy as np


def pearson_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("arrays must be finite and same-shaped")
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != x.shape or m.sum() < 3:
            raise ValueError("mask shape mismatch or fewer than three retained values")
        x, y = x[m], y[m]
    x, y = x.ravel(), y.ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def common_mask_fraction(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    if a.shape != b.shape:
        raise ValueError("masks must have same shape")
    union = np.logical_or(a, b).sum()
    inter = np.logical_and(a, b).sum()
    return {
        "fraction_a": float(a.mean()),
        "fraction_b": float(b.mean()),
        "intersection_over_union": float(inter / union) if union else 1.0,
        "fraction_jointly_observed": float(np.logical_and(a, b).mean()),
    }
