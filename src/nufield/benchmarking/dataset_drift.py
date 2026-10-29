"""Quantify simple distribution shift between train and evaluation data."""

from __future__ import annotations

import numpy as np


def standardized_mean_difference(
    reference: np.ndarray, comparison: np.ndarray, *, eps: float = 1e-8
) -> np.ndarray:
    a = np.asarray(reference, dtype=float)
    b = np.asarray(comparison, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("arrays must be [N,F] with matching feature dimensions")
    if len(a) < 2 or len(b) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("at least two finite samples per array required")
    pooled = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2)
    return (b.mean(axis=0) - a.mean(axis=0)) / np.maximum(pooled, eps)


def profile_shift(reference: np.ndarray, comparison: np.ndarray, *, threshold: float = 0.5) -> dict:
    smd = standardized_mean_difference(reference, comparison)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    flagged = np.flatnonzero(np.abs(smd) >= threshold)
    return {
        "n_features": int(len(smd)),
        "threshold_abs_smd": float(threshold),
        "n_flagged": int(len(flagged)),
        "flagged_feature_indices": flagged.tolist(),
        "max_abs_smd": float(np.max(np.abs(smd))),
        "median_abs_smd": float(np.median(np.abs(smd))),
        "warning": "A feature-level drift flag is a diagnostic, not a calibrated domain-shift detector or proof of scientific bias.",
    }
