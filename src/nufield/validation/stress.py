"""Controlled feature perturbations for software robustness tests."""

from __future__ import annotations

import numpy as np


def add_gaussian_feature_noise(
    features: np.ndarray, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Inject feature-space noise. This is not a substitute for observational systematics."""
    x = np.asarray(features, dtype=np.float32)
    if scale < 0 or not np.isfinite(x).all():
        raise ValueError("scale must be non-negative and features finite")
    return x + rng.normal(0, scale, size=x.shape).astype(np.float32)


def rescale_selected_modes(features: np.ndarray, indices: list[int], factor: float) -> np.ndarray:
    """Perturb selected feature columns to test sensitivity to normalization errors."""
    x = np.asarray(features, dtype=np.float32).copy()
    if factor <= 0 or any(i < 0 or i >= x.shape[1] for i in indices):
        raise ValueError("factor must be positive and feature indices valid")
    x[:, indices] *= factor
    return x


def stress_summary(reference: np.ndarray, perturbed: np.ndarray) -> dict[str, float]:
    a, b = np.asarray(reference, float), np.asarray(perturbed, float)
    if a.shape != b.shape:
        raise ValueError("arrays must have identical shapes")
    return {
        "mean_absolute_change": float(np.mean(np.abs(a - b))),
        "max_absolute_change": float(np.max(np.abs(a - b))),
        "relative_l2_change": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)),
    }
