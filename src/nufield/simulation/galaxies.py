"""Toy lognormal-Poisson tracer operator."""

from __future__ import annotations

import numpy as np


def sample_galaxy_counts(
    density_field: np.ndarray, bias: float, mean_counts_per_cell: float, rng: np.random.Generator
) -> np.ndarray:
    """Map a latent field to non-negative integer counts per voxel.

    This is a convenience observation operator, not an HOD, halo model, or DESI mock generator.
    """
    delta = np.asarray(density_field, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("density_field must be finite")
    if bias <= 0 or mean_counts_per_cell <= 0:
        raise ValueError("bias and mean_counts_per_cell must be positive")
    variance = float(np.var(delta))
    intensity = mean_counts_per_cell * np.exp(
        np.clip(bias * delta - 0.5 * bias * bias * variance, -15, 15)
    )
    return rng.poisson(intensity).astype(np.int32)


def counts_to_overdensity(counts: np.ndarray) -> np.ndarray:
    """Convert counts to dimensionless overdensity with safe handling of empty means."""
    arr = np.asarray(counts, dtype=np.float64)
    if np.any(arr < 0) or not np.isfinite(arr).all():
        raise ValueError("counts must be finite and non-negative")
    mean = float(arr.mean())
    if mean <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / mean - 1.0).astype(np.float32)
