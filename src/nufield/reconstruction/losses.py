"""Reconstruction loss functions for future validated field datasets."""

from __future__ import annotations

import numpy as np


def voxel_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    p, t = np.asarray(prediction, float), np.asarray(target, float)
    if p.shape != t.shape or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("prediction and target must be same-shaped finite arrays")
    return float(np.mean((p - t) ** 2))


def spectral_log_mse(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    """Compare log radial spectra without asserting these spectra have a science normalization."""
    p, t = np.asarray(prediction, float), np.asarray(target, float)
    if p.shape != t.shape or np.any(p < 0) or np.any(t < 0) or eps <= 0:
        raise ValueError("spectra must be same-shaped and non-negative")
    return float(np.mean((np.log(p + eps) - np.log(t + eps)) ** 2))


def combined_reconstruction_loss(
    prediction: np.ndarray, target: np.ndarray, spectral_weight: float = 0.1
) -> dict[str, float]:
    """Compute a transparent map-space and spectrum-space objective for tuning experiments."""
    if spectral_weight < 0:
        raise ValueError("spectral_weight must be non-negative")
    from nufield.features.power_spectrum import radial_power_spectrum

    mse = voxel_mse(prediction, target)
    _, p, _ = radial_power_spectrum(prediction, box_size_mpc_h=1.0, n_bins=8)
    _, t, _ = radial_power_spectrum(target, box_size_mpc_h=1.0, n_bins=8)
    valid = np.isfinite(p) & np.isfinite(t)
    spectral = spectral_log_mse(p[valid], t[valid]) if valid.any() else float("inf")
    return {
        "voxel_mse": mse,
        "spectral_log_mse": spectral,
        "combined": mse + spectral_weight * spectral,
    }
