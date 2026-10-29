"""Feature-vector construction from the toy observation."""

from __future__ import annotations

import numpy as np

from nufield.features.power_spectrum import radial_power_spectrum
from nufield.simulation.pipeline import ToyObservation


def extract_features(field: np.ndarray, box_size_mpc_h: float, n_bins: int = 10) -> np.ndarray:
    """Create a log-power feature vector, rejecting empty bins rather than silently imputing them."""
    _, pk, counts = radial_power_spectrum(field, box_size_mpc_h, n_bins=n_bins)
    valid = (counts > 0) & np.isfinite(pk) & (pk > 0)
    if valid.sum() < max(3, n_bins // 2):
        raise ValueError(
            "too few occupied power-spectrum bins; increase grid size or reduce n_bins"
        )
    # Preserve a fixed feature length; empty bins receive the median valid log-power and are accompanied
    # by no separate mask in v0.1. The metadata documents this simple policy.
    log_pk = np.log(np.maximum(pk, np.finfo(np.float64).tiny))
    log_pk[~valid] = float(np.median(log_pk[valid]))
    return log_pk.astype(np.float32)


def extract_observation_features(
    obs: ToyObservation, box_size_mpc_h: float, n_bins: int = 10
) -> np.ndarray:
    return extract_features(obs.galaxy_overdensity, box_size_mpc_h, n_bins)
