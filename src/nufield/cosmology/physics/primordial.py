"""Primordial scalar-curvature spectrum utilities."""

from __future__ import annotations

import math

import numpy as np

from nufield.cosmology.physics.parameters import PhysicalCosmology


def primordial_dimensionless_power(
    k_mpc_inv: float | np.ndarray,
    params: PhysicalCosmology,
    *,
    include_running: bool = True,
) -> np.ndarray:
    r"""Compute Δ_R²(k) using a power law with optional running.

    ln Δ_R² = ln A_s + (n_s-1) ln(k/k*) + alpha_s/2 [ln(k/k*)]².
    This is a primordial curvature spectrum, not the late-time matter spectrum.
    """
    k = np.asarray(k_mpc_inv, dtype=float)
    if np.any(k <= 0) or np.any(~np.isfinite(k)):
        raise ValueError("k_mpc_inv must be finite and positive")
    x = np.log(k / params.k_pivot_Mpc_inv)
    log_delta2 = math.log(params.A_s) + (params.n_s - 1.0) * x
    if include_running:
        log_delta2 = log_delta2 + 0.5 * params.alpha_s * x**2
    return np.exp(log_delta2)


def primordial_dimensional_power(
    k_mpc_inv: float | np.ndarray, params: PhysicalCosmology
) -> np.ndarray:
    """Return P_R(k) in Mpc^3 from Δ_R²=k³P_R/(2π²)."""
    k = np.asarray(k_mpc_inv, dtype=float)
    if np.any(k <= 0):
        raise ValueError("k_mpc_inv must be positive")
    return 2.0 * math.pi**2 * primordial_dimensionless_power(k, params) / k**3


__all__ = ["primordial_dimensionless_power", "primordial_dimensional_power"]
