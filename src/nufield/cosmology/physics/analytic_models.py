"""Exact/simple analytic cosmologies used as numerical unit tests and teaching references."""

from __future__ import annotations

import numpy as np

from nufield.cosmology.physics.constants import C_KM_S


def eds_E(z: float | np.ndarray) -> np.ndarray:
    """Einstein--de Sitter expansion E(z)=(1+z)^(3/2)."""
    zz = np.asarray(z, float)
    if np.any(zz < 0) or np.any(~np.isfinite(zz)):
        raise ValueError("z must be finite and non-negative")
    return (1 + zz) ** 1.5


def eds_comoving_distance_mpc(z: float | np.ndarray, H0_km_s_mpc: float = 70.0) -> np.ndarray:
    """Exact Einstein--de Sitter radial comoving distance 2c/H0 [1-(1+z)^(-1/2)]."""
    zz = np.asarray(z, float)
    if np.any(zz < 0) or H0_km_s_mpc <= 0 or np.any(~np.isfinite(zz)):
        raise ValueError("z must be finite/non-negative and H0 positive")
    return 2 * C_KM_S / H0_km_s_mpc * (1 - 1 / np.sqrt(1 + zz))


def eds_lookback_time_gyr(z: float | np.ndarray, H0_km_s_mpc: float = 70.0) -> np.ndarray:
    """Exact Einstein--de Sitter lookback time in Gyr."""
    zz = np.asarray(z, float)
    if np.any(zz < 0) or H0_km_s_mpc <= 0:
        raise ValueError("z must be non-negative and H0 positive")
    seconds_per_gyr = 365.25 * 24 * 3600 * 1e9
    mpc_km = 3.085677581491367e19
    h0_s = H0_km_s_mpc / mpc_km
    age0 = 2 / (3 * h0_s) / seconds_per_gyr
    age_at_z = age0 / (1 + zz) ** 1.5
    return age0 - age_at_z


def eds_growth_factor(z: float | np.ndarray) -> np.ndarray:
    """Exact growing-mode D(a)=a normalized to one at z=0 for Einstein--de Sitter."""
    zz = np.asarray(z, float)
    if np.any(zz < 0):
        raise ValueError("z must be non-negative")
    return 1 / (1 + zz)


def eds_growth_rate(z: float | np.ndarray) -> np.ndarray:
    """Exact f=d ln D/d ln a=1 in Einstein--de Sitter."""
    zz = np.asarray(z, float)
    if np.any(zz < 0):
        raise ValueError("z must be non-negative")
    return np.ones_like(zz, dtype=float)


__all__ = [
    "eds_E",
    "eds_comoving_distance_mpc",
    "eds_lookback_time_gyr",
    "eds_growth_factor",
    "eds_growth_rate",
]
