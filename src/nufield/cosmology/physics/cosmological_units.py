"""Unit conversions for common cosmology quantities, with names that encode source units."""

from __future__ import annotations

import numpy as np


def k_h_mpc_to_mpc_inv(k_h_mpc: np.ndarray | float, h: float) -> np.ndarray:
    """Convert k in h/Mpc to physical k in 1/Mpc: k_phys = h * k_h."""
    k = np.asarray(k_h_mpc, float)
    if h <= 0 or np.any(k < 0) or np.any(~np.isfinite(k)):
        raise ValueError("k must be finite/non-negative and h positive")
    return h * k


def k_mpc_inv_to_h_mpc(k_mpc_inv: np.ndarray | float, h: float) -> np.ndarray:
    """Convert physical k in 1/Mpc to k in h/Mpc: k_h = k_phys / h."""
    k = np.asarray(k_mpc_inv, float)
    if h <= 0 or np.any(k < 0) or np.any(~np.isfinite(k)):
        raise ValueError("k must be finite/non-negative and h positive")
    return k / h


def pk_mpc3_to_mpc_over_h_cubed(pk_mpc3: np.ndarray | float, h: float) -> np.ndarray:
    """Convert P in Mpc³ to P in (Mpc/h)³: P_h = h³ P_phys."""
    p = np.asarray(pk_mpc3, float)
    if h <= 0 or np.any(p < 0) or np.any(~np.isfinite(p)):
        raise ValueError("P must be finite/non-negative and h positive")
    return p * h**3


def pk_mpc_over_h_cubed_to_mpc3(pk_h: np.ndarray | float, h: float) -> np.ndarray:
    """Convert P in (Mpc/h)³ to P in Mpc³: P_phys = P_h/h³."""
    p = np.asarray(pk_h, float)
    if h <= 0 or np.any(p < 0) or np.any(~np.isfinite(p)):
        raise ValueError("P must be finite/non-negative and h positive")
    return p / h**3


def radius_8_mpc(h: float) -> float:
    """Return physical radius corresponding to 8 h^-1 Mpc."""
    if h <= 0:
        raise ValueError("h must be positive")
    return 8.0 / h


def density_critical_msun_mpc3(h: float) -> float:
    """Return rho_crit,0 in Msun/Mpc³ under the standard 2.77536627e11 h² conversion."""
    if h <= 0:
        raise ValueError("h must be positive")
    return 2.77536627e11 * h**2


__all__ = [
    "k_h_mpc_to_mpc_inv",
    "k_mpc_inv_to_h_mpc",
    "pk_mpc3_to_mpc_over_h_cubed",
    "pk_mpc_over_h_cubed_to_mpc3",
    "radius_8_mpc",
    "density_critical_msun_mpc3",
]
