"""Derived parameter combinations common in cosmological analyses."""

from __future__ import annotations

import numpy as np


def S8(sigma8: float | np.ndarray, omega_m: float | np.ndarray) -> np.ndarray:
    r"""Return S8 = σ8 sqrt(Ωm/0.3), with Ωm defined at z=0."""
    s, om = np.asarray(sigma8, float), np.asarray(omega_m, float)
    if np.any(s < 0) or np.any(om <= 0) or np.any(~np.isfinite(s)) or np.any(~np.isfinite(om)):
        raise ValueError("sigma8 must be finite/non-negative and omega_m finite/positive")
    return s * np.sqrt(om / 0.3)


def omega_cb(omega_b: float, omega_cdm: float) -> float:
    """Return Ω_cb=Ω_b+Ω_cdm, excluding neutrinos."""
    if omega_b < 0 or omega_cdm < 0:
        raise ValueError("density fractions must be non-negative")
    return float(omega_b + omega_cdm)


def neutrino_fraction(sum_mnu_ev: float, omega_nu0: float, omega_m0: float) -> float:
    """Return f_nu=Ω_nu/Ω_m at z=0 using density fractions from a specified physical model."""
    if sum_mnu_ev < 0 or omega_nu0 < 0 or omega_m0 <= 0:
        raise ValueError("invalid neutrino mass or density values")
    return float(omega_nu0 / omega_m0)


def dimensionless_wavenumber_h(k_mpc_inv: np.ndarray, h: float) -> np.ndarray:
    """Convert k [1/Mpc] to k [h/Mpc]."""
    k = np.asarray(k_mpc_inv, float)
    if h <= 0 or np.any(k < 0) or np.any(~np.isfinite(k)):
        raise ValueError("k must be finite/non-negative and h positive")
    return k / h


def power_to_h_units(pk_mpc3: np.ndarray, h: float) -> np.ndarray:
    """Convert P [Mpc^3] to P [(Mpc/h)^3]."""
    p = np.asarray(pk_mpc3, float)
    if h <= 0 or np.any(p < 0) or np.any(~np.isfinite(p)):
        raise ValueError("P must be finite/non-negative and h positive")
    return p * h**3


__all__ = ["S8", "omega_cb", "neutrino_fraction", "dimensionless_wavenumber_h", "power_to_h_units"]
