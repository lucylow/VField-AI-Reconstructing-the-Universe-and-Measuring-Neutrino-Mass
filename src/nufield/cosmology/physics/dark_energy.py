"""Dark-energy equation-of-state helpers for the CPL parameterization."""

from __future__ import annotations

import numpy as np


def cpl_w_of_scale_factor(a: float | np.ndarray, w0: float, wa: float) -> np.ndarray:
    """Return w(a)=w0+wa(1-a), requiring positive scale factors no larger than unity."""
    aa = np.asarray(a, float)
    if np.any(aa <= 0) or np.any(aa > 1) or np.any(~np.isfinite(aa)) or not np.isfinite(w0 + wa):
        raise ValueError("a must be finite in (0,1] and w0/wa finite")
    return w0 + wa * (1 - aa)


def cpl_density_ratio(a: float | np.ndarray, w0: float, wa: float) -> np.ndarray:
    r"""Return rho_DE(a)/rho_DE(1)=a^{-3(1+w0+wa)}exp[-3wa(1-a)]."""
    aa = np.asarray(a, float)
    if np.any(aa <= 0) or np.any(aa > 1) or np.any(~np.isfinite(aa)) or not np.isfinite(w0 + wa):
        raise ValueError("a must be finite in (0,1] and w0/wa finite")
    return aa ** (-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - aa))


def w_of_redshift(z: float | np.ndarray, w0: float, wa: float) -> np.ndarray:
    """Return CPL w(z)=w0+wa z/(1+z)."""
    zz = np.asarray(z, float)
    if np.any(zz < 0) or np.any(~np.isfinite(zz)) or not np.isfinite(w0 + wa):
        raise ValueError("z must be finite and non-negative")
    return w0 + wa * zz / (1 + zz)


def density_ratio_from_redshift(z: float | np.ndarray, w0: float, wa: float) -> np.ndarray:
    """CPL dark-energy density ratio in redshift coordinates."""
    zz = np.asarray(z, float)
    if np.any(zz < 0) or np.any(~np.isfinite(zz)):
        raise ValueError("z must be finite and non-negative")
    return (1 + zz) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * zz / (1 + zz))


__all__ = [
    "cpl_w_of_scale_factor",
    "cpl_density_ratio",
    "w_of_redshift",
    "density_ratio_from_redshift",
]
