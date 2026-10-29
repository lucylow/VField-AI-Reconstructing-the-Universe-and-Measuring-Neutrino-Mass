"""Mode-counting formulas and finite-volume diagnostics for clustering forecasts."""

from __future__ import annotations

import numpy as np


def shell_mode_count(volume_mpc3: float, k_min: np.ndarray, k_max: np.ndarray) -> np.ndarray:
    r"""Approximate number of Fourier modes in a shell: V/(2π²) integral k²dk."""
    lo, hi = np.asarray(k_min, float), np.asarray(k_max, float)
    if lo.shape != hi.shape or np.any(lo < 0) or np.any(hi <= lo) or volume_mpc3 <= 0:
        raise ValueError("require matching bins with 0<=k_min<k_max and positive volume")
    return volume_mpc3 / (6 * np.pi**2) * (hi**3 - lo**3)


def fundamental_wavenumber_mpc_inv(box_size_mpc: float) -> float:
    """Smallest nonzero k magnitude in a periodic cubic box, 2π/L."""
    if box_size_mpc <= 0:
        raise ValueError("box_size_mpc must be positive")
    return float(2 * np.pi / box_size_mpc)


def nyquist_wavenumber_mpc_inv(grid_size: int, box_size_mpc: float) -> float:
    """Axis Nyquist frequency pi*N/L; corner modes lie farther out and are anisotropic."""
    if grid_size < 2 or box_size_mpc <= 0:
        raise ValueError("grid_size>=2 and box_size positive required")
    return float(np.pi * grid_size / box_size_mpc)


def gaussian_fractional_power_error(mode_counts: np.ndarray) -> np.ndarray:
    """Approximate fractional standard deviation sqrt(2/N_modes) for Gaussian auto-power."""
    n = np.asarray(mode_counts, float)
    if np.any(n <= 0) or np.any(~np.isfinite(n)):
        raise ValueError("mode counts must be finite and positive")
    return np.sqrt(2 / n)


def effective_independent_modes(weights: np.ndarray) -> float:
    """Kish effective sample size for arbitrary real mode weights: (sum w)^2/sum(w²)."""
    w = np.asarray(weights, float)
    if w.ndim != 1 or np.any(~np.isfinite(w)) or np.all(w == 0):
        raise ValueError("weights must be a finite nonzero vector")
    denom = np.sum(w * w)
    if denom == 0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


__all__ = [
    "shell_mode_count",
    "fundamental_wavenumber_mpc_inv",
    "nyquist_wavenumber_mpc_inv",
    "gaussian_fractional_power_error",
    "effective_independent_modes",
]
