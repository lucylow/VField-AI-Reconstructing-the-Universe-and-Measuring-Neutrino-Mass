"""Explicit low-complexity systematic templates for controlled robustness experiments."""

from __future__ import annotations

import numpy as np


def multiplicative_power_calibration(pk: np.ndarray, fractional_calibration: float) -> np.ndarray:
    """Apply a single multiplicative power-calibration nuisance, P_obs=(1+m)P_true."""
    p = np.asarray(pk, float)
    if np.any(p < 0) or np.any(~np.isfinite(p)) or not np.isfinite(fractional_calibration):
        raise ValueError("pk and calibration must be finite; pk must be non-negative")
    if fractional_calibration <= -1:
        raise ValueError("fractional_calibration must exceed -1")
    return (1 + fractional_calibration) * p


def broadband_polynomial(
    k_mpc_inv: np.ndarray,
    coefficients: np.ndarray,
    *,
    pivot_mpc_inv: float = 0.1,
    log_k: bool = True,
) -> np.ndarray:
    """Evaluate additive broad-band nuisance polynomial on centered/scaled k or ln(k)."""
    k = np.asarray(k_mpc_inv, float)
    c = np.asarray(coefficients, float)
    if k.ndim != 1 or c.ndim != 1 or c.size < 1 or np.any(k <= 0) or pivot_mpc_inv <= 0:
        raise ValueError("k and pivot must be positive; coefficients must be a non-empty vector")
    x = np.log(k / pivot_mpc_inv) if log_k else (k - pivot_mpc_inv) / pivot_mpc_inv
    return np.polynomial.polynomial.polyval(x, c)


def shifted_redshift_distribution(z: np.ndarray, nz: np.ndarray, delta_z: float) -> np.ndarray:
    """Shift n(z) by delta_z using interpolation and renormalize on the observed z support."""
    zz, nn = np.asarray(z, float), np.asarray(nz, float)
    if zz.ndim != 1 or nn.shape != zz.shape or len(zz) < 3 or np.any(np.diff(zz) <= 0):
        raise ValueError("z and nz must be matching increasing vectors with >=3 samples")
    if np.any(nn < 0) or np.any(~np.isfinite(nn)) or not np.isfinite(delta_z):
        raise ValueError("nz must be finite/nonnegative and delta_z finite")
    shifted = np.interp(zz - delta_z, zz, nn, left=0.0, right=0.0)
    from nufield.cosmology.physics.lensing import normalize_n_of_z

    return normalize_n_of_z(zz, shifted)


def smooth_multiplicative_distortion(
    k: np.ndarray, amplitude: float, pivot: float, power: float = 2.0
) -> np.ndarray:
    """Smooth transfer-like distortion 1 + A (k/kpivot)^p for nuisance stress tests only."""
    kk = np.asarray(k, float)
    if np.any(kk <= 0) or pivot <= 0 or not np.isfinite(amplitude + power):
        raise ValueError("k and pivot must be positive; parameters finite")
    result = 1 + amplitude * (kk / pivot) ** power
    if np.any(result <= 0):
        raise ValueError("distortion becomes non-positive")
    return result


def apply_power_distortion(pk: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """Apply a positive multiplicative nuisance template to P(k)."""
    p, d = np.asarray(pk, float), np.asarray(distortion, float)
    if p.shape != d.shape or np.any(p < 0) or np.any(d <= 0) or np.any(~np.isfinite(p + d)):
        raise ValueError("pk and positive distortion must be matching finite arrays")
    return p * d


__all__ = [
    "multiplicative_power_calibration",
    "broadband_polynomial",
    "shifted_redshift_distribution",
    "smooth_multiplicative_distortion",
    "apply_power_distortion",
]
