"""Phenomenological BAO damping templates for controlled reconstruction experiments."""

from __future__ import annotations

import numpy as np


def gaussian_bao_damping(k_mpc_inv: np.ndarray | float, sigma_nl_mpc: float) -> np.ndarray:
    r"""Return exp[-k² Σ_nl²/2], the common Gaussian wiggle-damping template."""
    k = np.asarray(k_mpc_inv, float)
    if (
        np.any(k < 0)
        or np.any(~np.isfinite(k))
        or sigma_nl_mpc < 0
        or not np.isfinite(sigma_nl_mpc)
    ):
        raise ValueError("k must be finite/non-negative and sigma_nl_mpc finite/non-negative")
    return np.exp(-0.5 * (k * sigma_nl_mpc) ** 2)


def damped_bao_power(
    k_mpc_inv: np.ndarray,
    p_linear_mpc3: np.ndarray,
    p_smooth_mpc3: np.ndarray,
    sigma_nl_mpc: float,
) -> np.ndarray:
    r"""Return P_damped=P_smooth+exp[-k²Σ²/2](P_linear-P_smooth).

    This is a phenomenological broadening template. It does not calculate the nonlinear shift of the
    BAO peak or include mode-coupling, redshift-space anisotropy, reconstruction, or survey windows.
    """
    k, pl, ps = (
        np.asarray(k_mpc_inv, float),
        np.asarray(p_linear_mpc3, float),
        np.asarray(p_smooth_mpc3, float),
    )
    if (
        k.ndim != 1
        or pl.shape != k.shape
        or ps.shape != k.shape
        or np.any(k < 0)
        or np.any(pl <= 0)
        or np.any(ps <= 0)
    ):
        raise ValueError("k, linear P and smooth P must be matching vectors with positive spectra")
    out = ps + gaussian_bao_damping(k, sigma_nl_mpc) * (pl - ps)
    if np.any(out <= 0) or np.any(~np.isfinite(out)):
        raise ValueError("damped BAO template became non-positive or non-finite")
    return out


def estimate_damping_from_ratio(
    k: np.ndarray, wiggle_ratio: np.ndarray, sigma_nl_mpc: float
) -> np.ndarray:
    """Apply an explicit damping factor to a supplied fractional wiggle residual."""
    kk, w = np.asarray(k, float), np.asarray(wiggle_ratio, float)
    if kk.shape != w.shape or np.any(kk < 0) or np.any(~np.isfinite(kk + w)):
        raise ValueError("k and wiggle_ratio must be matching finite arrays")
    return gaussian_bao_damping(kk, sigma_nl_mpc) * w


__all__ = ["gaussian_bao_damping", "damped_bao_power", "estimate_damping_from_ratio"]
