"""Linear Kaiser redshift-space distortion predictions and multipoles."""

from __future__ import annotations

import numpy as np


def kaiser_power_multipoles(
    pk_matter: float | np.ndarray,
    bias: float,
    growth_rate: float,
) -> dict[int, np.ndarray]:
    r"""Return linear monopole, quadrupole and hexadecapole of P_s(k,mu).

    The formulas assume linear, plane-parallel Kaiser theory without fingers-of-God, wide-angle,
    relativistic, nonlinear bias, or scale-dependent growth corrections.
    """
    if bias <= 0 or not np.isfinite(bias) or not np.isfinite(growth_rate):
        raise ValueError("bias must be positive and parameters finite")
    p = np.asarray(pk_matter, dtype=float)
    if np.any(p < 0) or np.any(~np.isfinite(p)):
        raise ValueError("pk_matter must be finite and non-negative")
    b, f = float(bias), float(growth_rate)
    return {
        0: (b * b + 2 * b * f / 3 + f * f / 5) * p,
        2: (4 * b * f / 3 + 4 * f * f / 7) * p,
        4: (8 * f * f / 35) * p,
    }


def kaiser_power_mu(
    pk_matter: float | np.ndarray,
    mu: float | np.ndarray,
    bias: float,
    growth_rate: float,
) -> np.ndarray:
    """Evaluate P_s(k,mu)=(b+f mu²)² P_m(k) under linear Kaiser assumptions."""
    mm = np.asarray(mu, dtype=float)
    p = np.asarray(pk_matter, dtype=float)
    if np.any(np.abs(mm) > 1) or np.any(~np.isfinite(mm)):
        raise ValueError("mu must be finite and lie in [-1, 1]")
    if bias <= 0 or not np.isfinite(bias) or not np.isfinite(growth_rate):
        raise ValueError("bias must be positive and growth_rate finite")
    if np.any(p < 0) or np.any(~np.isfinite(p)):
        raise ValueError("pk_matter must be finite and non-negative")
    return (bias + growth_rate * mm**2) ** 2 * p


def gaussian_fog_damping(
    k: float | np.ndarray, mu: float | np.ndarray, sigma_v_mpc: float
) -> np.ndarray:
    """Phenomenological Gaussian small-scale LOS damping exp[-(k mu sigma_v)^2].

    ``sigma_v_mpc`` is an effective displacement scale in Mpc, not a velocity dispersion. The
    function is a nuisance template only; calibrate its convention against the analysis model.
    """
    kk, mm = np.asarray(k, float), np.asarray(mu, float)
    if sigma_v_mpc < 0 or np.any(kk < 0) or np.any(np.abs(mm) > 1):
        raise ValueError("k>=0, |mu|<=1 and sigma_v_mpc>=0 required")
    return np.exp(-((kk * mm * sigma_v_mpc) ** 2))


__all__ = ["kaiser_power_multipoles", "kaiser_power_mu", "gaussian_fog_damping"]
