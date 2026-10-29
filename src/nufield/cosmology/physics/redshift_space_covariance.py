"""Gaussian covariance approximations for redshift-space power-spectrum multipoles.

This implements the idealized periodic-volume Gaussian approximation for multipoles of the Kaiser
power spectrum. It omits non-Gaussian trispectrum terms, survey-window mixing, super-sample covariance,
wide-angle effects, and reconstruction effects; it is a controlled baseline, not a DESI covariance.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval
from scipy.integrate import trapezoid


def kaiser_power_mu(
    k: np.ndarray | float,
    mu: np.ndarray | float,
    pk_linear: np.ndarray | float,
    bias: float,
    growth_rate: float,
    *,
    shot_noise: float = 0.0,
    sigma_v_km_s: float = 0.0,
    H_km_s_Mpc: float = 100.0,
    redshift: float = 0.0,
) -> np.ndarray:
    """Anisotropic galaxy power with linear Kaiser RSD and a Gaussian FoG factor.

    The damping is ``exp[-(k mu sigma_v/(a H))²]`` with physical k in Mpc^-1 and sigma_v in km/s.
    This is only a phenomenological damping prescription. The shot-noise term is added after damping.
    """
    kk, mm, pp = np.broadcast_arrays(
        np.asarray(k, float), np.asarray(mu, float), np.asarray(pk_linear, float)
    )
    if (
        np.any(~np.isfinite(kk))
        or np.any(kk < 0)
        or np.any(~np.isfinite(mm))
        or np.any(np.abs(mm) > 1)
    ):
        raise ValueError("k must be finite non-negative and mu in [-1,1]")
    if np.any(~np.isfinite(pp)) or np.any(pp < 0) or bias <= 0 or growth_rate < 0:
        raise ValueError("power non-negative, bias positive and growth rate non-negative required")
    if shot_noise < 0 or sigma_v_km_s < 0 or H_km_s_Mpc <= 0 or redshift < 0:
        raise ValueError("invalid noise, velocity dispersion, Hubble rate or redshift")
    a = 1.0 / (1.0 + redshift)
    damp = np.exp(-((kk * mm * sigma_v_km_s / (a * H_km_s_Mpc)) ** 2))
    return (bias + growth_rate * mm**2) ** 2 * pp * damp + shot_noise


def kaiser_multipoles_analytic(
    pk: np.ndarray, bias: float, growth_rate: float
) -> dict[int, np.ndarray]:
    """Exact linear Kaiser monopole, quadrupole and hexadecapole without FoG damping."""
    p = np.asarray(pk, float)
    if np.any(~np.isfinite(p)) or np.any(p < 0) or bias <= 0 or growth_rate < 0:
        raise ValueError("invalid power, bias or growth rate")
    b, f = float(bias), float(growth_rate)
    return {
        0: (b * b + 2 * b * f / 3 + f * f / 5) * p,
        2: (4 * b * f / 3 + 4 * f * f / 7) * p,
        4: (8 * f * f / 35) * p,
    }


def multipole_from_mu_samples(
    pk_mu: np.ndarray,
    mu: np.ndarray,
    *,
    ells: tuple[int, ...] = (0, 2, 4),
) -> dict[int, np.ndarray]:
    """Integrate sampled P(k,mu) against Legendre multipoles with 0.5(2l+1) normalization."""
    values, m = np.asarray(pk_mu, float), np.asarray(mu, float)
    if values.ndim < 1 or m.ndim != 1 or values.shape[-1] != m.size or m.size < 3:
        raise ValueError("pk_mu last dimension must match mu samples (>=3)")
    if (
        np.any(~np.isfinite(values))
        or np.any(~np.isfinite(m))
        or np.any(np.abs(m) > 1)
        or np.any(np.diff(m) <= 0)
    ):
        raise ValueError("finite values and strictly increasing mu within [-1,1] required")
    result = {}
    for ell in ells:
        if ell < 0:
            raise ValueError("multipole ell must be non-negative")
        coeff = np.zeros(ell + 1)
        coeff[ell] = 1.0
        leg = legval(m, coeff)
        result[int(ell)] = (2 * ell + 1) / 2 * trapezoid(values * leg, x=m, axis=-1)
    return result


def gaussian_multipole_covariance(
    k_centers: np.ndarray,
    mode_counts: np.ndarray,
    pk_linear: np.ndarray,
    bias: float,
    growth_rate: float,
    *,
    ells: tuple[int, ...] = (0, 2, 4),
    mu_order: int = 256,
    shot_noise: float = 0.0,
    sigma_v_km_s: float = 0.0,
    H_km_s_Mpc: float = 100.0,
    redshift: float = 0.0,
) -> np.ndarray:
    """Return block covariance Cov[P_l(k_i),P_l'(k_j)] in independent-bin Gaussian approximation.

    Different k bins are treated as independent. Within a k bin, covariance is the angular integral
    ``2(2l+1)(2l'+1)/Nmodes * ∫dmu/2 P(k,mu)^2 L_l L_l'``. The exact prefactor depends on the
    full Fourier estimator convention; this implementation follows a full mu∈[-1,1] multipole basis.
    Validate the convention against a known isotropic limit before using for forecasts.
    """
    k, n, p = map(lambda x: np.asarray(x, float), (k_centers, mode_counts, pk_linear))
    if k.ndim != 1 or n.shape != k.shape or p.shape != k.shape or k.size < 1:
        raise ValueError("k, mode_counts, and pk must be matching 1-D vectors")
    if np.any(~np.isfinite(k + n + p)) or np.any(k <= 0) or np.any(n <= 0) or np.any(p < 0):
        raise ValueError("k positive, mode counts positive, and power non-negative required")
    if mu_order < 32:
        raise ValueError("mu_order must be at least 32")
    if not ells or any(ell < 0 for ell in ells) or len(set(ells)) != len(ells):
        raise ValueError("ells must be unique non-negative integers")
    mu, weights = np.polynomial.legendre.leggauss(mu_order)
    block_size = k.size
    out = np.zeros((len(ells) * block_size, len(ells) * block_size), float)
    for i, (ki, ni, pi) in enumerate(zip(k, n, p)):
        p_mu = kaiser_power_mu(
            ki,
            mu,
            pi,
            bias,
            growth_rate,
            shot_noise=shot_noise,
            sigma_v_km_s=sigma_v_km_s,
            H_km_s_Mpc=H_km_s_Mpc,
            redshift=redshift,
        )
        for a, ell in enumerate(ells):
            cl = np.zeros(ell + 1)
            cl[ell] = 1.0
            la = legval(mu, cl)
            for b, ellp in enumerate(ells):
                cp = np.zeros(ellp + 1)
                cp[ellp] = 1.0
                lb = legval(mu, cp)
                # gauss weights sum to two; the factor 1/2 averages over mu.
                cov = (
                    2.0
                    * (2 * ell + 1)
                    * (2 * ellp + 1)
                    / ni
                    * 0.5
                    * np.sum(weights * p_mu**2 * la * lb)
                )
                out[a * block_size + i, b * block_size + i] = cov
    # Numerical angular quadrature should give a symmetric covariance; symmetrize tiny roundoff.
    return 0.5 * (out + out.T)


def covariance_finite_mode_report(covariance: np.ndarray) -> dict[str, float | bool]:
    """Inspect symmetry, positive definiteness and eigenvalue condition number."""
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or np.any(~np.isfinite(c)):
        raise ValueError("covariance must be a finite square matrix")
    sym = float(np.max(np.abs(c - c.T)))
    eig = np.linalg.eigvalsh(0.5 * (c + c.T))
    positive = bool(eig.min() > 0)
    return {
        "max_asymmetry": sym,
        "min_eigenvalue": float(eig.min()),
        "max_eigenvalue": float(eig.max()),
        "positive_definite": positive,
        "condition_number": float(eig.max() / eig.min()) if positive else float("inf"),
    }


__all__ = [
    "kaiser_power_mu",
    "kaiser_multipoles_analytic",
    "multipole_from_mu_samples",
    "gaussian_multipole_covariance",
    "covariance_finite_mode_report",
]
