"""Isotropic two-point correlation from a tabulated matter power spectrum."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


def correlation_function_xi(
    spectrum: MatterPowerSpectrum,
    radius_mpc: np.ndarray,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
) -> np.ndarray:
    r"""Compute xi(r)=1/(2π²) int dk k² P(k) j0(kr) over the chosen supported k interval."""
    r = np.asarray(radius_mpc, float)
    if r.ndim < 1 or np.any(r < 0) or np.any(~np.isfinite(r)):
        raise ValueError("radius_mpc must be finite and non-negative")
    lo = (
        spectrum.k_mpc_inv[0]
        if k_min_mpc_inv is None
        else max(k_min_mpc_inv, spectrum.k_mpc_inv[0])
    )
    hi = (
        spectrum.k_mpc_inv[-1]
        if k_max_mpc_inv is None
        else min(k_max_mpc_inv, spectrum.k_mpc_inv[-1])
    )
    mask = (spectrum.k_mpc_inv >= lo) & (spectrum.k_mpc_inv <= hi)
    if mask.sum() < 3:
        raise ValueError("at least three supported k samples are needed")
    k = spectrum.k_mpc_inv[mask]
    p = spectrum.pk_mpc3[mask]
    flat = r.ravel()
    out = np.empty(flat.shape, float)
    for i, rr in enumerate(flat):
        integrand = k**3 * p * spherical_jn(0, k * rr)  # dlnk measure
        out[i] = simpson(integrand, x=np.log(k)) / (2 * math.pi**2)
    return out.reshape(r.shape)


def correlation_multipoles_from_power(
    k: np.ndarray, pk_multipole: np.ndarray, radius_mpc: np.ndarray, ell: int
) -> np.ndarray:
    r"""Hankel-transform a power-spectrum multipole using spherical Bessel j_ell.

    Convention: xi_l(r)=i^l/(2π²) integral dk k² P_l(k) j_l(kr). For even multipoles the factor i^l
    is real and alternates sign. Input k must be in Mpc^-1 and P_l in Mpc³.
    """
    kk, pp, rr = (
        np.asarray(k, float),
        np.asarray(pk_multipole, float),
        np.asarray(radius_mpc, float),
    )
    if kk.ndim != 1 or pp.shape != kk.shape or len(kk) < 3 or np.any(np.diff(kk) <= 0):
        raise ValueError("k and P must match increasing 1-D arrays")
    if ell < 0 or ell % 2 != 0:
        raise ValueError("this real-valued helper supports non-negative even multipoles")
    phase = (-1.0) ** (ell // 2)
    out = []
    for rv in rr.ravel():
        out.append(
            phase
            * simpson(kk**3 * pp * spherical_jn(ell, kk * rv), x=np.log(kk))
            / (2 * math.pi**2)
        )
    return np.asarray(out).reshape(rr.shape)


__all__ = ["correlation_function_xi", "correlation_multipoles_from_power"]
