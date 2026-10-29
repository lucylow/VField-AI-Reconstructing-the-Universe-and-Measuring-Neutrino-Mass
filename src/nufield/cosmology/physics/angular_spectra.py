"""Generic Limber projection for line-of-sight cosmological fields."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.constants import C_KM_S
from nufield.cosmology.physics.distances import radial_comoving_distance_mpc


def limber_cross_spectrum(
    ell: float | np.ndarray,
    background: BackgroundCosmology,
    z_grid: np.ndarray,
    weight_a_mpc_inv: np.ndarray,
    weight_b_mpc_inv: np.ndarray,
    matter_power,
    *,
    allow_curvature_approximation: bool = False,
) -> np.ndarray:
    r"""Compute C_ell^{AB}=int dz (c/H) W_A W_B P((ell+1/2)/chi,z)/chi².

    The kernels W_A and W_B must be defined per comoving distance with units Mpc^-1. The supplied
    matter_power callback must return P(k,z) in Mpc³ at physical k in Mpc^-1. This assumes a flat
    geometry unless the caller explicitly accepts the approximation.
    """
    if abs(background.params.omega_k) > 1e-6 and not allow_curvature_approximation:
        raise NotImplementedError("Limber projection currently uses flat transverse distance chi")
    z = np.asarray(z_grid, float)
    wa, wb = np.asarray(weight_a_mpc_inv, float), np.asarray(weight_b_mpc_inv, float)
    l = np.asarray(ell, float)
    if z.ndim != 1 or z.size < 8 or np.any(z < 0) or np.any(np.diff(z) <= 0):
        raise ValueError("z_grid must be increasing, non-negative and contain >=8 samples")
    if wa.shape != z.shape or wb.shape != z.shape or np.any(~np.isfinite(wa + wb)):
        raise ValueError("projection kernels must match z_grid and be finite")
    if np.any(l < 2) or np.any(~np.isfinite(l)):
        raise ValueError("ell must be finite and >=2")
    chi = np.array([radial_comoving_distance_mpc(background, float(zi)) for zi in z])
    h = np.asarray(background.H(z), float)
    valid = chi > 0
    result = np.empty(l.size, float)
    for j, ell_value in enumerate(l.ravel()):
        kvals = (ell_value + 0.5) / np.maximum(chi, 1e-30)
        pk = np.zeros_like(z)
        for i in np.flatnonzero(valid):
            p = float(np.asarray(matter_power(float(kvals[i]), float(z[i]))))
            if not np.isfinite(p) or p < 0:
                raise ValueError("matter_power callback must return finite non-negative P(k,z)")
            pk[i] = p
        integrand = C_KM_S / h * wa * wb * pk / np.maximum(chi, 1e-30) ** 2
        integrand[~valid] = 0
        result[j] = float(simpson(integrand, x=z))
    return result.reshape(l.shape)


def lensing_kernel_from_efficiency(
    background: BackgroundCosmology, z: np.ndarray, efficiency: np.ndarray
) -> np.ndarray:
    """Compute Wκ(z) in Mpc^-1 for an already-calculated source efficiency."""
    from nufield.cosmology.physics.lensing import convergence_kernel_mpc_inv

    return convergence_kernel_mpc_inv(background, z, efficiency)


__all__ = ["limber_cross_spectrum", "lensing_kernel_from_efficiency"]
