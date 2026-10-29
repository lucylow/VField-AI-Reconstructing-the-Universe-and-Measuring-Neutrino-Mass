"""Weak-lensing convergence kernels and Limber-projected angular spectra.

The functions use physical Mpc, k in Mpc^-1 and P in Mpc^3. They implement a flat-sky/standard Limber
projection for scalar density fluctuations and do not include reduced shear, intrinsic alignments,
source clustering, magnification, mask mode coupling, nonlinear baryonic feedback, or full-sky low-ell
corrections. Those effects need to be modeled or marginalized for survey inference.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.constants import C_KM_S
from nufield.cosmology.physics.distances import radial_comoving_distance_mpc


def normalize_n_of_z(z: np.ndarray, n_of_z: np.ndarray) -> np.ndarray:
    """Normalize nonnegative n(z) so integral dz n(z)=1."""
    zz, nz = np.asarray(z, dtype=float), np.asarray(n_of_z, dtype=float)
    if zz.ndim != 1 or nz.shape != zz.shape or len(zz) < 3:
        raise ValueError("z and n_of_z must be same-shape 1-D arrays with at least three points")
    if (
        np.any(~np.isfinite(zz))
        or np.any(~np.isfinite(nz))
        or np.any(zz < 0)
        or np.any(nz < 0)
        or np.any(np.diff(zz) <= 0)
    ):
        raise ValueError("z must increase and both arrays must be finite; n(z)>=0")
    norm = float(simpson(nz, x=zz))
    if norm <= 0:
        raise ValueError("n(z) has zero integral")
    return nz / norm


def lensing_efficiency(
    background: BackgroundCosmology,
    z_lens: np.ndarray,
    z_source: np.ndarray,
    n_source_of_z: np.ndarray,
) -> np.ndarray:
    r"""Compute g(z_l)=int_{z_l}^infty dz_s n(z_s)(chi_s-chi_l)/chi_s.

    Source n(z) is normalized over the provided source grid. The lens redshift grid must be within
    the source grid. In curved cosmologies this simple ratio is replaced by curvature-aware angular-
    diameter distance ratios; this implementation therefore rejects nonzero curvature.
    """
    if abs(background.params.omega_k) > 1e-6:
        raise NotImplementedError(
            "lensing_efficiency currently uses a flat-universe distance ratio"
        )
    zl = np.asarray(z_lens, dtype=float)
    zs = np.asarray(z_source, dtype=float)
    nz = normalize_n_of_z(zs, n_source_of_z)
    if zl.ndim != 1 or np.any(zl < 0) or np.any(zl < zs[0]) or np.any(zl > zs[-1]):
        raise ValueError("z_lens must be 1-D and lie within source redshift range")
    chi_s = np.array([radial_comoving_distance_mpc(background, float(z)) for z in zs])
    chi_l = np.array([radial_comoving_distance_mpc(background, float(z)) for z in zl])
    out = np.zeros_like(zl)
    for i, (zval, chil) in enumerate(zip(zl, chi_l)):
        mask = zs > zval
        if mask.sum() < 2:
            out[i] = 0.0
            continue
        ratio = np.zeros_like(chi_s[mask])
        positive = chi_s[mask] > 0
        ratio[positive] = np.maximum(chi_s[mask][positive] - chil, 0.0) / chi_s[mask][positive]
        out[i] = float(simpson(nz[mask] * ratio, x=zs[mask]))
    return out


def convergence_kernel_mpc_inv(
    background: BackgroundCosmology,
    z_lens: np.ndarray,
    efficiency: np.ndarray,
) -> np.ndarray:
    r"""Flat-universe lensing kernel Wκ(z)=3 Ωm H0²/(2c²) (1+z) χ(z) g(z)."""
    z = np.asarray(z_lens, dtype=float)
    g = np.asarray(efficiency, dtype=float)
    if z.ndim != 1 or g.shape != z.shape or np.any(z < 0) or np.any(g < 0):
        raise ValueError("z_lens and efficiency must be matching 1-D nonnegative arrays")
    chi = np.array([radial_comoving_distance_mpc(background, float(zz)) for zz in z])
    prefactor = 1.5 * background.omega_m0 * (background.params.H0_km_s_Mpc / C_KM_S) ** 2
    return prefactor * (1 + z) * chi * g


def limber_convergence_cl(
    ell: float | np.ndarray,
    background: BackgroundCosmology,
    z_grid: np.ndarray,
    source_nz: np.ndarray,
    matter_power,
    *,
    nonlinear: bool = False,
) -> np.ndarray:
    r"""Compute C_ell^κκ = int dz (H/c)^-1 Wκ²/χ² P((ell+1/2)/χ,z).

    ``matter_power(k_mpc_inv, z)`` must return P in Mpc³ and represent a physically suitable linear or
    nonlinear spectrum. Limber is generally not appropriate for the lowest multipoles; choose a
    redshift range and k-range with explicit support in the supplied backend.
    """
    if abs(background.params.omega_k) > 1e-6:
        raise NotImplementedError("curved-universe lensing distance ratios are not implemented")
    zz = np.asarray(z_grid, dtype=float)
    ellv = np.asarray(ell, dtype=float)
    if zz.ndim != 1 or len(zz) < 8 or np.any(np.diff(zz) <= 0) or np.any(zz < 0):
        raise ValueError("z_grid must be increasing, nonnegative, and contain at least 8 samples")
    if np.any(ellv < 2) or np.any(~np.isfinite(ellv)):
        raise ValueError("ell values must be finite and >=2 for this Limber implementation")
    nz = normalize_n_of_z(zz, source_nz)
    chi = np.array([radial_comoving_distance_mpc(background, float(z)) for z in zz])
    # g(z) integrated directly over the source distribution at the same grid.
    efficiency = lensing_efficiency(background, zz, zz, nz)
    kernel = convergence_kernel_mpc_inv(background, zz, efficiency)
    h_of_z = np.asarray(background.H(zz))
    out_shape = ellv.shape
    flat_ell = ellv.ravel()
    result = np.zeros(flat_ell.size, dtype=float)
    valid = (chi > 0) & (kernel >= 0)
    for j, lval in enumerate(flat_ell):
        kval = (lval + 0.5) / np.maximum(chi, 1e-30)
        pk = np.zeros_like(zz)
        for i in np.flatnonzero(valid):
            pval = float(np.asarray(matter_power(float(kval[i]), float(zz[i]))))
            if not np.isfinite(pval) or pval < 0:
                raise ValueError("matter_power callback returned non-finite or negative power")
            pk[i] = pval
        integrand = C_KM_S / h_of_z * kernel**2 / np.maximum(chi, 1e-30) ** 2 * pk
        integrand[~valid] = 0.0
        result[j] = float(simpson(integrand, x=zz))
    return result.reshape(out_shape)


def shear_shape_noise_cl(sigma_e_per_component: float, n_eff_arcmin2: float) -> float:
    """White shape-noise level N_ell = sigma_e²/(2 n_eff), with n_eff per arcmin²."""
    if sigma_e_per_component < 0 or n_eff_arcmin2 <= 0:
        raise ValueError("sigma_e_per_component must be non-negative and n_eff_arcmin2 positive")
    arcmin2_per_sr = (180.0 * 60.0 / np.pi) ** 2
    density_sr = n_eff_arcmin2 * arcmin2_per_sr
    return float(sigma_e_per_component**2 / (2 * density_sr))


__all__ = [
    "normalize_n_of_z",
    "lensing_efficiency",
    "convergence_kernel_mpc_inv",
    "limber_convergence_cl",
    "shear_shape_noise_cl",
]
