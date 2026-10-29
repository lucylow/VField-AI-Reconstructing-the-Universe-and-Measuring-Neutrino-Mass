"""Anisotropic galaxy power prediction: linear bias, Kaiser RSD and AP remapping."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.bao import bao_dilation_parameters
from nufield.cosmology.physics.distances import transverse_comoving_distance_mpc
from nufield.cosmology.physics.redshift_space import gaussian_fog_damping


def ap_scales_from_backgrounds(
    true_background: BackgroundCosmology,
    fiducial_background: BackgroundCosmology,
    z: float,
    r_drag_true_mpc: float,
    r_drag_fid_mpc: float,
) -> tuple[float, float]:
    """Compute alpha_perp and alpha_parallel under the standard fiducial-coordinate convention."""
    dmt = transverse_comoving_distance_mpc(true_background, z)
    dmf = transverse_comoving_distance_mpc(fiducial_background, z)
    ht = float(true_background.H(z))
    hf = float(fiducial_background.H(z))
    return bao_dilation_parameters(
        D_M_mpc=dmt,
        H_km_s_mpc=ht,
        r_drag_mpc=r_drag_true_mpc,
        D_M_fid_mpc=dmf,
        H_fid_km_s_mpc=hf,
        r_drag_fid_mpc=r_drag_fid_mpc,
    )


def ap_remapped_galaxy_power(
    k_fid_mpc_inv: float | np.ndarray,
    mu_fid: float | np.ndarray,
    matter_power,
    *,
    bias: float,
    growth_rate: float,
    alpha_perp: float,
    alpha_parallel: float,
    sigma_fog_mpc: float = 0.0,
    shot_noise_mpc3: float = 0.0,
) -> np.ndarray:
    r"""Evaluate AP-remapped P_g(k,μ) with linear Kaiser RSD and optional Gaussian damping.

    The Jacobian 1/(alpha_perp² alpha_parallel) converts a spectrum predicted in true coordinates to
    fiducial observed coordinates. ``matter_power(k_true)`` must return P_m in Mpc³ at the requested
    physical inverse-Mpc wavenumber. This is a controlled baseline, not a full DESI model.
    """
    kobs, muobs = np.asarray(k_fid_mpc_inv, float), np.asarray(mu_fid, float)
    if np.any(kobs <= 0) or np.any(np.abs(muobs) > 1) or np.any(~np.isfinite(kobs + muobs)):
        raise ValueError("k must be positive and mu in [-1,1]")
    if (
        bias <= 0
        or alpha_perp <= 0
        or alpha_parallel <= 0
        or sigma_fog_mpc < 0
        or shot_noise_mpc3 < 0
    ):
        raise ValueError("bias/alpha positive and damping/noise non-negative required")
    kperp2_true = kobs**2 * (1 - muobs**2) / alpha_perp**2
    kpar2_true = kobs**2 * muobs**2 / alpha_parallel**2
    ktrue = np.sqrt(kperp2_true + kpar2_true)
    mutrue = np.divide(np.sqrt(kpar2_true), ktrue, out=np.zeros_like(ktrue), where=ktrue > 0)
    pm = np.asarray(matter_power(ktrue), float)
    if np.any(pm < 0) or np.any(~np.isfinite(pm)):
        raise ValueError("matter_power callback returned invalid power")
    ptrue = (bias + growth_rate * mutrue**2) ** 2 * pm
    ptrue *= gaussian_fog_damping(ktrue, mutrue, sigma_fog_mpc)
    return ptrue / (alpha_perp**2 * alpha_parallel) + shot_noise_mpc3


def power_multipoles(
    k_fid_mpc_inv: np.ndarray,
    model,
    *,
    multipoles: tuple[int, ...] = (0, 2, 4),
    n_mu: int = 64,
) -> dict[int, np.ndarray]:
    """Integrate P_l(k)=(2l+1)/2 int_{-1}^1 dmu P(k,mu)L_l(mu) by Gauss-Legendre quadrature."""
    k = np.asarray(k_fid_mpc_inv, float)
    if k.ndim != 1 or np.any(k <= 0) or np.any(np.diff(k) <= 0) or n_mu < 8:
        raise ValueError("k must be positive increasing and n_mu>=8")
    if any(l < 0 or l % 2 for l in multipoles):
        raise ValueError("only nonnegative even multipoles are supported")
    mu, weights = leggauss(n_mu)
    out = {ell: np.zeros_like(k) for ell in multipoles}
    from numpy.polynomial.legendre import legval

    for ik, kval in enumerate(k):
        p_mu = np.asarray(model(float(kval), mu), float)
        if p_mu.shape != mu.shape or np.any(~np.isfinite(p_mu)):
            raise ValueError("model(k,mu) must return finite values matching mu")
        for ell in multipoles:
            coeff = np.zeros(ell + 1)
            coeff[ell] = 1.0
            legendre = legval(mu, coeff)
            out[ell][ik] = (2 * ell + 1) / 2 * np.sum(weights * p_mu * legendre)
    return out


__all__ = ["ap_scales_from_backgrounds", "ap_remapped_galaxy_power", "power_multipoles"]
