"""Weak-lensing intrinsic alignment, shape-noise and Gaussian bandpower covariance utilities.

The NLA kernel and covariance formulas in this module are controlled analysis templates. Survey-grade
cosmic-shear inference additionally needs realistic redshift distributions, shear calibration, masks,
reduced-shear corrections, baryonic uncertainty, source clustering where relevant, and a covariance
validated for the survey estimator and multipole binning.
"""

from __future__ import annotations

import math

import numpy as np

from nufield.cosmology.physics.angular_spectra import limber_cross_spectrum
from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.lensing import normalize_n_of_z

# Common NLA normalization C1*rho_crit0 in Mpc-based analyses. Because conventions vary, it is
# explicitly configurable and should be pinned to the choice in the target likelihood.
DEFAULT_C1_RHOCRIT_DIMENSIONLESS = 0.0134


def nonlinear_alignment_kernel_mpc_inv(
    background: BackgroundCosmology,
    z_lens: np.ndarray,
    source_nz: np.ndarray,
    source_redshift_grid: np.ndarray,
    growth_factor,
    *,
    amplitude: float = 1.0,
    eta: float = 0.0,
    pivot_redshift: float = 0.62,
    c1_rho_crit: float = DEFAULT_C1_RHOCRIT_DIMENSIONLESS,
    sign_convention: str = "standard_negative",
) -> np.ndarray:
    r"""Return a linear-alignment/NLA density kernel W_I(z) in Mpc⁻¹.

    ``source_nz`` is normalized per dz. Convert it to a distribution per comoving distance with
    n_chi = n_z H(z)/c, then multiply by an explicitly configured dimensionless IA amplitude
    F_IA(z) = -A_IA C1ρcrit Ω_m (1+z)^η / [D(z)(1+z_pivot)^η]. The default sign is the common
    convention in which positive A_IA yields a negative intrinsic-alignment kernel. Since conventions
    for n(z), growth normalization, and IA sign differ between codes, record the parameters exactly.
    """
    z = np.asarray(z_lens, float)
    zs = np.asarray(source_redshift_grid, float)
    nz = normalize_n_of_z(zs, source_nz)
    if z.ndim != 1 or len(z) < 3 or np.any(np.diff(z) <= 0) or np.any(z < 0):
        raise ValueError("z_lens must be increasing, non-negative, with at least three samples")
    if z[0] < zs[0] or z[-1] > zs[-1]:
        raise ValueError("lensing redshift grid must lie inside source distribution support")
    if (
        not math.isfinite(amplitude + eta + pivot_redshift + c1_rho_crit)
        or pivot_redshift <= -1
        or c1_rho_crit < 0
    ):
        raise ValueError("IA model parameters must be finite and physically valid")
    if sign_convention not in {"standard_negative", "positive"}:
        raise ValueError("sign_convention must be standard_negative or positive")
    nzi = np.interp(z, zs, nz, left=0.0, right=0.0)
    h = np.asarray(background.H(z), float)
    d = np.asarray(growth_factor(z), float)
    if d.shape != z.shape or np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("growth_factor(z) must return finite positive values matching z")
    sign = -1.0 if sign_convention == "standard_negative" else 1.0
    scale = (
        sign
        * amplitude
        * c1_rho_crit
        * background.omega_m0
        * ((1 + z) / (1 + pivot_redshift)) ** eta
        / d
    )
    from nufield.cosmology.physics.constants import C_KM_S

    return nzi * h / C_KM_S * scale


def tomographic_shear_spectra_with_ia(
    ell: np.ndarray,
    background: BackgroundCosmology,
    z_grid: np.ndarray,
    lensing_kernels_mpc_inv: np.ndarray,
    ia_kernels_mpc_inv: np.ndarray,
    matter_power,
) -> dict[str, np.ndarray]:
    r"""Return GG, GI+IG, II and total E-mode C_ell matrices under Limber projection.

    Kernels are arranged as (n_bin,n_z) and each pair is evaluated with the same matter-power callback.
    The returned ``GI`` matrix contains both G_i I_j and I_i G_j, not just one ordered contribution.
    For real kernels, the final matrix is symmetrized numerically. A survey likelihood may additionally
    require B modes, redshift-dependent nuisance amplitudes, and an angular mask coupling matrix.
    """
    ll, z, wg, wi = (
        np.asarray(ell, float),
        np.asarray(z_grid, float),
        np.asarray(lensing_kernels_mpc_inv, float),
        np.asarray(ia_kernels_mpc_inv, float),
    )
    if ll.ndim != 1 or ll.size == 0 or np.any(ll < 2) or np.any(np.diff(ll) <= 0):
        raise ValueError("ell must be a strictly increasing vector with values >=2")
    if z.ndim != 1 or z.size < 8 or np.any(np.diff(z) <= 0) or np.any(z < 0):
        raise ValueError("z_grid must be increasing, non-negative and have >=8 samples")
    if (
        wg.ndim != 2
        or wi.shape != wg.shape
        or wg.shape[1] != z.size
        or np.any(~np.isfinite(wg + wi))
    ):
        raise ValueError("lensing and IA kernels must be finite matching arrays shaped (n_bin,n_z)")
    nb = wg.shape[0]
    gg = np.zeros((ll.size, nb, nb))
    gi = np.zeros_like(gg)
    ii = np.zeros_like(gg)
    for i in range(nb):
        for j in range(i, nb):
            gg_ij = limber_cross_spectrum(ll, background, z, wg[i], wg[j], matter_power)
            gi_ij = limber_cross_spectrum(ll, background, z, wg[i], wi[j], matter_power)
            ig_ij = limber_cross_spectrum(ll, background, z, wi[i], wg[j], matter_power)
            ii_ij = limber_cross_spectrum(ll, background, z, wi[i], wi[j], matter_power)
            gg[:, i, j] = gg[:, j, i] = gg_ij
            ii[:, i, j] = ii[:, j, i] = ii_ij
            gi[:, i, j] = gi_ij + ig_ij
            gi[:, j, i] = gi[:, i, j]
    total = gg + gi + ii
    return {
        "GG": gg,
        "GI_plus_IG": gi,
        "II": ii,
        "total_signal": total,
        "metadata": np.asarray(
            [
                "Limber projection; NLA kernels supplied by caller",
                "No survey mask, reduced shear, or baryonic correction included",
            ],
            dtype=str,
        ),
    }


def shape_noise_matrix(
    n_eff_arcmin2: np.ndarray,
    sigma_e_per_component: float | np.ndarray,
    *,
    cross_bin_correlations: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return white tomographic shape-noise matrix N_ell in dimensionless shear units.

    For independent bins, N_ell^{ij}=δ_ij σ_e,i²/(2 n_eff,i), with n_eff converted from arcmin⁻² to
    sr⁻¹. An optional correlation matrix allows explicitly modeled cross-bin correlations; it must be
    positive semidefinite. Nonzero off-diagonal noise is not inferred automatically.
    """
    n = np.asarray(n_eff_arcmin2, float)
    sig = np.asarray(sigma_e_per_component, float)
    if n.ndim != 1 or n.size < 1 or np.any(n <= 0) or np.any(~np.isfinite(n)):
        raise ValueError("n_eff_arcmin2 must be finite positive 1-D")
    if sig.ndim == 0:
        sig = np.full(n.shape, float(sig))
    if sig.shape != n.shape or np.any(sig < 0) or np.any(~np.isfinite(sig)):
        raise ValueError("sigma_e_per_component must be non-negative and match n_eff")
    arcmin2_per_sr = (180.0 * 60.0 / math.pi) ** 2
    diagonal = sig**2 / (2 * n * arcmin2_per_sr)
    corr = (
        np.eye(n.size)
        if cross_bin_correlations is None
        else np.asarray(cross_bin_correlations, float)
    )
    if (
        corr.shape != (n.size, n.size)
        or np.any(~np.isfinite(corr))
        or not np.allclose(corr, corr.T, atol=1e-10)
    ):
        raise ValueError("cross_bin_correlations must be finite symmetric and match bin count")
    if np.max(np.abs(np.diag(corr) - 1)) > 1e-8:
        raise ValueError("cross-bin noise correlation matrix must have unit diagonal")
    if np.linalg.eigvalsh(corr).min() < -1e-10:
        raise ValueError("cross-bin noise correlation matrix must be positive semidefinite")
    scale = np.sqrt(diagonal)
    return corr * np.outer(scale, scale)


def gaussian_tomographic_cl_covariance(
    ell_centers: np.ndarray,
    delta_ell: np.ndarray | float,
    signal_cl: np.ndarray,
    noise_matrix: np.ndarray,
    *,
    f_sky: float,
    pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    r"""Construct per-ell Gaussian covariance for tomographic C_ell bandpowers.

    The covariance is block diagonal in ell bins under the Gaussian full-sky approximation, multiplied
    by ``1/[(2ell+1) Δell f_sky]``. For field-pair estimators (ij),(mn),

    Cov(C_ell^ij,C_ell^mn) = [(C_tot^im C_tot^jn)+(C_tot^in C_tot^jm)] / [(2ell+1)Δell f_sky].

    The output ordering is ell-major, then ``pairs`` order. This ignores non-Gaussian lensing covariance,
    super-sample covariance and mask coupling; those must be added through separate terms.
    """
    ell = np.asarray(ell_centers, float)
    sig = np.asarray(signal_cl, float)
    noise = np.asarray(noise_matrix, float)
    if ell.ndim != 1 or ell.size < 1 or np.any(ell < 2) or np.any(np.diff(ell) <= 0):
        raise ValueError("ell_centers must be increasing and >=2")
    if (
        sig.ndim != 3
        or sig.shape[0] != ell.size
        or sig.shape[1] != sig.shape[2]
        or np.any(~np.isfinite(sig))
    ):
        raise ValueError("signal_cl must have shape (n_ell,n_bin,n_bin)")
    nb = sig.shape[1]
    if not np.allclose(sig, np.swapaxes(sig, 1, 2), atol=1e-8):
        raise ValueError("signal C_ell matrices must be symmetric")
    if (
        noise.shape != (nb, nb)
        or np.any(~np.isfinite(noise))
        or not np.allclose(noise, noise.T, atol=1e-10)
    ):
        raise ValueError("noise matrix must be finite symmetric and match source-bin count")
    de = (
        np.full(ell.shape, float(delta_ell))
        if np.asarray(delta_ell).ndim == 0
        else np.asarray(delta_ell, float)
    )
    if de.shape != ell.shape or np.any(de <= 0) or np.any(~np.isfinite(de)):
        raise ValueError("delta_ell must be positive and match ell centers")
    if not 0 < f_sky <= 1:
        raise ValueError("f_sky must lie in (0,1]")
    pair_tuple = tuple(pairs or [(i, j) for i in range(nb) for j in range(i, nb)])
    if not pair_tuple or any(i < 0 or j < i or j >= nb for i, j in pair_tuple):
        raise ValueError("pairs must be valid ordered source-bin pairs i<=j")
    for pair in pair_tuple:
        if pair[0] >= nb or pair[1] >= nb:
            raise ValueError("pair index exceeds tomographic bin count")
    n_pair = len(pair_tuple)
    out = np.zeros((ell.size * n_pair, ell.size * n_pair), float)
    for a, lval in enumerate(ell):
        total = sig[a] + noise
        denom = (2 * lval + 1) * de[a] * f_sky
        block = np.empty((n_pair, n_pair), float)
        for i, (p, q) in enumerate(pair_tuple):
            for j, (r, s) in enumerate(pair_tuple):
                block[i, j] = (total[p, r] * total[q, s] + total[p, s] * total[q, r]) / denom
        block = 0.5 * (block + block.T)
        sl = slice(a * n_pair, (a + 1) * n_pair)
        out[sl, sl] = block
    eig = np.linalg.eigvalsh(out)
    if eig.min() < -1e-8 * max(1.0, abs(float(eig.max()))):
        raise ArithmeticError("Gaussian C_ell covariance is unexpectedly non-positive-semidefinite")
    return out, pair_tuple


__all__ = [
    "DEFAULT_C1_RHOCRIT_DIMENSIONLESS",
    "nonlinear_alignment_kernel_mpc_inv",
    "tomographic_shear_spectra_with_ia",
    "shape_noise_matrix",
    "gaussian_tomographic_cl_covariance",
]
