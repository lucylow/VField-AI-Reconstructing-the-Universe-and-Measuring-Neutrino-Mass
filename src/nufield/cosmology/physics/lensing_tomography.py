"""Tomographic weak-lensing source-bin kernels and Limber C_ell matrices."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from nufield.cosmology.physics.angular_spectra import limber_cross_spectrum
from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.lensing import (
    convergence_kernel_mpc_inv,
    lensing_efficiency,
    normalize_n_of_z,
)


def source_bin_distributions(
    z: np.ndarray, n_of_z: np.ndarray, bin_edges: np.ndarray
) -> np.ndarray:
    """Return normalized n_i(z) for top-hat selections, shape (n_bin,n_z).

    Objects outside the supplied redshift range are not redistributed into bins. Empty bins raise an
    error rather than silently generating an all-zero lensing kernel.
    """
    zz, nz, edges = np.asarray(z, float), np.asarray(n_of_z, float), np.asarray(bin_edges, float)
    if zz.ndim != 1 or nz.shape != zz.shape or np.any(np.diff(zz) <= 0):
        raise ValueError("z and n_of_z must be matching increasing 1-D arrays")
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be a strictly increasing vector")
    base = normalize_n_of_z(zz, nz)
    bins = []
    for i in range(len(edges) - 1):
        mask = (zz >= edges[i]) & (zz < edges[i + 1] if i < len(edges) - 2 else zz <= edges[i + 1])
        selected = np.where(mask, base, 0.0)
        if np.count_nonzero(mask) < 3 or trapezoid(selected, zz) <= 0:
            raise ValueError(f"tomographic bin {i} has insufficient or zero source support")
        # Smooth tapering/windowing belongs to the survey definition; this is a literal top-hat bin.
        bins.append(normalize_n_of_z(zz, selected))
    return np.vstack(bins)


def tomographic_lensing_kernels(
    background: BackgroundCosmology,
    z: np.ndarray,
    source_distributions: np.ndarray,
) -> np.ndarray:
    """Return Wκ_i(z) kernels with shape (n_bin,n_z), units Mpc^-1."""
    zz, nzs = np.asarray(z, float), np.asarray(source_distributions, float)
    if nzs.ndim != 2 or nzs.shape[1] != zz.size:
        raise ValueError("source_distributions must have shape (n_bin,n_z)")
    kernels = []
    for nz in nzs:
        g = lensing_efficiency(background, zz, zz, nz)
        kernels.append(convergence_kernel_mpc_inv(background, zz, g))
    return np.vstack(kernels)


def tomographic_cl_matrix(
    ell: np.ndarray,
    background: BackgroundCosmology,
    z: np.ndarray,
    kernels_mpc_inv: np.ndarray,
    matter_power,
) -> np.ndarray:
    """Return C_ell^{ij}, shape (n_ell,n_bin,n_bin), via Limber cross-spectrum projections."""
    ll, zz, kernels = (
        np.asarray(ell, float),
        np.asarray(z, float),
        np.asarray(kernels_mpc_inv, float),
    )
    if kernels.ndim != 2 or kernels.shape[1] != zz.size:
        raise ValueError("kernels_mpc_inv must have shape (n_bin,n_z)")
    nbin = kernels.shape[0]
    result = np.zeros((ll.size, nbin, nbin), dtype=float)
    for i in range(nbin):
        for j in range(i, nbin):
            cl = limber_cross_spectrum(ll, background, zz, kernels[i], kernels[j], matter_power)
            result[:, i, j] = cl
            result[:, j, i] = cl
    return result


__all__ = ["source_bin_distributions", "tomographic_lensing_kernels", "tomographic_cl_matrix"]
