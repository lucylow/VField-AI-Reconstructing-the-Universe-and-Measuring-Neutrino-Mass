"""Line-of-sight kernels for galaxy counts and weak-lensing projection exercises."""

from __future__ import annotations

import numpy as np

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.constants import C_KM_S
from nufield.cosmology.physics.lensing import (
    convergence_kernel_mpc_inv,
    lensing_efficiency,
    normalize_n_of_z,
)


def galaxy_counts_kernel_mpc_inv(
    background: BackgroundCosmology,
    z: np.ndarray,
    n_of_z: np.ndarray,
    bias: float | np.ndarray,
) -> np.ndarray:
    r"""Compute W_g(χ)=b(z)n(z)H(z)/c for normalized n(z) per unit redshift.

    This is a simple projected-density kernel; magnification, relativistic number-count terms, RSD
    projection and nonlinear bias are omitted. Curved geometry uses the same radial χ here, so a
    precision curved-universe projection must use a curvature-aware implementation.
    """
    zz, nz = np.asarray(z, float), np.asarray(n_of_z, float)
    b = np.asarray(bias, float)
    if zz.ndim != 1 or nz.shape != zz.shape or len(zz) < 3 or np.any(np.diff(zz) <= 0):
        raise ValueError("z and n_of_z must be matching increasing vectors")
    if np.any(b <= 0) or np.any(~np.isfinite(b)):
        raise ValueError("galaxy bias must be finite and positive")
    if b.ndim and b.shape != zz.shape:
        raise ValueError("array-valued bias must match z shape")
    nz_norm = normalize_n_of_z(zz, nz)
    return b * nz_norm * np.asarray(background.H(zz)) / C_KM_S


def galaxy_lensing_source_kernels(
    background: BackgroundCosmology,
    z: np.ndarray,
    source_nz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (source efficiency g(z), convergence Wκ(z)) for one source sample."""
    zz = np.asarray(z, float)
    nz = normalize_n_of_z(zz, source_nz)
    g = lensing_efficiency(background, zz, zz, nz)
    return g, convergence_kernel_mpc_inv(background, zz, g)


__all__ = ["galaxy_counts_kernel_mpc_inv", "galaxy_lensing_source_kernels"]
