"""Photometric-redshift response and tomographic-bin migration utilities."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def photoz_pdf(
    z_phot: np.ndarray | float,
    z_true: np.ndarray | float,
    *,
    sigma0: float,
    bias: float = 0.0,
    outlier_fraction: float = 0.0,
    outlier_sigma_factor: float = 3.0,
) -> np.ndarray:
    r"""Evaluate p(z_phot|z_true) for a Gaussian core plus broad Gaussian outlier component.

    Core scatter scales as sigma0*(1+z_true). This is a generic nuisance model, not calibrated for a
    particular photometric survey. It produces a density per unit photometric redshift.
    """
    zp, zt = np.asarray(z_phot, float), np.asarray(z_true, float)
    if sigma0 <= 0 or outlier_sigma_factor < 1 or not 0 <= outlier_fraction < 1:
        raise ValueError("sigma0>0, outlier_sigma_factor>=1 and outlier_fraction in [0,1) required")
    if (
        np.any(zt < 0)
        or np.any(~np.isfinite(zp))
        or np.any(~np.isfinite(zt))
        or not np.isfinite(bias)
    ):
        raise ValueError("redshifts must be finite and z_true non-negative")
    mu = zt + bias * (1 + zt)
    sig = sigma0 * (1 + zt)

    def gaussian(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)

    core = gaussian(zp, mu, sig)
    broad = gaussian(zp, mu, sig * outlier_sigma_factor)
    return (1 - outlier_fraction) * core + outlier_fraction * broad


def photoz_migration_matrix(
    z_true_centers: np.ndarray,
    z_phot_edges: np.ndarray,
    *,
    sigma0: float,
    bias: float = 0.0,
    outlier_fraction: float = 0.0,
    outlier_sigma_factor: float = 3.0,
) -> np.ndarray:
    """Compute Pr(z_phot bin j | z_true center i), matrix shape (n_photo_bins, n_true).

    Column sums may be below one because probability outside the supplied photo-z range is retained
    as selection loss. Do not renormalize unless the analysis explicitly conditions on selected objects.
    """
    zt = np.asarray(z_true_centers, float)
    edges = np.asarray(z_phot_edges, float)
    if zt.ndim != 1 or edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("z_true_centers and increasing z_phot_edges must be 1-D")
    if np.any(zt < 0) or np.any(~np.isfinite(zt)):
        raise ValueError("z_true_centers must be finite and non-negative")
    mu = zt + bias * (1 + zt)
    sig = sigma0 * (1 + zt)
    if sigma0 <= 0 or not 0 <= outlier_fraction < 1 or outlier_sigma_factor < 1:
        raise ValueError("invalid photo-z nuisance parameters")

    def cdf(x: np.ndarray, sd: np.ndarray) -> np.ndarray:
        return ndtr((edges[:, None] - x[None, :]) / sd[None, :])

    core_cdf = cdf(mu, sig)
    broad_cdf = cdf(mu, sig * outlier_sigma_factor)
    prob = (1 - outlier_fraction) * np.diff(core_cdf, axis=0) + outlier_fraction * np.diff(
        broad_cdf, axis=0
    )
    if np.any(prob < -1e-12) or np.any(prob > 1 + 1e-12):
        raise ArithmeticError("migration probabilities outside [0,1]")
    return np.clip(prob, 0, 1)


def observed_bin_distribution(
    true_nz: np.ndarray,
    migration: np.ndarray,
) -> np.ndarray:
    """Map true-bin counts/densities into observed-bin counts with a migration matrix."""
    true = np.asarray(true_nz, float)
    m = np.asarray(migration, float)
    if true.ndim != 1 or m.ndim != 2 or m.shape[1] != true.size:
        raise ValueError("migration columns must match true distribution length")
    if np.any(true < 0) or np.any(m < 0) or np.any(m > 1) or np.any(~np.isfinite(true)):
        raise ValueError(
            "true distribution and migration probabilities must be non-negative and finite"
        )
    return m @ true


__all__ = ["photoz_pdf", "photoz_migration_matrix", "observed_bin_distribution"]
