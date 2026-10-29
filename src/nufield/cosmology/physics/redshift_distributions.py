"""Survey redshift distributions, tomographic selections and photo-z migration matrices.

The routines operate on declared arrays rather than assuming a particular survey n(z). A realistic
analysis must ingest the survey's calibrated redshift distribution and its selection/completeness model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import ndtr


@dataclass(frozen=True, slots=True)
class RedshiftDistribution:
    redshift: np.ndarray
    density_per_redshift: np.ndarray
    label: str
    normalized: bool
    provenance: str


def validate_redshift_distribution(
    z: np.ndarray,
    n_of_z: np.ndarray,
    *,
    normalize: bool = False,
    label: str = "unspecified",
    provenance: str = "user supplied",
) -> RedshiftDistribution:
    """Validate and optionally normalize n(z), where integral n(z) dz is dimensionless."""
    zz, nz = np.asarray(z, float), np.asarray(n_of_z, float)
    if zz.ndim != 1 or nz.shape != zz.shape or zz.size < 3:
        raise ValueError("z and n_of_z must be matching vectors with at least three samples")
    if np.any(~np.isfinite(zz)) or np.any(~np.isfinite(nz)) or np.any(zz < 0) or np.any(nz < 0):
        raise ValueError("redshift and n(z) must be finite and non-negative")
    if np.any(np.diff(zz) <= 0):
        raise ValueError("redshift grid must be strictly increasing")
    norm = float(trapezoid(nz, zz))
    if norm <= 0:
        raise ValueError("n(z) has zero integrated mass")
    arr = nz / norm if normalize else nz.copy()
    return RedshiftDistribution(zz.copy(), arr, str(label), bool(normalize), str(provenance))


def gaussian_photoz_bin_probability(
    z_true: np.ndarray | float,
    z_photo_low: float,
    z_photo_high: float,
    *,
    sigma0: float,
    bias: float = 0.0,
) -> np.ndarray:
    r"""Probability that a true-z object enters a photo-z bin under Gaussian photo-z errors.

    Assumes z_phot ~ Normal(z_true + bias, sigma0*(1+z_true)). Catastrophic outliers and asymmetric
    tails are not represented; use a calibrated response kernel for precision survey work.
    """
    zt = np.asarray(z_true, float)
    if np.any(~np.isfinite(zt)) or np.any(zt < 0) or not (0 <= z_photo_low < z_photo_high):
        raise ValueError("true z must be finite/nonnegative and photo-z bin bounds ordered")
    if not np.isfinite(sigma0) or sigma0 < 0 or not np.isfinite(bias):
        raise ValueError("sigma0 must be finite/non-negative and bias finite")
    if sigma0 == 0:
        mean = zt + bias
        return ((mean >= z_photo_low) & (mean < z_photo_high)).astype(float)
    sigma = sigma0 * (1.0 + zt)
    mean = zt + bias
    return ndtr((z_photo_high - mean) / sigma) - ndtr((z_photo_low - mean) / sigma)


def photoz_migration_matrix(
    z_true_grid: np.ndarray,
    photoz_edges: np.ndarray,
    *,
    sigma0: float,
    bias: float = 0.0,
    include_under_overflow: bool = False,
) -> np.ndarray:
    """Return M[photo_bin,true_z] using probabilities averaged over a finite true-z cell.

    Without overflow bins, the column sum may be less than one because objects can scatter outside the
    chosen photo-z domain. The lost probability is deliberately not renormalized away.
    """
    zt, edges = np.asarray(z_true_grid, float), np.asarray(photoz_edges, float)
    if zt.ndim != 1 or zt.size < 2 or np.any(np.diff(zt) <= 0) or np.any(zt < 0):
        raise ValueError(
            "z_true_grid must be increasing, non-negative and contain at least two samples"
        )
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0) or np.any(edges < 0):
        raise ValueError("photoz_edges must be increasing and non-negative")
    mat = np.vstack(
        [
            gaussian_photoz_bin_probability(zt, lo, hi, sigma0=sigma0, bias=bias)
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    if include_under_overflow:
        mean = zt + bias
        if sigma0 == 0:
            under = (mean < edges[0]).astype(float)
        else:
            sigma = sigma0 * (1.0 + zt)
            # Include all photo-z values below the first edge, even negative values.
            under = ndtr((edges[0] - mean) / sigma)
        over = np.maximum(0.0, 1.0 - mat.sum(axis=0) - under)
        mat = np.vstack((under, mat, over))
    if np.any(mat < -1e-14) or np.any(mat > 1 + 1e-12):
        raise ArithmeticError("photo-z probabilities fall outside [0,1]")
    return np.clip(mat, 0.0, 1.0)


def migrate_redshift_distribution(
    z_true: np.ndarray,
    n_true: np.ndarray,
    photoz_edges: np.ndarray,
    *,
    sigma0: float,
    bias: float = 0.0,
) -> np.ndarray:
    """Integrate a tabulated true n(z) into photo-z bins using the Gaussian response model."""
    zz, nz = np.asarray(z_true, float), np.asarray(n_true, float)
    if zz.shape != nz.shape or zz.ndim != 1 or zz.size < 3 or np.any(nz < 0):
        raise ValueError("z_true and n_true must be matching non-negative vectors")
    if np.any(np.diff(zz) <= 0):
        raise ValueError("z_true must be strictly increasing")
    edges = np.asarray(photoz_edges, float)
    probs = photoz_migration_matrix(zz, edges, sigma0=sigma0, bias=bias)
    # Integrate each bin's response against the continuous input distribution.
    counts = np.array([trapezoid(row * nz, zz) for row in probs])
    return counts


def effective_redshift(z: np.ndarray, weight: np.ndarray) -> float:
    """Compute weighted effective redshift integral(z w)/integral(w)."""
    zz, ww = np.asarray(z, float), np.asarray(weight, float)
    if zz.ndim != 1 or ww.shape != zz.shape or zz.size < 2 or np.any(np.diff(zz) <= 0):
        raise ValueError("z/weight must be matching vectors on an increasing grid")
    if np.any(~np.isfinite(zz)) or np.any(~np.isfinite(ww)) or np.any(zz < 0) or np.any(ww < 0):
        raise ValueError("z and non-negative weights must be finite")
    denominator = float(trapezoid(ww, zz))
    if denominator <= 0:
        raise ValueError("weight has zero integral")
    return float(trapezoid(zz * ww, zz) / denominator)


def tomographic_bin_mask(
    z: np.ndarray, lower: float, upper: float, *, include_upper: bool = False
) -> np.ndarray:
    """Return deterministic true-redshift membership in [lower,upper) (upper inclusive if requested)."""
    zz = np.asarray(z, float)
    if lower < 0 or upper <= lower or np.any(~np.isfinite(zz)):
        raise ValueError("invalid redshift values or bin bounds")
    return (zz >= lower) & ((zz <= upper) if include_upper else (zz < upper))


__all__ = [
    "RedshiftDistribution",
    "validate_redshift_distribution",
    "gaussian_photoz_bin_probability",
    "photoz_migration_matrix",
    "migrate_redshift_distribution",
    "effective_redshift",
    "tomographic_bin_mask",
]
