"""Survey selection functions, number densities, and simple effective-volume diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.distances import volume_element_mpc3_sr_dz


def normalize_selection_function(z: np.ndarray, selection: np.ndarray) -> np.ndarray:
    """Normalize a nonnegative selection function with respect to redshift."""
    zz, ss = np.asarray(z, float), np.asarray(selection, float)
    if zz.ndim != 1 or zz.shape != ss.shape or len(zz) < 3:
        raise ValueError("z and selection must be same-shape 1-D arrays with >=3 points")
    if (
        np.any(~np.isfinite(zz))
        or np.any(~np.isfinite(ss))
        or np.any(np.diff(zz) <= 0)
        or np.any(ss < 0)
    ):
        raise ValueError("z must be increasing; selection must be finite and non-negative")
    norm = float(simpson(ss, x=zz))
    if norm <= 0:
        raise ValueError("selection function integrates to zero")
    return ss / norm


def comoving_number_density(
    background: BackgroundCosmology,
    z: np.ndarray,
    dN_dz_domega: np.ndarray,
) -> np.ndarray:
    """Convert counts per redshift per steradian to n_comoving(z) in Mpc^-3."""
    zz, counts = np.asarray(z, float), np.asarray(dN_dz_domega, float)
    if zz.ndim != 1 or zz.shape != counts.shape or np.any(np.diff(zz) <= 0):
        raise ValueError("z and dN_dz_domega must be matching increasing arrays")
    if np.any(counts < 0) or np.any(~np.isfinite(counts)):
        raise ValueError("counts must be finite and non-negative")
    dvol = np.array([volume_element_mpc3_sr_dz(background, float(zv)) for zv in zz])
    return np.divide(counts, dvol, out=np.zeros_like(counts), where=dvol > 0)


def fkp_weight(nbar_mpc3: np.ndarray, P0_mpc3: float) -> np.ndarray:
    """Feldman-Kaiser-Peacock weight w=1/(1+nbar P0), before completeness weights."""
    nbar = np.asarray(nbar_mpc3, dtype=float)
    if np.any(nbar < 0) or np.any(~np.isfinite(nbar)) or P0_mpc3 < 0 or not np.isfinite(P0_mpc3):
        raise ValueError("nbar and P0 must be finite and non-negative")
    return 1.0 / (1.0 + nbar * P0_mpc3)


def effective_volume_mpc3(
    background: BackgroundCosmology,
    z: np.ndarray,
    nbar_mpc3: np.ndarray,
    P0_mpc3: float,
    solid_angle_sr: float,
) -> float:
    r"""Compute V_eff = int dV [nP0/(1+nP0)]² over supplied z bins."""
    zz, nbar = np.asarray(z, float), np.asarray(nbar_mpc3, float)
    if zz.ndim != 1 or zz.shape != nbar.shape or len(zz) < 3 or np.any(np.diff(zz) <= 0):
        raise ValueError("z and nbar must be matching increasing arrays with >=3 points")
    if solid_angle_sr < 0 or P0_mpc3 < 0 or np.any(nbar < 0):
        raise ValueError("solid angle, P0 and nbar must be non-negative")
    factor = (nbar * P0_mpc3 / (1 + nbar * P0_mpc3)) ** 2
    dvol = np.array([volume_element_mpc3_sr_dz(background, float(zi)) for zi in zz])
    return float(solid_angle_sr * simpson(dvol * factor, x=zz))


def shot_noise_power(nbar_mpc3: float) -> float:
    """Poisson shot-noise power 1/nbar in Mpc^3."""
    if nbar_mpc3 <= 0 or not np.isfinite(nbar_mpc3):
        raise ValueError("nbar_mpc3 must be finite and positive")
    return 1.0 / nbar_mpc3


__all__ = [
    "normalize_selection_function",
    "comoving_number_density",
    "fkp_weight",
    "effective_volume_mpc3",
    "shot_noise_power",
]
