"""Approximate Press--Schechter and Sheth--Tormen halo mass functions.

These analytic fits are pedagogical baselines and are not calibrated to a chosen massive-neutrino
N-body suite or halo definition. Use them to test variance integrals and pipeline contracts, not to
claim precision halo abundance predictions.
"""

from __future__ import annotations

import math

import numpy as np

from nufield.cosmology.physics.constants import MASS_DENSITY_CONVERSION_MSUN_MPC3
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


def top_hat_radius_mpc(mass_msun: float | np.ndarray, mean_density_msun_mpc3: float) -> np.ndarray:
    """Radius of a real-space spherical top-hat enclosing mass M at mean density rho-bar."""
    mass = np.asarray(mass_msun, float)
    if np.any(mass <= 0) or np.any(~np.isfinite(mass)) or mean_density_msun_mpc3 <= 0:
        raise ValueError("mass and mean density must be positive and finite")
    return (3 * mass / (4 * math.pi * mean_density_msun_mpc3)) ** (1.0 / 3.0)


def sigma_mass(
    spectrum: MatterPowerSpectrum,
    mass_msun: float | np.ndarray,
    mean_density_msun_mpc3: float,
) -> np.ndarray:
    """Compute σ(M) by mapping M to a real-space top-hat radius and integrating P(k)."""
    mass = np.asarray(mass_msun, float)
    radii = np.asarray(top_hat_radius_mpc(mass, mean_density_msun_mpc3))
    flat_r = radii.ravel()
    result = np.array([spectrum.sigma_r(float(r)) for r in flat_r]).reshape(radii.shape)
    return result


def sheth_tormen_multiplicity(
    nu: np.ndarray | float,
    *,
    A: float = 0.3222,
    a: float = 0.707,
    p: float = 0.3,
) -> np.ndarray:
    """Classic Sheth--Tormen multiplicity approximation as a function of ν=δ_c/σ."""
    v = np.asarray(nu, float)
    if np.any(v <= 0) or A <= 0 or a <= 0 or p < 0 or np.any(~np.isfinite(v)):
        raise ValueError("nu, A and a must be positive and p non-negative")
    return A * np.sqrt(2 * a / np.pi) * v * (1 + (a * v * v) ** (-p)) * np.exp(-0.5 * a * v * v)


def sheth_tormen_dn_dlnm(
    spectrum: MatterPowerSpectrum,
    mass_msun: np.ndarray,
    mean_density_msun_mpc3: float,
    *,
    delta_c: float = 1.686,
) -> np.ndarray:
    """Return approximate dn/dlnM in Mpc^-3 for strictly increasing mass values."""
    mass = np.asarray(mass_msun, float)
    if mass.ndim != 1 or len(mass) < 3 or np.any(mass <= 0) or np.any(np.diff(mass) <= 0):
        raise ValueError("mass_msun must be a strictly increasing positive 1-D array")
    if delta_c <= 0:
        raise ValueError("delta_c must be positive")
    sig = np.asarray(sigma_mass(spectrum, mass, mean_density_msun_mpc3))
    if np.any(sig <= 0):
        raise ValueError("sigma(M) must be positive; check spectrum scale support")
    nu = delta_c / sig
    dlnnu_dlnm = np.gradient(np.log(nu), np.log(mass))
    multiplicity = sheth_tormen_multiplicity(nu)
    return mean_density_msun_mpc3 / mass * multiplicity * np.abs(dlnnu_dlnm)


def mean_matter_density_msun_mpc3(omega_m: float, h: float) -> float:
    """Present-day mean matter density in Msun/Mpc³ under the rho_crit conversion used here."""
    if omega_m <= 0 or h <= 0:
        raise ValueError("omega_m and h must be positive")
    return float(omega_m * MASS_DENSITY_CONVERSION_MSUN_MPC3 * h**2)


__all__ = [
    "top_hat_radius_mpc",
    "sigma_mass",
    "sheth_tormen_multiplicity",
    "sheth_tormen_dn_dlnm",
    "mean_matter_density_msun_mpc3",
]
