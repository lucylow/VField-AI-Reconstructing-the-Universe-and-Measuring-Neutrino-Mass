"""Neutrino mass hierarchy and thermal-background calculations.

The thermal integrals here implement the homogeneous Fermi--Dirac background energy density. They do
not solve neutrino perturbations, anisotropic stress, or scale-dependent free streaming. Those effects
must be handled by a Boltzmann solver such as CLASS or CAMB for precision power spectra.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from nufield.cosmology.physics.constants import (
    HBAR_C_EV_CM,
    K_B_EV_K,
    NEUTRINO_TO_PHOTON_REL,
    RHO_CRIT_H2_EV_CM3,
    TCMB0_K,
    TNU_TO_TGAMMA,
)

# Representative oscillation values used for hierarchy bookkeeping. They are configurable at the
# function level and should be pinned to a cited global fit in any published analysis.
DEFAULT_DM2_SOLAR_EV2 = 7.42e-5
DEFAULT_DM2_ATM_NORMAL_EV2 = 2.517e-3
DEFAULT_DM2_ATM_INVERTED_EV2 = 2.498e-3

_GL_NODES, _GL_WEIGHTS = leggauss(160)
# Integrand decays exponentially; mapping Gauss-Legendre nodes to [0, 50] is sufficient for this
# background integral at double precision for the intended use in a homogeneous expansion model.
_Q_NODES = 25.0 * (_GL_NODES + 1.0)
_Q_WEIGHTS = 25.0 * _GL_WEIGHTS
_FD_DENOM = np.exp(_Q_NODES) + 1.0
_FD_BASE = _Q_NODES**2 / _FD_DENOM


def solve_neutrino_masses(
    sum_mnu_ev: float,
    hierarchy: str = "normal",
    *,
    dm2_solar_ev2: float = DEFAULT_DM2_SOLAR_EV2,
    dm2_atm_normal_ev2: float = DEFAULT_DM2_ATM_NORMAL_EV2,
    dm2_atm_inverted_ev2: float = DEFAULT_DM2_ATM_INVERTED_EV2,
) -> tuple[float, float, float]:
    """Solve three mass eigenvalues for a fixed mass sum and hierarchy.

    The normal hierarchy has m2²-m1²=dm2_solar and m3²-m1²=dm2_atm. The inverted hierarchy
    uses m1 as the heavier partner of m2, with m2²-m1²=dm2_solar and m1²-m3²=dm2_atm.
    These are conventional reduced parameterizations; exact global-fit splittings evolve by fit.
    """
    total = float(sum_mnu_ev)
    if not math.isfinite(total) or total < 0:
        raise ValueError("sum_mnu_ev must be finite and non-negative")
    if dm2_solar_ev2 <= 0 or dm2_atm_normal_ev2 <= 0 or dm2_atm_inverted_ev2 <= 0:
        raise ValueError("mass-squared splittings must be positive")
    if hierarchy not in {"normal", "inverted"}:
        raise ValueError("hierarchy must be 'normal' or 'inverted'")
    if hierarchy == "normal":

        def masses(m0: float) -> tuple[float, float, float]:
            return (m0, math.sqrt(m0 * m0 + dm2_solar_ev2), math.sqrt(m0 * m0 + dm2_atm_normal_ev2))
    else:

        def masses(m0: float) -> tuple[float, float, float]:
            m1 = math.sqrt(m0 * m0 + dm2_atm_inverted_ev2)
            return (m1, math.sqrt(m1 * m1 + dm2_solar_ev2), m0)

    minimum = sum(masses(0.0))
    if total < minimum - 1e-12:
        raise ValueError(
            f"sum_mnu_ev={total:.6g} eV is below the {hierarchy} hierarchy minimum "
            f"{minimum:.6g} eV for the configured mass splittings"
        )
    if abs(total - minimum) <= 1e-12:
        return masses(0.0)
    root = brentq(lambda m0: sum(masses(m0)) - total, 0.0, max(total, 1.0), xtol=1e-14)
    result = masses(root)
    if not np.isclose(sum(result), total, rtol=0, atol=1e-10):
        raise ArithmeticError("neutrino hierarchy root did not recover the requested mass sum")
    return tuple(float(x) for x in result)


def neutrino_temperature_eV(z: float | np.ndarray, T_cmb_K: float = TCMB0_K) -> np.ndarray:
    """Relic neutrino temperature in eV for standard post-e+e- annihilation redshifting."""
    zz = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(zz)) or np.any(zz < 0) or T_cmb_K <= 0:
        raise ValueError("redshift must be finite and >= 0; T_cmb_K must be positive")
    return TNU_TO_TGAMMA * T_cmb_K * K_B_EV_K * (1.0 + zz)


def neutrino_energy_density_eV_cm3(
    mass_ev: float | np.ndarray,
    z: float | np.ndarray = 0.0,
    *,
    T_cmb_K: float = TCMB0_K,
) -> np.ndarray:
    r"""Homogeneous energy density for one neutrino+antineutrino species (eV/cm^3).

    Uses rho = T_nu^4/(pi^2 (hbar c)^3) int dq q^2 sqrt(q^2+(m/T_nu)^2)/(exp(q)+1).
    This background integral assumes zero chemical potential.
    """
    m = np.asarray(mass_ev, dtype=float)
    zz = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(m)) or np.any(m < 0):
        raise ValueError("mass_ev must be finite and non-negative")
    if np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("z must be finite and non-negative")
    tnu = neutrino_temperature_eV(zz, T_cmb_K)
    y = m / tnu
    # Broadcast over z and species without allocating a q grid with Python loops.
    integrand = _FD_BASE * np.sqrt(_Q_NODES**2 + np.expand_dims(y, axis=-1) ** 2)
    integral = np.sum(integrand * _Q_WEIGHTS, axis=-1)
    rho = np.asarray(tnu) ** 4 * integral / (math.pi**2 * HBAR_C_EV_CM**3)
    return np.asarray(rho)


def omega_gamma_h2(T_cmb_K: float = TCMB0_K) -> float:
    """Photon density parameter times h^2 for a blackbody CMB temperature."""
    if T_cmb_K <= 0 or not math.isfinite(T_cmb_K):
        raise ValueError("T_cmb_K must be finite and positive")
    return 2.472975e-5 * (T_cmb_K / 2.7255) ** 4


def omega_gamma(T_cmb_K: float = TCMB0_K, h: float = 0.674) -> float:
    """Photon density parameter Ω_gamma at z=0."""
    if h <= 0:
        raise ValueError("h must be positive")
    return omega_gamma_h2(T_cmb_K) / h**2


def omega_nu_massive(
    masses_ev: tuple[float, ...] | list[float] | np.ndarray,
    *,
    h: float,
    T_cmb_K: float = TCMB0_K,
) -> float:
    """Present-day density fraction of the specified thermal massive species."""
    if h <= 0:
        raise ValueError("h must be positive")
    masses = np.asarray(masses_ev, dtype=float)
    if masses.ndim != 1:
        raise ValueError("masses_ev must be one-dimensional")
    rho = np.asarray(neutrino_energy_density_eV_cm3(masses, 0.0, T_cmb_K=T_cmb_K))
    rho_crit = RHO_CRIT_H2_EV_CM3 * h**2
    return float(rho.sum() / rho_crit)


def omega_nu_nonrelativistic_approx(sum_mnu_ev: float, h: float) -> float:
    """Common late-time approximation Ω_ν h² ≈ Σmν/(93.14 eV).

    Do not use as an exact background density through the relativistic-to-nonrelativistic transition.
    """
    if sum_mnu_ev < 0 or h <= 0:
        raise ValueError("sum_mnu_ev must be non-negative and h positive")
    return float(sum_mnu_ev / (93.14 * h**2))


def massless_neutrino_omega(
    N_eff: float,
    n_massive_species: int,
    *,
    T_cmb_K: float = TCMB0_K,
    h: float = 0.674,
) -> float:
    """Ω of residual massless effective neutrinos, excluding explicitly modelled massive species."""
    residual = max(float(N_eff) - int(n_massive_species), 0.0)
    return omega_gamma(T_cmb_K, h) * NEUTRINO_TO_PHOTON_REL * residual


def neutrino_number_density_cm3(z: float = 0.0, T_cmb_K: float = TCMB0_K) -> float:
    """Number density for one thermal neutrino+antineutrino species, approximated standardly."""
    if z < 0:
        raise ValueError("z must be non-negative")
    # Standard massless Fermi-Dirac number density per ν+anti-ν species is about 112 cm^-3 today.
    return 112.0 * (T_cmb_K / 2.7255) ** 3 * (1.0 + z) ** 3


def thermal_velocity_km_s(mass_ev: float, z: float = 0.0, T_cmb_K: float = TCMB0_K) -> float:
    """Characteristic non-relativistic thermal speed, using <p>≈3.151 T_nu.

    This is not a full velocity distribution and is not a free-streaming scale calculation.
    """
    if mass_ev <= 0 or z < 0:
        raise ValueError("mass_ev must be positive and z non-negative")
    p_ev = 3.151 * float(neutrino_temperature_eV(z, T_cmb_K))
    return 299_792.458 * p_ev / math.sqrt(p_ev**2 + mass_ev**2)


__all__ = [
    "solve_neutrino_masses",
    "neutrino_temperature_eV",
    "neutrino_energy_density_eV_cm3",
    "omega_gamma_h2",
    "omega_gamma",
    "omega_nu_massive",
    "omega_nu_nonrelativistic_approx",
    "massless_neutrino_omega",
    "neutrino_number_density_cm3",
    "thermal_velocity_km_s",
    "neutrino_pressure_eV_cm3",
    "neutrino_equation_of_state_w",
]


def neutrino_pressure_eV_cm3(
    mass_ev: float | np.ndarray,
    z: float | np.ndarray = 0.0,
    *,
    T_cmb_K: float = TCMB0_K,
) -> np.ndarray:
    r"""Homogeneous pressure for one or more thermal neutrino species in eV/cm^3.

    P_nu = T_nu^4/[3 pi^2 (hbar c)^3] int dq q^4/[sqrt(q^2+(m/T)^2)(exp(q)+1)].
    The result describes the isotropic thermal background only, not perturbed anisotropic stress.
    """
    m = np.asarray(mass_ev, dtype=float)
    zz = np.asarray(z, dtype=float)
    if np.any(~np.isfinite(m)) or np.any(m < 0) or np.any(~np.isfinite(zz)) or np.any(zz < 0):
        raise ValueError("mass_ev and z must be finite and non-negative")
    tnu = neutrino_temperature_eV(zz, T_cmb_K)
    y = m / tnu
    denom = np.sqrt(_Q_NODES**2 + np.expand_dims(y, axis=-1) ** 2)
    integrand = (_Q_NODES**2 / _FD_DENOM) * _Q_NODES**2 / denom
    integral = np.sum(integrand * _Q_WEIGHTS, axis=-1)
    pressure = np.asarray(tnu) ** 4 * integral / (3.0 * math.pi**2 * HBAR_C_EV_CM**3)
    return np.asarray(pressure)


def neutrino_equation_of_state_w(
    mass_ev: float | np.ndarray,
    z: float | np.ndarray = 0.0,
    *,
    T_cmb_K: float = TCMB0_K,
) -> np.ndarray:
    """Return background neutrino w_nu=P_nu/rho_nu, interpolating radiation to matter behavior."""
    rho = neutrino_energy_density_eV_cm3(mass_ev, z, T_cmb_K=T_cmb_K)
    pressure = neutrino_pressure_eV_cm3(mass_ev, z, T_cmb_K=T_cmb_K)
    return np.divide(
        pressure, rho, out=np.zeros_like(np.asarray(pressure)), where=np.asarray(rho) > 0
    )
