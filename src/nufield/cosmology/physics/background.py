"""Homogeneous expansion history for curved CPL w0-wa cosmologies.

This module handles background evolution only. It does not compute perturbations, the CMB source
functions, or scale-dependent massive-neutrino growth. Precision observables should be obtained from
a validated Boltzmann solver and cross-checked against this background calculation.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from nufield.cosmology.physics.neutrinos import (
    massless_neutrino_omega,
    neutrino_energy_density_eV_cm3,
    omega_gamma,
    omega_nu_massive,
    omega_nu_nonrelativistic_approx,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology


class BackgroundCosmology:
    """Evaluate E(z), H(z), and component density fractions for a physical parameter object."""

    def __init__(self, params: PhysicalCosmology):
        self.params = params
        self._og = omega_gamma(params.T_cmb_K, params.h)
        self._onu_rel = massless_neutrino_omega(
            params.N_eff, params.n_massive_species, T_cmb_K=params.T_cmb_K, h=params.h
        )
        self._onu_massive0 = omega_nu_massive(
            params.neutrino_masses_ev, h=params.h, T_cmb_K=params.T_cmb_K
        )
        self._ode = (
            1.0
            - params.omega_k
            - self._og
            - self._onu_rel
            - params.omega_b
            - params.omega_cdm
            - self._onu_massive0
        )
        if self._ode < -0.5 or self._ode > 2.0:
            raise ValueError(f"derived Omega_DE={self._ode:.4g} is outside supported sanity bounds")

    @property
    def omega_gamma0(self) -> float:
        return self._og

    @property
    def omega_nu_rel0(self) -> float:
        return self._onu_rel

    @property
    def omega_nu_massive0(self) -> float:
        return self._onu_massive0

    @property
    def omega_de0(self) -> float:
        return self._ode

    @property
    def omega_m0(self) -> float:
        """Present-day baryon + CDM + massive-neutrino energy density fraction."""
        # The homogeneous energy density above includes the residual kinetic contribution. For the
        # conventional Omega_m parameter, use the standard late-time non-relativistic conversion; at
        # exactly zero mass this correctly excludes species that are radiation today.
        omega_nu_matter = omega_nu_nonrelativistic_approx(self.params.sum_mnu_ev, self.params.h)
        return self.params.omega_b + self.params.omega_cdm + omega_nu_matter

    def _massive_neutrino_omega_z(self, z: np.ndarray) -> np.ndarray:
        masses = np.asarray(self.params.neutrino_masses_ev, dtype=float)
        zz = np.asarray(z, dtype=float)
        mass_shape = (1,) * zz.ndim + (masses.size,)
        rho = np.asarray(
            neutrino_energy_density_eV_cm3(
                masses.reshape(mass_shape), zz[..., None], T_cmb_K=self.params.T_cmb_K
            )
        )
        from nufield.cosmology.physics.constants import RHO_CRIT_H2_EV_CM3

        return rho.sum(axis=-1) / (RHO_CRIT_H2_EV_CM3 * self.params.h**2)

    def e2(self, z: float | np.ndarray) -> np.ndarray:
        """Dimensionless expansion rate squared ``E(z)^2 = H(z)^2 / H0^2``."""
        zz = np.asarray(z, dtype=float)
        if np.any(~np.isfinite(zz)) or np.any(zz < 0):
            raise ValueError("z must be finite and non-negative")
        a_inv = 1.0 + zz
        de = (
            self._ode
            * a_inv ** (3.0 * (1.0 + self.params.w0 + self.params.wa))
            * np.exp(-3.0 * self.params.wa * zz / a_inv)
        )
        result = (
            self._og * a_inv**4
            + self._onu_rel * a_inv**4
            + (self.params.omega_b + self.params.omega_cdm) * a_inv**3
            + self.params.omega_k * a_inv**2
            + de
            + self._massive_neutrino_omega_z(zz)
        )
        if np.any(result <= 0) or np.any(~np.isfinite(result)):
            raise ValueError("E(z)^2 became non-positive or non-finite")
        return result

    def E(self, z: float | np.ndarray) -> np.ndarray:
        return np.sqrt(self.e2(z))

    def H(self, z: float | np.ndarray) -> np.ndarray:
        """Hubble parameter in km s^-1 Mpc^-1."""
        return self.params.H0_km_s_Mpc * self.E(z)

    def density_fractions(self, z: float) -> dict[str, float]:
        """Return component fractions at redshift z using homogeneous energy densities."""
        if z < 0 or not math.isfinite(z):
            raise ValueError("z must be finite and non-negative")
        a1 = 1.0 + z
        e2 = float(self.e2(z))
        nu_m = float(self._massive_neutrino_omega_z(np.asarray(z))) * 1.0
        de = (
            self._ode
            * a1 ** (3.0 * (1 + self.params.w0 + self.params.wa))
            * math.exp(-3 * self.params.wa * z / a1)
        )
        vals = {
            "gamma": self._og * a1**4,
            "nu_rel": self._onu_rel * a1**4,
            "baryon": self.params.omega_b * a1**3,
            "cdm": self.params.omega_cdm * a1**3,
            "nu_massive": nu_m,
            "curvature": self.params.omega_k * a1**2,
            "dark_energy": de,
        }
        return {key: float(value / e2) for key, value in vals.items()}

    def q_deceleration(self, z: float, dz: float = 1e-4) -> float:
        """Numerically evaluate q(z) = -1 + (1+z) H'(z)/H(z)."""
        if z < dz:
            dlnh_dz = (math.log(float(self.H(z + dz))) - math.log(float(self.H(z)))) / dz
        else:
            dlnh_dz = (math.log(float(self.H(z + dz))) - math.log(float(self.H(z - dz)))) / (2 * dz)
        return -1.0 + (1.0 + z) * dlnh_dz

    def age_integrand_gyr_per_dz(self, z: float) -> float:
        """Lookback-time integrand, Gyr per unit redshift."""
        seconds_per_gyr = 365.25 * 24 * 3600 * 1e9
        mpc_km = 3.085677581491367e19
        hubble_time_seconds = mpc_km / float(self.H(z))
        return hubble_time_seconds / seconds_per_gyr / (1.0 + z)

    def lookback_time_gyr(self, z: float, *, epsrel: float = 1e-7) -> float:
        if z < 0 or not math.isfinite(z):
            raise ValueError("z must be finite and non-negative")
        return float(quad(self.age_integrand_gyr_per_dz, 0.0, z, epsrel=epsrel)[0])

    def age_today_gyr(self, *, zmax: float = 1e5, epsrel: float = 1e-7) -> float:
        """Approximate cosmic age by integrating to a finite large-redshift boundary."""
        if zmax <= 0:
            raise ValueError("zmax must be positive")
        return float(quad(self.age_integrand_gyr_per_dz, 0.0, zmax, epsrel=epsrel, limit=300)[0])

    def omega_de_evolution_factor(self, z: float | np.ndarray) -> np.ndarray:
        """CPL dark-energy density ratio rho_DE(z)/rho_DE(0)."""
        zz = np.asarray(z, dtype=float)
        if np.any(zz < 0):
            raise ValueError("z must be non-negative")
        return (1.0 + zz) ** (3 * (1 + self.params.w0 + self.params.wa)) * np.exp(
            -3 * self.params.wa * zz / (1.0 + zz)
        )


__all__ = ["BackgroundCosmology"]
