"""Back-of-the-envelope neutrino free-streaming scales for diagnostics only.

The characteristic estimates are useful for selecting plots and intuition but are not transfer-function
predictions. Massive-neutrino suppression and its time dependence should be taken from CLASS/CAMB or a
validated simulation with the intended neutrino treatment.
"""

from __future__ import annotations

import math

import numpy as np

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.neutrinos import thermal_velocity_km_s


def characteristic_free_streaming_wavenumber_mpc_inv(
    background: BackgroundCosmology,
    mass_ev: float,
    z: float,
) -> float:
    r"""Estimate k_fs≈sqrt(3/2) a H(z)/v_th(z), in comoving Mpc^-1.

    This instantaneous estimate uses a characteristic thermal velocity and is not a sharp physical
    cutoff. The formula is only intended to orient scale choices in diagnostics.
    """
    if mass_ev <= 0 or z < 0:
        raise ValueError("mass_ev must be positive and z non-negative")
    a = 1.0 / (1.0 + z)
    velocity = thermal_velocity_km_s(mass_ev, z, background.params.T_cmb_K)
    return math.sqrt(1.5) * a * float(background.H(z)) / velocity


def free_streaming_horizon_estimate_mpc(
    background: BackgroundCosmology,
    mass_ev: float,
    z_start: float,
    z_end: float = 0.0,
    *,
    samples: int = 1024,
) -> float:
    """Integrate characteristic v/(a² H) over redshift for an approximate comoving travel distance.

    This is a diagnostic approximation, not a Boltzmann hierarchy solution. It becomes unreliable
    while a neutrino is relativistic and should never be used to generate a suppression transfer function.
    """
    if mass_ev <= 0 or z_start <= z_end or z_end < 0 or samples < 16:
        raise ValueError("require mass>0, z_start>z_end>=0, samples>=16")
    z = np.linspace(z_end, z_start, samples)
    # dχ_free = v dt/a = v dz/[H(z)] because |dt/dz|=1/[(1+z)H], 1/a=(1+z).
    v = np.array([thermal_velocity_km_s(mass_ev, float(zi), background.params.T_cmb_K) for zi in z])
    h = np.asarray(background.H(z), float)
    integrand = v / h
    return float(np.trapz(integrand, z))


__all__ = [
    "characteristic_free_streaming_wavenumber_mpc_inv",
    "free_streaming_horizon_estimate_mpc",
]
