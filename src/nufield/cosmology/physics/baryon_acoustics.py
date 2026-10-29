"""Photon-baryon sound speed and approximate sound-horizon integration.

The drag sound horizon r_d used in precision BAO analyses should normally be taken from a validated
recombination/Boltzmann calculation (e.g. CAMB or CLASS). The integral here assumes tight coupling and
an externally supplied drag redshift; it is a useful dimensional and implementation cross-check only.
"""

from __future__ import annotations

import math

from scipy.integrate import quad

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.constants import C_KM_S


def baryon_photon_inertia_ratio(background: BackgroundCosmology, scale_factor: float) -> float:
    """R(a)=3 rho_b/(4 rho_gamma), ignoring small thermal-history corrections."""
    if not 0 < scale_factor <= 1:
        raise ValueError("scale_factor must lie in (0, 1]")
    return 3 * background.params.omega_b / (4 * background.omega_gamma0) * scale_factor


def photon_baryon_sound_speed_km_s(background: BackgroundCosmology, scale_factor: float) -> float:
    """Tight-coupling sound speed c_s=c/sqrt(3(1+R))."""
    ratio = baryon_photon_inertia_ratio(background, scale_factor)
    return C_KM_S / math.sqrt(3 * (1 + ratio))


def approximate_sound_horizon_mpc(
    background: BackgroundCosmology,
    z_end: float,
    *,
    a_min: float = 1e-8,
    epsrel: float = 1e-6,
) -> float:
    """Integrate r_s(z_end)=integral_0^{a_end} c_s(a)/(a² H(a)) da.

    This method does not solve recombination or baryon drag. Use a solver-provided ``rdrag`` for
    inference. It assumes the background model remains applicable at the high redshifts integrated.
    """
    if z_end <= 0 or not math.isfinite(z_end):
        raise ValueError("z_end must be finite and positive")
    a_end = 1.0 / (1.0 + z_end)
    if not 0 < a_min < a_end:
        raise ValueError("a_min must be in (0, 1/(1+z_end))")

    def integrand(log_a: float) -> float:
        a = math.exp(log_a)
        z = 1 / a - 1
        cs = photon_baryon_sound_speed_km_s(background, a)
        return cs / (a * float(background.H(z)))

    val = quad(integrand, math.log(a_min), math.log(a_end), epsrel=epsrel, limit=300)[0]
    return float(val)


__all__ = [
    "baryon_photon_inertia_ratio",
    "photon_baryon_sound_speed_km_s",
    "approximate_sound_horizon_mpc",
]
