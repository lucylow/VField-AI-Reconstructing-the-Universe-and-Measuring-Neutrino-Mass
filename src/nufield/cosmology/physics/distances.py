"""Cosmological distance, volume and redshift-conversion utilities."""

from __future__ import annotations

import math

from scipy.integrate import quad
from scipy.optimize import brentq

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.constants import C_KM_S


def radial_comoving_distance_mpc(
    background: BackgroundCosmology, z: float, *, epsrel: float = 1e-8
) -> float:
    """Line-of-sight comoving distance D_C in Mpc."""
    if not math.isfinite(z) or z < 0:
        raise ValueError("z must be finite and non-negative")
    integral = quad(lambda x: 1.0 / float(background.E(x)), 0.0, z, epsrel=epsrel, limit=250)[0]
    return C_KM_S / background.params.H0_km_s_Mpc * float(integral)


def transverse_comoving_distance_mpc(background: BackgroundCosmology, z: float) -> float:
    """Transverse comoving distance D_M including constant spatial curvature."""
    dc = radial_comoving_distance_mpc(background, z)
    ok = background.params.omega_k
    if abs(ok) < 1e-8:
        return dc
    dh = C_KM_S / background.params.H0_km_s_Mpc
    chi = dc / dh
    if ok > 0:
        return dh / math.sqrt(ok) * math.sinh(math.sqrt(ok) * chi)
    return dh / math.sqrt(-ok) * math.sin(math.sqrt(-ok) * chi)


def angular_diameter_distance_mpc(background: BackgroundCosmology, z: float) -> float:
    return transverse_comoving_distance_mpc(background, z) / (1 + z)


def luminosity_distance_mpc(background: BackgroundCosmology, z: float) -> float:
    return transverse_comoving_distance_mpc(background, z) * (1 + z)


def hubble_distance_mpc(background: BackgroundCosmology, z: float) -> float:
    return C_KM_S / float(background.H(z))


def volume_element_mpc3_sr_dz(background: BackgroundCosmology, z: float) -> float:
    """Differential comoving volume dV/(dz dOmega) in Mpc^3 sr^-1."""
    dm = transverse_comoving_distance_mpc(background, z)
    return dm**2 * C_KM_S / float(background.H(z))


def comoving_volume_shell_mpc3(
    background: BackgroundCosmology,
    z_min: float,
    z_max: float,
    solid_angle_sr: float = 4 * math.pi,
) -> float:
    """Integrate the differential comoving volume across a redshift shell."""
    if z_min < 0 or z_max <= z_min or solid_angle_sr < 0:
        raise ValueError("require 0 <= z_min < z_max and solid_angle_sr >= 0")
    volume = quad(
        lambda z: volume_element_mpc3_sr_dz(background, z),
        z_min,
        z_max,
        epsrel=1e-6,
        limit=200,
    )[0]
    return float(volume * solid_angle_sr)


def redshift_at_comoving_distance(
    background: BackgroundCosmology,
    distance_mpc: float,
    *,
    z_bracket: tuple[float, float] = (0.0, 20.0),
) -> float:
    """Invert D_C(z) using a bounded root solver."""
    if distance_mpc < 0 or not math.isfinite(distance_mpc):
        raise ValueError("distance_mpc must be finite and non-negative")
    if distance_mpc == 0:
        return 0.0
    lo, hi = z_bracket
    if lo < 0 or hi <= lo:
        raise ValueError("invalid redshift bracket")
    f_lo = radial_comoving_distance_mpc(background, lo) - distance_mpc
    f_hi = radial_comoving_distance_mpc(background, hi) - distance_mpc
    if f_lo * f_hi > 0:
        raise ValueError("requested distance is outside the supplied redshift bracket")
    return float(
        brentq(lambda z: radial_comoving_distance_mpc(background, z) - distance_mpc, lo, hi)
    )


def distance_modulus(background: BackgroundCosmology, z: float) -> float:
    """Distance modulus 5 log10(D_L/Mpc) + 25; undefined at z=0."""
    dl = luminosity_distance_mpc(background, z)
    if dl <= 0:
        raise ValueError("distance modulus is undefined for zero luminosity distance")
    return float(5 * math.log10(dl) + 25)


__all__ = [
    "radial_comoving_distance_mpc",
    "transverse_comoving_distance_mpc",
    "angular_diameter_distance_mpc",
    "luminosity_distance_mpc",
    "hubble_distance_mpc",
    "volume_element_mpc3_sr_dz",
    "comoving_volume_shell_mpc3",
    "redshift_at_comoving_distance",
    "distance_modulus",
]
