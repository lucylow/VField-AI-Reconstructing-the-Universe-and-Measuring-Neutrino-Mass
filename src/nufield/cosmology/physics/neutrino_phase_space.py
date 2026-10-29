"""Thermal Fermi--Dirac momentum sampling utilities for simulation diagnostics.

This samples the homogeneous relic momentum distribution. Real massive-neutrino N-body initial
conditions also require correct transfer functions, bulk perturbation velocities, mass hierarchy,
redshift convention, particle weights and a validated gravity/force treatment. This sampler alone does
not create scientifically complete neutrino initial conditions.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from nufield.cosmology.physics.constants import C_KM_S, TCMB0_K
from nufield.cosmology.physics.neutrinos import neutrino_temperature_eV


def fermi_dirac_momentum_pdf(q: np.ndarray | float) -> np.ndarray:
    r"""Normalized number-weighted momentum magnitude PDF f(q) ∝ q²/(exp(q)+1), q=p/Tν."""
    qq = np.asarray(q, float)
    if np.any(qq < 0) or np.any(~np.isfinite(qq)):
        raise ValueError("q must be finite and non-negative")
    normalization = 1.5 * 1.202056903159594  # ∫ q²/(e^q+1)dq = 3 ζ(3)/2
    return qq**2 / (np.exp(np.minimum(qq, 700)) + 1) / normalization


def sample_dimensionless_momentum(
    n_samples: int,
    *,
    seed: int,
    q_max: float = 40.0,
    table_size: int = 100_000,
) -> np.ndarray:
    """Sample q=p/T from an inverse-CDF table with a reproducible RNG seed."""
    if n_samples < 1 or seed < 0 or q_max < 10 or table_size < 1000:
        raise ValueError("n_samples>=1, seed>=0, q_max>=10 and table_size>=1000 required")
    q = np.linspace(0.0, q_max, table_size)
    pdf = fermi_dirac_momentum_pdf(q)
    cdf = np.concatenate(([0.0], cumulative_trapezoid(pdf, q)))
    cdf /= cdf[-1]
    # Remove tiny non-monotonicity due to floating-point plateau at the tail.
    unique, idx = np.unique(cdf, return_index=True)
    q_unique = q[idx]
    draws = np.random.default_rng(seed).uniform(0, 1, n_samples)
    return np.interp(draws, unique, q_unique)


def sample_neutrino_thermal_velocities(
    mass_ev: float,
    z: float,
    n_samples: int,
    *,
    seed: int,
    T_cmb_K: float = TCMB0_K,
) -> dict[str, np.ndarray]:
    """Sample isotropic 3-D thermal velocities with speed c p/sqrt(p²+m²).

    Momentum is represented in eV with c=1 in the energy relation. The vector directions are isotropic;
    perturbative bulk velocities and correlations with density are not included.
    """
    if mass_ev <= 0 or z < 0 or not np.isfinite(mass_ev + z):
        raise ValueError("mass_ev must be finite positive and z finite non-negative")
    q = sample_dimensionless_momentum(n_samples, seed=seed)
    tnu = float(neutrino_temperature_eV(z, T_cmb_K))
    p = q * tnu
    speed = C_KM_S * p / np.sqrt(p * p + mass_ev * mass_ev)
    rng = np.random.default_rng(seed + 1)
    cos_theta = rng.uniform(-1, 1, n_samples)
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    sin_theta = np.sqrt(1 - cos_theta**2)
    unit = np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))
    return {
        "q_dimensionless": q,
        "momentum_ev": p,
        "speed_km_s": speed,
        "velocity_km_s": unit * speed[:, None],
        "metadata_note": "thermal draws only; no bulk perturbation velocity",
    }


__all__ = [
    "fermi_dirac_momentum_pdf",
    "sample_dimensionless_momentum",
    "sample_neutrino_thermal_velocities",
]
