"""Batch growth-history table and reproducible CSV export."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.growth import growth_factor_scale_independent, growth_rate_f


def growth_table(
    background: BackgroundCosmology,
    redshifts: np.ndarray,
    *,
    allow_massive_neutrino_approximation: bool = False,
) -> dict[str, np.ndarray]:
    """Produce D(z), f(z), H(z), and Ω_m(z) diagnostics on a supplied redshift grid."""
    z = np.asarray(redshifts, float)
    if z.ndim != 1 or len(z) < 2 or np.any(z < 0) or np.any(np.diff(z) <= 0):
        raise ValueError("redshifts must be increasing, non-negative, 1-D values")
    d = growth_factor_scale_independent(
        background, z, allow_massive_neutrino_approximation=allow_massive_neutrino_approximation
    )
    f = growth_rate_f(
        background, z, allow_massive_neutrino_approximation=allow_massive_neutrino_approximation
    )
    omega_m = []
    for zi in z:
        frac = background.density_fractions(float(zi))
        omega_m.append(frac["baryon"] + frac["cdm"] + frac["nu_massive"])
    return {
        "z": z,
        "H_km_s_Mpc": np.asarray(background.H(z)),
        "D_normalized": d,
        "f_growth": f,
        "omega_m_fraction": np.asarray(omega_m),
    }


def write_growth_csv(path: str | Path, table: dict[str, np.ndarray]) -> Path:
    """Write a growth table whose columns and row counts are checked before output."""
    out = Path(path)
    if not table:
        raise ValueError("table cannot be empty")
    arrays = {key: np.asarray(value) for key, value in table.items()}
    lengths = {arr.size for arr in arrays.values()}
    if len(lengths) != 1 or any(arr.ndim != 1 for arr in arrays.values()):
        raise ValueError("all table columns must be 1-D and equal length")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(arrays.keys())
        for row in zip(*arrays.values()):
            writer.writerow([float(v) for v in row])
    return out


__all__ = ["growth_table", "write_growth_csv"]
