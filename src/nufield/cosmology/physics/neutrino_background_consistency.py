"""Continuity-equation and asymptotic checks for thermal neutrino backgrounds.

For an uncoupled homogeneous species, energy conservation implies dρ/dln(a) + 3(ρ+P)=0.
This module checks the numerical Fermi--Dirac rho/P integrals already implemented by νField. These
checks do not include perturbations, anisotropic stress, or Boltzmann hierarchy evolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.neutrinos import (
    neutrino_energy_density_eV_cm3,
    neutrino_equation_of_state_w,
    neutrino_pressure_eV_cm3,
)


@dataclass(frozen=True, slots=True)
class ContinuityCheckReport:
    mass_ev: float
    z_range: tuple[float, float]
    n_samples: int
    max_fractional_continuity_residual: float
    median_fractional_continuity_residual: float
    w_min: float
    w_max: float
    relativistic_scaling_error: float | None
    nonrelativistic_scaling_error: float | None
    passed: bool
    tolerance: float
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mass_ev": self.mass_ev,
            "z_range": list(self.z_range),
            "n_samples": self.n_samples,
            "max_fractional_continuity_residual": self.max_fractional_continuity_residual,
            "median_fractional_continuity_residual": self.median_fractional_continuity_residual,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "relativistic_scaling_error": self.relativistic_scaling_error,
            "nonrelativistic_scaling_error": self.nonrelativistic_scaling_error,
            "passed": self.passed,
            "tolerance": self.tolerance,
            "caveats": list(self.caveats),
        }


def validate_neutrino_continuity(
    mass_ev: float,
    *,
    z_min: float = 0.0,
    z_max: float = 1e4,
    n_samples: int = 512,
    T_cmb_K: float = 2.7255,
    tolerance: float = 3e-2,
) -> ContinuityCheckReport:
    """Check the thermal rho/P continuity equation over a logarithmic scale-factor grid.

    Differentiation is performed with respect to ln(a), so the grid is uniform in the independent
    variable appropriate for the continuity equation. Endpoint derivative error is larger; report
    quantiles instead of hiding the maximum. Increase resolution and tighten quadrature if checking a
    precision solver.
    """
    if not np.isfinite(mass_ev) or mass_ev < 0:
        raise ValueError("mass_ev must be finite and non-negative")
    if not np.isfinite(z_min + z_max) or z_min < 0 or z_max <= z_min:
        raise ValueError("require finite 0 <= z_min < z_max")
    if n_samples < 32:
        raise ValueError("n_samples must be >=32")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    # Uniform in ln(a), equivalently uniform in ln(1+z) in reverse order.
    ln_a = np.linspace(-np.log1p(z_max), -np.log1p(z_min), n_samples)
    z = np.expm1(-ln_a)
    rho = np.asarray(neutrino_energy_density_eV_cm3(mass_ev, z, T_cmb_K=T_cmb_K), float)
    pressure = np.asarray(neutrino_pressure_eV_cm3(mass_ev, z, T_cmb_K=T_cmb_K), float)
    derivative = np.gradient(rho, ln_a, edge_order=2)
    residual = derivative + 3.0 * (rho + pressure)
    scale = np.maximum(np.abs(derivative) + 3.0 * (rho + pressure), np.finfo(float).tiny)
    fractional = np.abs(residual) / scale
    w = np.asarray(neutrino_equation_of_state_w(mass_ev, z, T_cmb_K=T_cmb_K), float)
    # In the relativistic subset m/T<0.1, rho scales approximately a^-4; in the deep NR subset m/T>100,
    # rho scales approximately a^-3. These are optional if the requested redshift interval covers them.
    from nufield.cosmology.physics.neutrinos import neutrino_temperature_eV

    t = np.asarray(neutrino_temperature_eV(z, T_cmb_K), float)
    rel = mass_ev / t < 0.1
    nr = mass_ev / t > 100
    rel_err = None
    if np.count_nonzero(rel) >= 8:
        slope = np.polyfit(ln_a[rel], np.log(rho[rel]), 1)[0]
        rel_err = float(abs(slope + 4.0))
    nr_err = None
    if np.count_nonzero(nr) >= 8:
        slope = np.polyfit(ln_a[nr], np.log(rho[nr]), 1)[0]
        nr_err = float(abs(slope + 3.0))
    max_res = float(np.max(fractional))
    # Ignore a small number of finite-difference boundary points when deciding this coarse numerical check.
    interior = fractional[2:-2]
    med_res = float(np.median(interior))
    passed = bool(med_res <= tolerance and np.isfinite(max_res))
    return ContinuityCheckReport(
        float(mass_ev),
        (float(z_min), float(z_max)),
        int(n_samples),
        max_res,
        med_res,
        float(w.min()),
        float(w.max()),
        rel_err,
        nr_err,
        passed,
        float(tolerance),
        (
            "Finite-difference residual depends on grid resolution and quadrature.",
            "This tests homogeneous energy conservation only; it does not validate perturbation dynamics.",
            "Asymptotic slope diagnostics are only reported where enough points meet the regime threshold.",
        ),
    )


def continuity_resolution_study(
    mass_ev: float,
    *,
    resolutions: tuple[int, ...] = (64, 128, 256, 512, 1024),
    z_max: float = 1e4,
) -> dict[str, object]:
    """Report how the median continuity residual changes with differentiation resolution."""
    if (
        len(resolutions) < 2
        or any(n < 32 for n in resolutions)
        or any(b <= a for a, b in zip(resolutions, resolutions[1:]))
    ):
        raise ValueError("resolutions must be strictly increasing with at least two values >=32")
    reports = [
        validate_neutrino_continuity(mass_ev, z_max=z_max, n_samples=n, tolerance=1.0)
        for n in resolutions
    ]
    medians = np.asarray([r.median_fractional_continuity_residual for r in reports])
    changes = [float(abs(b - a) / max(abs(b), 1e-14)) for a, b in zip(medians, medians[1:])]
    return {
        "mass_ev": float(mass_ev),
        "resolutions": list(resolutions),
        "median_residuals": medians.tolist(),
        "fractional_change_per_refinement": changes,
        "max_residuals": [r.max_fractional_continuity_residual for r in reports],
        "warning": "Resolution convergence does not guarantee quadrature accuracy or a correct perturbation solver.",
    }


__all__ = ["ContinuityCheckReport", "validate_neutrino_continuity", "continuity_resolution_study"]
