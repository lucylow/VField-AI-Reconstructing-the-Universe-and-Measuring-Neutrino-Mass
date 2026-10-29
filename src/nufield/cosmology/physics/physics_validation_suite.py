"""Composable equation-level validation checks for the νField physics layer.

The suite is intentionally conservative: internal consistency is necessary but not sufficient for
scientific accuracy. Passing these checks does not validate a cosmology against observations or a
Boltzmann solver. Results are serializable and can be attached to experiment manifests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.neutrinos import (
    neutrino_energy_density_eV_cm3,
    neutrino_equation_of_state_w,
    neutrino_pressure_eV_cm3,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class PhysicsCheck:
    name: str
    passed: bool
    severity: str
    value: float | str | None
    expected: str
    tolerance: float | None
    message: str


@dataclass(frozen=True, slots=True)
class PhysicsValidationReport:
    schema_version: str
    model_label: str
    checks: tuple[PhysicsCheck, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed or check.severity != "error" for check in self.checks)

    @property
    def errors(self) -> tuple[PhysicsCheck, ...]:
        return tuple(c for c in self.checks if not c.passed and c.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_label": self.model_label,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "warnings": list(self.warnings),
        }


def _check_close(
    name: str,
    value: float,
    expected: float,
    atol: float,
    *,
    severity: str = "error",
    message: str = "",
) -> PhysicsCheck:
    passed = bool(np.isfinite(value) and abs(value - expected) <= atol)
    return PhysicsCheck(
        name,
        passed,
        severity,
        float(value),
        f"{expected:g} ± {atol:g}",
        atol,
        message or ("within tolerance" if passed else "outside tolerance"),
    )


def validate_background(
    params: PhysicalCosmology, redshifts: np.ndarray | None = None
) -> list[PhysicsCheck]:
    """Check normalization, positivity, and component closure of a homogeneous background."""
    z = np.asarray([0.0, 0.5, 1.0, 2.0, 5.0] if redshifts is None else redshifts, float)
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)) or np.any(z < 0):
        raise ValueError("redshifts must be a finite non-negative vector with at least two values")
    z = np.unique(np.sort(z))
    bg = BackgroundCosmology(params)
    checks: list[PhysicsCheck] = []
    checks.append(
        _check_close(
            "E(z=0)",
            float(bg.E(0.0)),
            1.0,
            2e-8,
            message="Background normalization is imposed through closure.",
        )
    )
    hz = np.asarray(bg.H(z), float)
    checks.append(
        PhysicsCheck(
            "H(z)>0",
            bool(np.all(np.isfinite(hz)) and np.all(hz > 0)),
            "error",
            float(np.min(hz)),
            "finite and positive for every requested z",
            None,
            "Expansion rate must remain finite and positive.",
        )
    )
    from nufield.cosmology.physics.distances import radial_comoving_distance_mpc

    dc = np.array([radial_comoving_distance_mpc(bg, float(zi)) for zi in z])
    checks.append(
        PhysicsCheck(
            "D_C(z) monotonic",
            bool(np.all(np.diff(dc) > -1e-8)),
            "error",
            float(np.min(np.diff(dc))),
            "non-decreasing with redshift",
            None,
            "Comoving radial distance must increase for a positive expansion rate.",
        )
    )
    closure_errors = []
    for zi in z:
        fractions = bg.density_fractions(float(zi))
        closure_errors.append(abs(sum(fractions.values()) - 1.0))
    max_closure = float(max(closure_errors))
    checks.append(
        _check_close(
            "density-fraction closure max error",
            max_closure,
            0.0,
            2e-6,
            message="Includes photons, residual radiation, baryons, CDM, massive neutrinos, curvature and dark energy.",
        )
    )
    return checks


def validate_neutrino_thermodynamics() -> list[PhysicsCheck]:
    """Check relativistic and non-relativistic limits of thermal background integrals."""
    checks: list[PhysicsCheck] = []
    z = np.array([0.0, 1.0, 10.0])
    rho0 = float(neutrino_energy_density_eV_cm3(0.0, 0.0))
    rhoz = np.asarray(neutrino_energy_density_eV_cm3(0.0, z), float)
    expected_ratio = (1.0 + z) ** 4
    rel_err = float(np.max(np.abs(rhoz / rho0 / expected_ratio - 1.0)))
    checks.append(
        _check_close(
            "massless rho_nu scales as a^-4",
            rel_err,
            0.0,
            2e-8,
            message="Thermal energy density should scale as (1+z)^4 for massless species.",
        )
    )
    wrel = float(neutrino_equation_of_state_w(0.0, 0.0))
    checks.append(_check_close("massless neutrino equation of state", wrel, 1.0 / 3.0, 2e-8))
    wr = float(neutrino_equation_of_state_w(0.6, 0.0))
    checks.append(
        PhysicsCheck(
            "massive species pressure suppressed today",
            wr < 0.02,
            "error",
            wr,
            "w < 0.02 for m=0.6 eV at z=0",
            0.02,
            "A species much heavier than its present thermal temperature is non-relativistic.",
        )
    )
    rho = float(neutrino_energy_density_eV_cm3(0.3, 0.0))
    pressure = float(neutrino_pressure_eV_cm3(0.3, 0.0))
    checks.append(
        PhysicsCheck(
            "thermal pressure non-negative and below rho/3",
            bool(np.isfinite(rho + pressure) and 0 <= pressure <= rho / 3 * (1 + 1e-10)),
            "error",
            pressure / rho if rho else None,
            "0 <= P/rho <= 1/3",
            None,
            "For an isotropic thermal species, the equation-of-state ratio approaches 1/3 when relativistic.",
        )
    )
    return checks


def validate_power_spectrum(spectrum: MatterPowerSpectrum) -> list[PhysicsCheck]:
    """Check basic power-spectrum contract and dimensional support without presuming a shape."""
    k, p = spectrum.k_mpc_inv, spectrum.pk_mpc3
    return [
        PhysicsCheck(
            "power spectrum finite and positive",
            bool(np.all(np.isfinite(k + p)) and np.all(p > 0)),
            "error",
            float(np.min(p)),
            "P(k)>0 and finite",
            None,
            "Power is stored in Mpc^3 and k in Mpc^-1.",
        ),
        PhysicsCheck(
            "k ordering",
            bool(np.all(np.diff(k) > 0)),
            "error",
            float(
                np.min(np.diff(k)),
            ),
            "strictly increasing k",
            None,
            "The interpolation contract requires ordered wave numbers.",
        ),
        PhysicsCheck(
            "physical units declared",
            True,
            "info",
            "k: Mpc^-1; P: Mpc^3",
            "physical coordinates",
            None,
            "Check solver unit flags before comparing external outputs.",
        ),
    ]


def run_core_physics_validation(
    params: PhysicalCosmology | None = None, *, redshifts: np.ndarray | None = None
) -> PhysicsValidationReport:
    """Run inexpensive equation-level checks for a parameter configuration."""
    p = params or PhysicalCosmology()
    checks = validate_background(p, redshifts) + validate_neutrino_thermodynamics()
    warnings = (
        "This is an internal consistency report, not validation against CAMB/CLASS or observational data.",
        "Neutrino background integrals do not solve perturbation hierarchies or nonlinear free streaming.",
        "Do not interpret this report as an uncertainty on sum_mnu.",
    )
    return PhysicsValidationReport(
        "1.0", "homogeneous_background_plus_thermal_neutrinos", tuple(checks), warnings
    )


def write_validation_report(path: str | Path, report: PhysicsValidationReport) -> Path:
    """Write a sorted JSON report and return the output path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return out


__all__ = [
    "PhysicsCheck",
    "PhysicsValidationReport",
    "validate_background",
    "validate_neutrino_thermodynamics",
    "validate_power_spectrum",
    "run_core_physics_validation",
    "write_validation_report",
]
