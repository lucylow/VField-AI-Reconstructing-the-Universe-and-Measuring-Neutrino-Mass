"""Auditable parameter mappings for CAMB/CLASS adapters.

The most common cross-solver failure is not a different integrator but a mismatched parameter convention:
H0 versus h, Ωi versus Ωi h², neutrino eigenmasses versus a total mass, effective relativistic species,
unit flags, density variable, gauge, or nonlinear prescription. This module writes a machine-readable
review summary before costly multi-solver campaigns.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from nufield.cosmology.physics.parameters import PhysicalCosmology


@dataclass(frozen=True, slots=True)
class MappingCheck:
    name: str
    passed: bool
    value: float | str | None
    expected: float | str | None
    tolerance: float | None
    message: str


@dataclass(frozen=True, slots=True)
class ParameterMappingAudit:
    parameters: dict[str, object]
    target_neutrino_masses_ev: tuple[float, ...]
    checks: tuple[MappingCheck, ...]
    solver_settings: dict[str, object]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "parameters": self.parameters,
            "target_neutrino_masses_ev": list(self.target_neutrino_masses_ev),
            "checks": [asdict(c) for c in self.checks],
            "solver_settings": self.solver_settings,
            "warnings": list(self.warnings),
            "warning": "Mapping checks do not establish that solver physics models are identical.",
        }


def _close_check(
    name: str, value: float, expected: float, tol: float, message: str
) -> MappingCheck:
    return MappingCheck(
        name,
        bool(np.isfinite(value) and abs(value - expected) <= tol),
        float(value),
        float(expected),
        float(tol),
        message,
    )


def audit_classy_parameter_mapping(
    params: PhysicalCosmology,
    classy_backend: Any,
    *,
    z: float = 0.0,
    k_max_mpc_inv: float = 1.0,
    tolerance: float = 5e-8,
) -> ParameterMappingAudit:
    """Inspect a ClassyPhysicsBackend settings dictionary without running a Boltzmann solve."""
    if not callable(getattr(classy_backend, "settings_for", None)):
        raise TypeError("classy_backend must expose settings_for(params,z,k_max_mpc_inv)")
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    settings = dict(classy_backend.settings_for(params, z, k_max_mpc_inv))
    checks: list[MappingCheck] = []

    def number(key: str, expected: float, label: str) -> None:
        actual = float(settings[key])
        checks.append(
            _close_check(
                label,
                actual,
                expected,
                tolerance * max(1.0, abs(expected)),
                f"CLASS settings key {key} should match the physical parameter schema.",
            )
        )

    number("h", params.h, "Hubble h mapping")
    number("omega_b", params.omega_b_h2, "baryon physical density mapping")
    number("omega_cdm", params.omega_cdm_h2, "CDM physical density mapping")
    number("T_cmb", params.T_cmb_K, "CMB temperature mapping")
    number("A_s", params.A_s, "primordial amplitude mapping")
    number("n_s", params.n_s, "scalar spectral index mapping")
    number("Omega_k", params.omega_k, "curvature mapping")
    expected_masses = params.neutrino_masses_ev if params.sum_mnu_ev > 0 else ()
    raw_masses = settings.get("m_ncdm", "")
    target_masses = tuple(float(v) for v in str(raw_masses).split(",") if v.strip())
    checks.append(
        MappingCheck(
            "massive-neutrino count",
            len(target_masses) == int(settings.get("N_ncdm", 0)),
            len(target_masses),
            int(settings.get("N_ncdm", 0)),
            0,
            "m_ncdm vector length must match N_ncdm.",
        )
    )
    if len(target_masses) != len(expected_masses):
        checks.append(
            MappingCheck(
                "neutrino eigenmass count",
                False,
                len(target_masses),
                len(expected_masses),
                0,
                "CLASS mass vector must represent the selected PhysicalCosmology hierarchy.",
            )
        )
    else:
        for i, (got, expected) in enumerate(zip(target_masses, expected_masses)):
            checks.append(
                _close_check(
                    f"neutrino mass eigenstate {i + 1}",
                    got,
                    float(expected),
                    tolerance * max(1.0, abs(expected)),
                    "Mass ordering must be the same in solver and recorded parameter schema.",
                )
            )
    if expected_masses:
        t_ncdm = tuple(float(v) for v in str(settings.get("T_ncdm", "")).split(",") if v.strip())
        deg = tuple(float(v) for v in str(settings.get("deg_ncdm", "")).split(",") if v.strip())
        if len(t_ncdm) != len(expected_masses) or len(deg) != len(expected_masses):
            checks.append(
                MappingCheck(
                    "ncdm temperature/degeneracy vectors",
                    False,
                    f"{len(t_ncdm)}/{len(deg)}",
                    len(expected_masses),
                    None,
                    "T_ncdm and deg_ncdm must have one entry per massive eigenstate.",
                )
            )
        else:
            # N_eff convention: each ncdm species contributes deg*(T_ncdm/T_standard)^4 to the radiation count.
            t_standard = (4.0 / 11.0) ** (1.0 / 3.0)
            neff_reconstructed = float(settings.get("N_ur", 0.0)) + sum(
                d * (t / t_standard) ** 4 for t, d in zip(t_ncdm, deg)
            )
            checks.append(
                _close_check(
                    "effective relativistic species bookkeeping",
                    neff_reconstructed,
                    params.N_eff,
                    max(1e-5, tolerance * 10),
                    "N_ur plus thermal ncdm contribution should reproduce configured N_eff.",
                )
            )
    else:
        checks.append(
            _close_check(
                "massless effective species mapping",
                float(settings.get("N_ur", -1)),
                params.N_eff,
                max(1e-5, tolerance * 10),
                "With no explicit massive species, N_ur should carry the effective radiation count.",
            )
        )
    safe_settings = {
        str(k): v for k, v in settings.items() if isinstance(v, (str, int, float, bool))
    }
    warnings = (
        "This audit checks parameter bookkeeping, not equivalence of perturbation hierarchies or nonlinear prescriptions.",
        "Check the installed CLASS version's conventions for T_ncdm, N_ur, mass degeneracies and dark-energy settings.",
        "The background parameter schema's effective neutrino energy density is an independent approximation and may not match either solver exactly.",
        "Pin the exact oscillation mass splittings and hierarchy assumptions in any scientific artifact.",
    )
    return ParameterMappingAudit(
        params.to_dict(), tuple(expected_masses), tuple(checks), safe_settings, warnings
    )


def summarize_parameter_conventions(params: PhysicalCosmology) -> dict[str, object]:
    """Create a solver-neutral physical parameter map with explicit units and mass conventions."""
    masses = params.neutrino_masses_ev
    return {
        "H0_km_s_Mpc": params.H0_km_s_Mpc,
        "h_dimensionless": params.h,
        "omega_b_fraction": params.omega_b,
        "omega_cdm_fraction": params.omega_cdm,
        "omega_b_h2": params.omega_b_h2,
        "omega_cdm_h2": params.omega_cdm_h2,
        "sum_mnu_ev": params.sum_mnu_ev,
        "neutrino_hierarchy": params.neutrino_hierarchy,
        "neutrino_mass_eigenvalues_ev": list(masses),
        "N_eff": params.N_eff,
        "n_massive_species": params.n_massive_species,
        "A_s": params.A_s,
        "n_s": params.n_s,
        "k_pivot_Mpc_inv": params.k_pivot_Mpc_inv,
        "length_units": "Mpc",
        "power_units": "Mpc^3",
        "k_units": "Mpc^-1",
        "mass_interpretation": "Σmν is the sum of mass eigenvalues; eigenmass mapping depends on hierarchy and splittings.",
        "parameter_convention_note": "Do not interchange Ω_i and Ω_i h^2 or Mpc and Mpc/h without explicit conversion.",
    }


__all__ = [
    "MappingCheck",
    "ParameterMappingAudit",
    "audit_classy_parameter_mapping",
    "summarize_parameter_conventions",
]
