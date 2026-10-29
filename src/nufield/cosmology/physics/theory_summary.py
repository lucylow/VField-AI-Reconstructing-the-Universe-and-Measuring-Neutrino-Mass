"""Machine-readable scientific scope and derived-quantity report helpers."""

from __future__ import annotations

import json
from pathlib import Path

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.distances import (
    angular_diameter_distance_mpc,
    luminosity_distance_mpc,
    radial_comoving_distance_mpc,
    transverse_comoving_distance_mpc,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.validity import PhysicsScope


def background_summary(
    params: PhysicalCosmology, redshifts: list[float] | tuple[float, ...]
) -> dict[str, object]:
    """Return reproducible background table including model and convention metadata."""
    bg = BackgroundCosmology(params)
    rows = []
    for z in redshifts:
        rows.append(
            {
                "z": float(z),
                "E": float(bg.E(z)),
                "H_km_s_Mpc": float(bg.H(z)),
                "D_C_Mpc": radial_comoving_distance_mpc(bg, float(z)),
                "D_M_Mpc": transverse_comoving_distance_mpc(bg, float(z)),
                "D_A_Mpc": angular_diameter_distance_mpc(bg, float(z)),
                "D_L_Mpc": luminosity_distance_mpc(bg, float(z)),
                "density_fractions": bg.density_fractions(float(z)),
            }
        )
    return {
        "schema_version": "1.0",
        "model": "homogeneous_cpl_background",
        "parameters": params.to_dict(),
        "derived": {
            "omega_gamma0": bg.omega_gamma0,
            "omega_nu_rel0": bg.omega_nu_rel0,
            "omega_nu_massive0": bg.omega_nu_massive0,
            "omega_m0": bg.omega_m0,
            "omega_de0": bg.omega_de0,
        },
        "redshift_table": rows,
        "validity": [
            "Background expansion only; no perturbation evolution.",
            "Massive-neutrino homogeneous Fermi-Dirac background is included.",
            "For precision distances, compare against CAMB/CLASS using matched conventions.",
        ],
    }


def write_background_summary(
    path: str | Path, params: PhysicalCosmology, redshifts: list[float] | tuple[float, ...]
) -> Path:
    """Write a sorted, indented JSON report; return its final path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(background_summary(params, redshifts), indent=2, sort_keys=True) + "\n"
    )
    return out


def physics_scope_record(scope: PhysicsScope) -> dict[str, object]:
    """Serialize model capabilities and notes without upgrading its physics level automatically."""
    return {
        "level": scope.level.value,
        "neutrino_perturbations": scope.neutrino_perturbations,
        "nonlinear_gravity": scope.nonlinear_gravity,
        "survey_window": scope.survey_window,
        "observational_noise": scope.observational_noise,
        "validated_against_reference": scope.validated_against_reference,
        "notes": list(scope.notes),
    }


__all__ = ["background_summary", "write_background_summary", "physics_scope_record"]
