"""Create a self-contained, auditable theory snapshot from a cosmology backend.

A snapshot combines an explicitly unit-bearing linear spectrum with basic physical validity, σ8-support,
Fourier-domain, BAO-filter, and optional total-vs-cb component diagnostics. It is intended as a preflight
artifact before training or inference, not an automatic scientific acceptance certificate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nufield.cosmology.physics.component_power_diagnostics import compare_total_and_cb_power
from nufield.cosmology.physics.integral_diagnostics import sigma8_support_report
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.physics_validation_suite import validate_power_spectrum
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum
from nufield.cosmology.physics.theory_domain_audit import audit_k_domain
from nufield.cosmology.physics.transfer import matter_response_amplitude


@dataclass(frozen=True, slots=True)
class PhysicsSnapshot:
    schema_version: str
    backend_name: str
    backend_version: str
    redshift: float
    parameters: dict[str, object]
    spectrum: MatterPowerSpectrum
    k_domain: dict[str, object] | None
    sigma8_support: dict[str, object]
    spectrum_checks: tuple[dict[str, object], ...]
    component_comparison: dict[str, object] | None
    transfer_amplitude: dict[str, object]
    caveats: tuple[str, ...]

    def to_dict(self, *, include_spectrum: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "schema_version": self.schema_version,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "redshift": self.redshift,
            "parameters": self.parameters,
            "linear": self.spectrum.linear,
            "spectrum_source": self.spectrum.source,
            "spectrum_metadata": dict(self.spectrum.metadata),
            "sigma8_support": self.sigma8_support,
            "spectrum_checks": list(self.spectrum_checks),
            "k_domain": self.k_domain,
            "component_comparison": self.component_comparison,
            "transfer_amplitude": self.transfer_amplitude,
            "caveats": list(self.caveats),
            "units": {"k": "Mpc^-1", "P": "Mpc^3", "length": "Mpc"},
        }
        if include_spectrum:
            out["spectrum"] = {
                "k_mpc_inv": self.spectrum.k_mpc_inv.tolist(),
                "pk_mpc3": self.spectrum.pk_mpc3.tolist(),
            }
        return out


def build_physics_snapshot(
    backend: Any,
    parameters: PhysicalCosmology,
    redshift: float,
    k_mpc_inv: np.ndarray,
    *,
    box_size_mpc: float | None = None,
    grid_size: int | None = None,
    solver_k_max_mpc_inv: float | None = None,
    declared_linear_k_max_mpc_inv: float | None = None,
    declared_survey_k_max_mpc_inv: float | None = None,
    compare_cb_component: bool = True,
) -> PhysicsSnapshot:
    """Evaluate one theory point and summarize the domain/support needed for reproducible follow-up."""
    k = np.asarray(k_mpc_inv, float)
    if (
        k.ndim != 1
        or len(k) < 8
        or np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError("k_mpc_inv must be finite, positive, increasing and length >=8")
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and non-negative")
    matter_power = getattr(backend, "matter_power", None)
    if not callable(matter_power):
        raise TypeError("backend must implement matter_power(params,z,k)")
    spectrum = matter_power(parameters, float(redshift), k)
    if not spectrum.linear:
        raise ValueError(
            "physics snapshot requires a linear matter spectrum; nonlinear outputs need a separate manifest"
        )
    checks_raw = validate_power_spectrum(spectrum)
    checks = tuple(
        {
            "name": c.name,
            "passed": bool(c.passed),
            "severity": c.severity,
            "value": c.value,
            "expected": c.expected,
            "tolerance": c.tolerance,
            "message": c.message,
        }
        for c in checks_raw
    )
    support = sigma8_support_report(spectrum, parameters.h)
    domain = None
    if box_size_mpc is not None or grid_size is not None:
        if box_size_mpc is None or grid_size is None:
            raise ValueError("box_size_mpc and grid_size must be supplied together")
        domain = audit_k_domain(
            k,
            box_size_mpc=box_size_mpc,
            grid_size=grid_size,
            solver_k_max_mpc_inv=solver_k_max_mpc_inv,
            declared_linear_k_max_mpc_inv=declared_linear_k_max_mpc_inv,
            declared_survey_k_max_mpc_inv=declared_survey_k_max_mpc_inv,
        ).to_dict()
    component = None
    if compare_cb_component and callable(getattr(backend, "matter_power_component", None)):
        try:
            component = compare_total_and_cb_power(backend, parameters, float(redshift), k).to_dict(
                include_arrays=False
            )
        except (ValueError, NotImplementedError) as exc:
            component = {"available": False, "reason": str(exc)}
    kvals, amp = matter_response_amplitude(spectrum, parameters)
    transfer = {
        "k_mpc_inv": kvals.tolist(),
        "abs_M_k": amp.tolist(),
        "definition": "|M(k,z)| inferred from P_m=(2π²/k³) Δ_R² |M|²",
        "caveats": [
            "This is an inferred amplitude, not a signed/gauge-specific Boltzmann transfer variable.",
            "The density component convention must match the power spectrum definition.",
        ],
    }
    return PhysicsSnapshot(
        "1.0",
        type(backend).__name__,
        str(getattr(backend, "version", "unknown")),
        float(redshift),
        parameters.to_dict(),
        spectrum,
        domain,
        support,
        checks,
        component,
        transfer,
        (
            "This artifact is a theory snapshot, not a posterior or measurement of Σmν.",
            "The default k-domain audit is only active when box/grid sizes are supplied.",
            "Solver version, accuracy, neutrino hierarchy, component variable and unit flags must be preserved.",
            "Internal checks do not establish agreement with another Boltzmann solver or observed survey data.",
        ),
    )


def write_physics_snapshot(
    path: str | Path, snapshot: PhysicsSnapshot, *, include_spectrum_in_json: bool = False
) -> Path:
    """Write a human-readable JSON snapshot; the caller may save raw spectrum arrays separately."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            snapshot.to_dict(include_spectrum=include_spectrum_in_json), indent=2, sort_keys=True
        )
        + "\n"
    )
    return output


def save_snapshot_spectrum(path: str | Path, snapshot: PhysicsSnapshot) -> Path:
    """Save spectrum arrays with metadata in an adjacent JSON sidecar."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        k_mpc_inv=snapshot.spectrum.k_mpc_inv,
        pk_mpc3=snapshot.spectrum.pk_mpc3,
        redshift=np.asarray(snapshot.redshift),
        linear=np.asarray(True),
    )
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "backend_name": snapshot.backend_name,
                "backend_version": snapshot.backend_version,
                "parameters": snapshot.parameters,
                "spectrum_source": snapshot.spectrum.source,
                "spectrum_metadata": dict(snapshot.spectrum.metadata),
                "units": {"k": "Mpc^-1", "P": "Mpc^3"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output


__all__ = [
    "PhysicsSnapshot",
    "build_physics_snapshot",
    "write_physics_snapshot",
    "save_snapshot_spectrum",
]
