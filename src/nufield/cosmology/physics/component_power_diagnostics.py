"""Diagnostics that keep total-matter and cold+baryon power definitions separate.

Massive-neutrino analyses must record which density contrast is being modeled. A total-matter power
spectrum includes neutrino perturbations with their own scale dependence; the CDM+baryon spectrum
(``P_cb``) follows a different response. This module compares spectra supplied by a Boltzmann backend
and never invents a neutrino suppression template.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from nufield.cosmology.physics.parameter_variations import vary_mnu_fixed_cb, vary_mnu_fixed_omega_m
from nufield.cosmology.physics.parameters import PhysicalCosmology

PowerComponent = Literal["total", "cb", "cdm", "baryon", "nu"]


@dataclass(frozen=True, slots=True)
class ComponentPowerComparison:
    """Paired component spectra on a common k grid."""

    k_mpc_inv: np.ndarray
    total_matter_mpc3: np.ndarray
    cb_matter_mpc3: np.ndarray
    total_to_cb_ratio: np.ndarray
    log_ratio: np.ndarray
    redshift: float
    backend: str
    backend_version: str
    parameters: dict[str, object]
    nonlinear: bool
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        k = np.asarray(self.k_mpc_inv, float)
        ptotal = np.asarray(self.total_matter_mpc3, float)
        pcb = np.asarray(self.cb_matter_mpc3, float)
        ratio = np.asarray(self.total_to_cb_ratio, float)
        logratio = np.asarray(self.log_ratio, float)
        if (
            k.ndim != 1
            or k.size < 3
            or any(a.shape != k.shape for a in (ptotal, pcb, ratio, logratio))
        ):
            raise ValueError("component comparison arrays must be matching 1-D arrays")
        if (
            np.any(~np.isfinite(k + ptotal + pcb + ratio + logratio))
            or np.any(k <= 0)
            or np.any(ptotal <= 0)
            or np.any(pcb <= 0)
        ):
            raise ValueError("component comparison arrays must be finite and physical")
        if (
            np.any(np.diff(k) <= 0)
            or np.any(ratio <= 0)
            or not np.allclose(ratio, ptotal / pcb, rtol=1e-7, atol=1e-10)
        ):
            raise ValueError("k grid, ratio, or log-ratio contract violated")
        for name, arr in (
            ("k_mpc_inv", k),
            ("total_matter_mpc3", ptotal),
            ("cb_matter_mpc3", pcb),
            ("total_to_cb_ratio", ratio),
            ("log_ratio", logratio),
        ):
            object.__setattr__(self, name, arr.copy())

    @property
    def maximum_abs_log_ratio(self) -> float:
        return float(np.max(np.abs(self.log_ratio)))

    @property
    def rms_log_ratio(self) -> float:
        return float(np.sqrt(np.mean(self.log_ratio**2)))

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "redshift": self.redshift,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "parameters": self.parameters,
            "nonlinear": self.nonlinear,
            "max_abs_ln_Ptotal_over_Pcb": self.maximum_abs_log_ratio,
            "rms_ln_Ptotal_over_Pcb": self.rms_log_ratio,
            "caveats": list(self.caveats),
            "units": {"k": "Mpc^-1", "power": "Mpc^3"},
        }
        if include_arrays:
            out["k_mpc_inv"] = self.k_mpc_inv.tolist()
            out["total_matter_mpc3"] = self.total_matter_mpc3.tolist()
            out["cb_matter_mpc3"] = self.cb_matter_mpc3.tolist()
            out["total_to_cb_ratio"] = self.total_to_cb_ratio.tolist()
        return out


@dataclass(frozen=True, slots=True)
class NeutrinoMassResponse:
    """Matched mass-variation spectra, preserving the selected parameter-path definition."""

    masses_ev: np.ndarray
    k_mpc_inv: np.ndarray
    power_mpc3: np.ndarray
    reference_mass_ev: float
    reference_index: int
    redshift: float
    component: str
    variation_design: str
    backend: str
    backend_version: str
    base_parameters: dict[str, object]
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        m, k, p = (
            np.asarray(self.masses_ev, float),
            np.asarray(self.k_mpc_inv, float),
            np.asarray(self.power_mpc3, float),
        )
        if m.ndim != 1 or m.size < 2 or np.any(m < 0) or np.any(np.diff(m) <= 0):
            raise ValueError("mass values must be a strictly increasing non-negative vector")
        if k.ndim != 1 or k.size < 3 or np.any(k <= 0) or np.any(np.diff(k) <= 0):
            raise ValueError("k must be a positive increasing vector with at least three samples")
        if p.shape != (m.size, k.size) or np.any(~np.isfinite(p)) or np.any(p <= 0):
            raise ValueError("power_mpc3 shape must be (n_mass, n_k), positive and finite")
        if not (0 <= self.reference_index < len(m)) or not np.isclose(
            m[self.reference_index], self.reference_mass_ev
        ):
            raise ValueError("reference mass must match reference_index")
        if self.component not in {"total", "cb", "cdm", "baryon", "nu"}:
            raise ValueError("unsupported power component")
        if self.variation_design not in {"fixed_omega_cb", "fixed_omega_m", "custom"}:
            raise ValueError("unknown mass-variation design")
        object.__setattr__(self, "masses_ev", m.copy())
        object.__setattr__(self, "k_mpc_inv", k.copy())
        object.__setattr__(self, "power_mpc3", p.copy())

    @property
    def ratio_to_reference(self) -> np.ndarray:
        return self.power_mpc3 / self.power_mpc3[self.reference_index][None, :]

    @property
    def fractional_change_from_reference(self) -> np.ndarray:
        return self.ratio_to_reference - 1.0

    @property
    def log_response_at_mass_nodes(self) -> np.ndarray:
        """Estimate ∂lnP/∂Σmν using actual grid node spacing (units eV⁻¹)."""
        order = 2 if self.masses_ev.size >= 3 else 1
        return np.gradient(np.log(self.power_mpc3), self.masses_ev, axis=0, edge_order=order)

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "reference_mass_ev": self.reference_mass_ev,
            "reference_index": self.reference_index,
            "redshift": self.redshift,
            "component": self.component,
            "variation_design": self.variation_design,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "base_parameters": self.base_parameters,
            "caveats": list(self.caveats),
            "units": {"mass": "eV", "k": "Mpc^-1", "P": "Mpc^3", "dlnP/dM": "eV^-1"},
        }
        if include_arrays:
            out["masses_ev"] = self.masses_ev.tolist()
            out["k_mpc_inv"] = self.k_mpc_inv.tolist()
            out["power_mpc3"] = self.power_mpc3.tolist()
            out["ratio_to_reference"] = self.ratio_to_reference.tolist()
            out["fractional_change_from_reference"] = self.fractional_change_from_reference.tolist()
        return out


def compare_total_and_cb_power(
    backend: Any, params: PhysicalCosmology, z: float, k_mpc_inv: np.ndarray
) -> ComponentPowerComparison:
    """Calculate ``P_total/P_cb`` from a backend exposing ``matter_power_component``.

    The ratio is a comparison of component auto-power spectra. It is not the cross-spectrum needed to
    reconstruct the total-matter power from component spectra, and it is not itself a neutrino-mass
    suppression factor. Both spectra must come from the same cosmology, backend, units, redshift and
    linear/nonlinear convention.
    """
    if not callable(getattr(backend, "matter_power_component", None)):
        raise TypeError("backend must implement matter_power_component(params,z,k,component=...)")
    total = backend.matter_power_component(params, z, k_mpc_inv, component="total")
    cb = backend.matter_power_component(params, z, k_mpc_inv, component="cb")
    if total.linear != cb.linear or total.redshift != cb.redshift:
        raise ValueError("total and cb spectra must share linear/nonlinear mode and redshift")
    lo = max(total.k_mpc_inv[0], cb.k_mpc_inv[0])
    hi = min(total.k_mpc_inv[-1], cb.k_mpc_inv[-1])
    k = np.asarray(k_mpc_inv, float)
    k = k[(k >= lo) & (k <= hi)]
    if k.size < 3:
        raise ValueError("requested grid has fewer than three points in common spectrum support")
    pt, pc = total.evaluate(k), cb.evaluate(k)
    ratio = pt / pc
    return ComponentPowerComparison(
        k,
        pt,
        pc,
        ratio,
        np.log(ratio),
        float(z),
        type(backend).__name__,
        str(getattr(backend, "version", "unknown")),
        params.to_dict(),
        not total.linear,
        (
            "P_total and P_cb are different density-contrast conventions in massive-neutrino cosmology.",
            "Their auto-spectra alone do not determine the cross spectrum between components.",
            "Interpret the ratio only after checking the solver gauge and nonlinear prescriptions.",
        ),
    )


def compute_neutrino_mass_response(
    backend: Any,
    base_parameters: PhysicalCosmology,
    masses_ev: list[float] | np.ndarray,
    z: float,
    k_mpc_inv: np.ndarray,
    *,
    component: PowerComponent = "total",
    variation_design: str = "fixed_omega_cb",
    reference_mass_ev: float | None = None,
) -> NeutrinoMassResponse:
    """Calculate a component spectrum over a mass grid with an explicit held-fixed design.

    ``fixed_omega_cb`` holds Ωb and Ωcdm fixed while total Ωm changes. ``fixed_omega_m`` compensates
    Ωcdm to keep today's Ωm fixed. These are distinct scientific questions and must not be pooled
    without an explicit analysis of parameter degeneracies.
    """
    m = np.asarray(masses_ev, float)
    k = np.asarray(k_mpc_inv, float)
    if (
        m.ndim != 1
        or len(m) < 2
        or np.any(~np.isfinite(m))
        or np.any(m < 0)
        or np.any(np.diff(m) <= 0)
    ):
        raise ValueError(
            "masses_ev must be finite, non-negative, strictly increasing and length >=2"
        )
    if (
        k.ndim != 1
        or len(k) < 3
        or np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError("k_mpc_inv must be positive and increasing with at least 3 points")
    if z < 0 or not math.isfinite(z):
        raise ValueError("z must be finite and non-negative")
    if component not in {"total", "cb", "cdm", "baryon", "nu"}:
        raise ValueError("unsupported component")
    factory: Callable[[PhysicalCosmology, float], PhysicalCosmology]
    if variation_design == "fixed_omega_cb":
        factory = vary_mnu_fixed_cb
    elif variation_design == "fixed_omega_m":
        factory = vary_mnu_fixed_omega_m
    else:
        raise ValueError("variation_design must be fixed_omega_cb or fixed_omega_m")
    compute_component = getattr(backend, "matter_power_component", None)
    if not callable(compute_component):
        raise TypeError("backend must implement matter_power_component")
    rows = []
    for mass in m:
        varied = factory(base_parameters, float(mass))
        spectrum = compute_component(varied, float(z), k, component=component)
        if not spectrum.linear:
            raise ValueError(
                "neutrino response workflow currently requires explicitly linear spectra"
            )
        rows.append(spectrum.evaluate(k))
    target_ref = float(m[0] if reference_mass_ev is None else reference_mass_ev)
    matches = np.flatnonzero(np.isclose(m, target_ref, atol=1e-12, rtol=0))
    if not len(matches):
        raise ValueError("reference_mass_ev must be one of the explicitly computed grid nodes")
    idx = int(matches[0])
    return NeutrinoMassResponse(
        m,
        k,
        np.asarray(rows),
        target_ref,
        idx,
        float(z),
        component,
        variation_design,
        type(backend).__name__,
        str(getattr(backend, "version", "unknown")),
        base_parameters.to_dict(),
        (
            "The response is conditional on the named variation_design.",
            "No real-data posterior is computed; this is theory-response scaffolding.",
            "Response reliability depends on solver accuracy and mass-grid step convergence.",
        ),
    )


__all__ = [
    "ComponentPowerComparison",
    "NeutrinoMassResponse",
    "compare_total_and_cb_power",
    "compute_neutrino_mass_response",
]
