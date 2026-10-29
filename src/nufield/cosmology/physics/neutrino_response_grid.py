"""Neutrino-mass response grids built from explicit theory-backend spectra.

This module is intentionally backend-driven: it does not invent a neutrino suppression template. All
responses are derived from a grid of spectra calculated by CAMB, CLASS, or another registered theory
backend. Since varying Σmν changes interpretation depending on what else is held fixed, every grid
must declare its parameter-variation design. Grids using different designs must not be pooled without
an explicit analysis of the differing parameter path through cosmological parameter space.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from nufield.cosmology.physics.parameter_variations import vary_mnu_fixed_cb, vary_mnu_fixed_omega_m
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class ResponseDerivativeDiagnostics:
    step_ev: float
    coarse_step_ev: float
    rms_log_derivative_difference: float
    max_abs_log_derivative_difference: float
    passed: bool
    tolerance: float


@dataclass(frozen=True, slots=True)
class NeutrinoResponseGrid:
    """Power-spectrum grid with array axes (sum_mnu, redshift, k)."""

    masses_ev: np.ndarray
    redshifts: np.ndarray
    k_mpc_inv: np.ndarray
    pk_mpc3: np.ndarray
    variation_design: str
    backend_name: str
    backend_version: str
    base_parameters: dict[str, object]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        masses = np.asarray(self.masses_ev, float)
        zs = np.asarray(self.redshifts, float)
        k = np.asarray(self.k_mpc_inv, float)
        p = np.asarray(self.pk_mpc3, float)
        if (
            masses.ndim != 1
            or masses.size < 2
            or np.any(masses < 0)
            or np.any(np.diff(masses) <= 0)
        ):
            raise ValueError(
                "masses_ev must be a strictly increasing vector with at least two masses"
            )
        if zs.ndim != 1 or zs.size < 1 or np.any(zs < 0) or np.any(np.diff(zs) <= 0):
            raise ValueError("redshifts must be a strictly increasing non-negative vector")
        if k.ndim != 1 or k.size < 3 or np.any(k <= 0) or np.any(np.diff(k) <= 0):
            raise ValueError("k must be a positive increasing vector with at least 3 samples")
        if p.shape != (masses.size, zs.size, k.size) or np.any(~np.isfinite(p)) or np.any(p <= 0):
            raise ValueError("pk_mpc3 must be positive and shaped (n_mass, n_z, n_k)")
        if self.variation_design not in {"fixed_omega_cb", "fixed_omega_m", "custom"}:
            raise ValueError("variation_design must be fixed_omega_cb, fixed_omega_m, or custom")
        if not self.backend_name or not self.backend_version:
            raise ValueError("backend name and version must be supplied")
        object.__setattr__(self, "masses_ev", masses.copy())
        object.__setattr__(self, "redshifts", zs.copy())
        object.__setattr__(self, "k_mpc_inv", k.copy())
        object.__setattr__(self, "pk_mpc3", p.copy())

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.pk_mpc3.shape

    def spectrum(self, mass_index: int, redshift_index: int) -> MatterPowerSpectrum:
        """Materialize one grid plane as a unit-bearing spectrum object."""
        return MatterPowerSpectrum(
            self.k_mpc_inv,
            self.pk_mpc3[mass_index, redshift_index],
            float(self.redshifts[redshift_index]),
            linear=True,
            source=f"{self.backend_name} {self.backend_version}",
            metadata=(
                ("variation_design", self.variation_design),
                ("sum_mnu_ev", repr(float(self.masses_ev[mass_index]))),
            ),
        )

    def fractional_suppression(self, reference_mass_ev: float, *, redshift: float) -> np.ndarray:
        """Return 1-P(k, mass)/P(k, reference_mass) interpolated across the grid mass axis."""
        p_ref = self.power_at_mass(reference_mass_ev, redshift=redshift)
        return 1.0 - self.pk_mpc3_at_redshift(redshift) / p_ref[None, :]

    def pk_mpc3_at_redshift(self, redshift: float) -> np.ndarray:
        """Interpolate ln P in redshift at each mass/k; no extrapolation is permitted."""
        z = float(redshift)
        if z < self.redshifts[0] or z > self.redshifts[-1]:
            raise ValueError("requested redshift outside response grid support")
        if self.redshifts.size == 1:
            return self.pk_mpc3[:, 0, :].copy()
        out = np.empty((self.masses_ev.size, self.k_mpc_inv.size))
        for i in range(self.masses_ev.size):
            for j in range(self.k_mpc_inv.size):
                interp = PchipInterpolator(
                    self.redshifts, np.log(self.pk_mpc3[i, :, j]), extrapolate=False
                )
                out[i, j] = np.exp(float(interp(z)))
        return out

    def power_at_mass(self, mass_ev: float, *, redshift: float) -> np.ndarray:
        """Interpolate the grid in mass using PCHIP of ln P; never extrapolate."""
        mass = float(mass_ev)
        if mass < self.masses_ev[0] or mass > self.masses_ev[-1]:
            raise ValueError("requested neutrino mass lies outside response grid support")
        pz = self.pk_mpc3_at_redshift(redshift)
        values = np.empty(self.k_mpc_inv.size)
        for j in range(self.k_mpc_inv.size):
            interp = PchipInterpolator(self.masses_ev, np.log(pz[:, j]), extrapolate=False)
            values[j] = np.exp(float(interp(mass)))
        return values

    def dlnp_dmass(self, *, redshift: float, edge_order: int = 2) -> np.ndarray:
        """Return finite-difference ∂lnP/∂Σmν [eV⁻¹] on the grid mass nodes."""
        pz = self.pk_mpc3_at_redshift(redshift)
        if edge_order not in (1, 2):
            raise ValueError("edge_order must be 1 or 2")
        effective_order = min(edge_order, self.masses_ev.size - 1)
        return np.gradient(np.log(pz), self.masses_ev, axis=0, edge_order=effective_order)

    def dlnp_dlnk(self, *, redshift: float) -> np.ndarray:
        """Return ∂lnP/∂lnk for every mass at a given redshift."""
        pz = self.pk_mpc3_at_redshift(redshift)
        return np.gradient(np.log(pz), np.log(self.k_mpc_inv), axis=1, edge_order=2)

    def to_manifest(self) -> dict[str, object]:
        """Return metadata without embedding large numerical arrays."""
        return {
            "schema_version": "1.0",
            "shape": list(self.shape),
            "variation_design": self.variation_design,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "base_parameters": self.base_parameters,
            "masses_ev": self.masses_ev.tolist(),
            "redshifts": self.redshifts.tolist(),
            "k_range_mpc_inv": [float(self.k_mpc_inv[0]), float(self.k_mpc_inv[-1])],
            "n_k": int(self.k_mpc_inv.size),
            "metadata": dict(self.metadata),
            "units": {"mass": "eV", "k": "Mpc^-1", "P": "Mpc^3"},
        }

    def save(self, path: str | Path) -> Path:
        """Store compressed arrays and a JSON manifest adjacent to the archive."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            masses_ev=self.masses_ev,
            redshifts=self.redshifts,
            k_mpc_inv=self.k_mpc_inv,
            pk_mpc3=self.pk_mpc3,
        )
        out.with_suffix(out.suffix + ".json").write_text(
            json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n"
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> NeutrinoResponseGrid:
        """Load a grid archive together with the required provenance manifest."""
        pth = Path(path)
        manifest_path = pth.with_suffix(pth.suffix + ".json")
        if not pth.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                "grid archive and adjacent .json provenance manifest are both required"
            )
        manifest = json.loads(manifest_path.read_text())
        with np.load(pth, allow_pickle=False) as data:
            return cls(
                data["masses_ev"],
                data["redshifts"],
                data["k_mpc_inv"],
                data["pk_mpc3"],
                str(manifest["variation_design"]),
                str(manifest["backend_name"]),
                str(manifest["backend_version"]),
                dict(manifest["base_parameters"]),
                tuple(sorted((str(k), str(v)) for k, v in manifest.get("metadata", {}).items())),
            )


def build_neutrino_response_grid(
    backend,
    base_parameters: PhysicalCosmology,
    masses_ev: Sequence[float],
    redshifts: Sequence[float],
    k_mpc_inv: np.ndarray,
    *,
    variation_design: str = "fixed_omega_cb",
    metadata: dict[str, str] | None = None,
) -> NeutrinoResponseGrid:
    """Compute P(k,z,Σmν) using a backend, with explicit fixed-parameter semantics.

    The operation may be expensive because each mass/redshift pair can require a solver call. It is a
    transparent correctness-first implementation; use a content-addressed cache or approved emulator
    after establishing agreement with direct calculations. The function does not silently parallelize
    jobs because thread/process safety differs across external Boltzmann solvers.
    """
    masses = np.asarray(masses_ev, float)
    zs = np.asarray(redshifts, float)
    ks = np.asarray(k_mpc_inv, float)
    # Reuse dataclass validators with a temporary dummy array after validation of coordinates.
    if masses.ndim != 1 or len(masses) < 2 or np.any(masses < 0) or np.any(np.diff(masses) <= 0):
        raise ValueError("masses_ev must be strictly increasing, non-negative, with >=2 values")
    if zs.ndim != 1 or len(zs) < 1 or np.any(zs < 0) or np.any(np.diff(zs) <= 0):
        raise ValueError("redshifts must be increasing and non-negative")
    if ks.ndim != 1 or len(ks) < 3 or np.any(ks <= 0) or np.any(np.diff(ks) <= 0):
        raise ValueError("k_mpc_inv must be positive and increasing with >=3 samples")
    factories: dict[str, Callable[[PhysicalCosmology, float], PhysicalCosmology]] = {
        "fixed_omega_cb": vary_mnu_fixed_cb,
        "fixed_omega_m": vary_mnu_fixed_omega_m,
    }
    if variation_design not in factories:
        raise ValueError("variation_design must be fixed_omega_cb or fixed_omega_m")
    result = np.empty((len(masses), len(zs), len(ks)), float)
    factory = factories[variation_design]
    for im, mass in enumerate(masses):
        p = factory(base_parameters, float(mass))
        for iz, z in enumerate(zs):
            spectrum = backend.matter_power(p, float(z), ks)
            if not np.isclose(spectrum.redshift, z, atol=1e-10, rtol=0):
                raise ValueError("backend returned a spectrum at a different redshift")
            if not spectrum.linear:
                raise ValueError(
                    "response-grid builder expects linear spectra; nonlinear grids need an explicit mode"
                )
            if ks[0] < spectrum.k_mpc_inv[0] or ks[-1] > spectrum.k_mpc_inv[-1]:
                raise ValueError("backend spectrum does not cover requested k support")
            result[im, iz, :] = spectrum.evaluate(ks)
    return NeutrinoResponseGrid(
        masses,
        zs,
        ks,
        result,
        variation_design,
        type(backend).__name__,
        str(getattr(backend, "version", "unknown")),
        base_parameters.to_dict(),
        tuple(sorted((metadata or {}).items())),
    )


def derivative_step_stability(
    masses_ev: np.ndarray,
    log_power_at_masses: np.ndarray,
    *,
    center_mass_ev: float,
    fine_step_ev: float,
    coarse_step_ev: float,
    tolerance: float = 0.1,
) -> ResponseDerivativeDiagnostics:
    """Compare central finite-difference response estimates at two step sizes.

    Arrays must contain a mass axis first and may carry arbitrary trailing dimensions. The function
    linearly interpolates lnP at centre±step only within the provided grid; it never extrapolates.
    The comparison is a diagnostic, not proof that either derivative is converged.
    """
    masses = np.asarray(masses_ev, float)
    lp = np.asarray(log_power_at_masses, float)
    if masses.ndim != 1 or lp.shape[0] != masses.size or len(masses) < 3:
        raise ValueError("log_power_at_masses must have the mass axis first")
    if np.any(np.diff(masses) <= 0) or np.any(~np.isfinite(lp)):
        raise ValueError("masses must increase and log-power values be finite")
    if fine_step_ev <= 0 or coarse_step_ev <= fine_step_ev or tolerance <= 0:
        raise ValueError("require 0 < fine_step < coarse_step and positive tolerance")
    if center_mass_ev - coarse_step_ev < masses[0] or center_mass_ev + coarse_step_ev > masses[-1]:
        raise ValueError("mass grid does not cover both finite-difference stencils")

    def interp(m: float) -> np.ndarray:
        flat = lp.reshape(masses.size, -1)
        out = np.array([np.interp(m, masses, flat[:, j]) for j in range(flat.shape[1])])
        return out.reshape(lp.shape[1:])

    fine = (interp(center_mass_ev + fine_step_ev) - interp(center_mass_ev - fine_step_ev)) / (
        2 * fine_step_ev
    )
    coarse = (interp(center_mass_ev + coarse_step_ev) - interp(center_mass_ev - coarse_step_ev)) / (
        2 * coarse_step_ev
    )
    diff = fine - coarse
    denom = np.maximum(np.abs(fine), np.median(np.abs(fine)) * 1e-3 + 1e-12)
    frac = np.abs(diff) / denom
    rms = float(np.sqrt(np.mean(frac**2)))
    maximum = float(np.max(frac))
    return ResponseDerivativeDiagnostics(
        float(fine_step_ev),
        float(coarse_step_ev),
        rms,
        maximum,
        bool(maximum <= tolerance),
        float(tolerance),
    )


__all__ = [
    "ResponseDerivativeDiagnostics",
    "NeutrinoResponseGrid",
    "build_neutrino_response_grid",
    "derivative_step_stability",
]
