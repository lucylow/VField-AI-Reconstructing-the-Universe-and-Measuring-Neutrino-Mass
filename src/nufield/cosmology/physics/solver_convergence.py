"""Accuracy and grid-convergence campaigns for deterministic cosmology theory backends.

A single solver run is not a convergence test. This module runs an explicit list of settings, evaluates
all spectra on common support, and reports changes in ln P(k). It does not prescribe universally valid
tolerances: the appropriate numerical accuracy depends on the target observable and desired parameter
precision. Solver setup, package version, and variable conventions must be saved beside every result.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class ConvergenceLevel:
    level_name: str
    backend_name: str
    backend_version: str
    accuracy_value: float
    fractional_rms_vs_previous: float | None
    log_rms_vs_previous: float | None
    log_max_abs_vs_previous: float | None
    n_compared_points: int
    spectrum: MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class SolverConvergenceReport:
    parameters: dict[str, object]
    redshift: float
    k_range_mpc_inv: tuple[float, float]
    levels: tuple[ConvergenceLevel, ...]
    final_log_rms_change: float | None
    final_log_max_change: float | None
    accepted_by_user_thresholds: bool | None
    thresholds: dict[str, float]
    caveats: tuple[str, ...]

    def to_dict(self, *, include_spectra: bool = False) -> dict[str, object]:
        levels: list[dict[str, object]] = []
        for level in self.levels:
            row = {k: v for k, v in asdict(level).items() if k != "spectrum"}
            if include_spectra:
                row["spectrum"] = {
                    "k_mpc_inv": level.spectrum.k_mpc_inv.tolist(),
                    "pk_mpc3": level.spectrum.pk_mpc3.tolist(),
                    "linear": level.spectrum.linear,
                    "metadata": dict(level.spectrum.metadata),
                }
            levels.append(row)
        return {
            "parameters": self.parameters,
            "redshift": self.redshift,
            "k_range_mpc_inv": list(self.k_range_mpc_inv),
            "levels": levels,
            "final_log_rms_change": self.final_log_rms_change,
            "final_log_max_change": self.final_log_max_change,
            "accepted_by_user_thresholds": self.accepted_by_user_thresholds,
            "thresholds": self.thresholds,
            "caveats": list(self.caveats),
        }


def _on_common_grid(
    reference: MatterPowerSpectrum, candidate: MatterPowerSpectrum, n_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference.redshift != candidate.redshift or reference.linear != candidate.linear:
        raise ValueError("convergence levels must share redshift and linear/nonlinear setting")
    lo = max(reference.k_mpc_inv[0], candidate.k_mpc_inv[0])
    hi = min(reference.k_mpc_inv[-1], candidate.k_mpc_inv[-1])
    if hi <= lo or hi / lo < 1.01:
        raise ValueError("successive spectra lack sufficient common k support")
    k = np.geomspace(lo, hi, n_points)
    p0, p1 = reference.evaluate(k), candidate.evaluate(k)
    return k, p0, p1


def run_solver_convergence_campaign(
    backend_factory: Callable[[float], object],
    parameters: PhysicalCosmology,
    redshift: float,
    k_mpc_inv: np.ndarray,
    accuracy_values: Sequence[float],
    *,
    level_names: Sequence[str] | None = None,
    max_log_rms: float | None = None,
    max_log_abs: float | None = None,
) -> SolverConvergenceReport:
    """Evaluate a backend factory across increasing explicit accuracy values.

    ``backend_factory(accuracy)`` must return an object with ``matter_power(params,z,k)`` and optionally
    ``version``. Accuracy values must be strictly increasing; no implicit default or silent reordering
    is allowed. The report compares each rung to the immediately preceding rung.
    """
    acc = np.asarray(accuracy_values, float)
    k = np.asarray(k_mpc_inv, float)
    if (
        acc.ndim != 1
        or len(acc) < 2
        or np.any(~np.isfinite(acc))
        or np.any(acc <= 0)
        or np.any(np.diff(acc) <= 0)
    ):
        raise ValueError(
            "accuracy_values must be positive, finite, strictly increasing and length >=2"
        )
    if (
        k.ndim != 1
        or len(k) < 8
        or np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError("k_mpc_inv must be positive, increasing and length >=8")
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and non-negative")
    if (max_log_rms is not None and (not np.isfinite(max_log_rms) or max_log_rms <= 0)) or (
        max_log_abs is not None and (not np.isfinite(max_log_abs) or max_log_abs <= 0)
    ):
        raise ValueError("convergence thresholds must be positive and finite")
    names = tuple(level_names or [f"accuracy_{x:g}" for x in acc])
    if len(names) != len(acc) or len(set(names)) != len(names):
        raise ValueError("level_names must be unique and match accuracy_values")
    levels: list[ConvergenceLevel] = []
    previous: MatterPowerSpectrum | None = None
    for name, value in zip(names, acc):
        backend = backend_factory(float(value))
        spectrum = backend.matter_power(parameters, float(redshift), k)
        if not spectrum.linear:
            raise ValueError(
                "solver convergence campaign expects identical linear spectrum settings"
            )
        if not np.isclose(spectrum.redshift, redshift, atol=1e-10, rtol=0):
            raise ValueError("backend returned an unexpected redshift")
        if previous is None:
            rms = maxabs = frac_rms = None
            n_points = 0
        else:
            _, p0, p1 = _on_common_grid(previous, spectrum, min(len(k), 1024))
            diff = np.log(p1) - np.log(p0)
            rms = float(np.sqrt(np.mean(diff**2)))
            maxabs = float(np.max(np.abs(diff)))
            frac_rms = float(np.sqrt(np.mean(np.expm1(diff) ** 2)))
            n_points = len(diff)
        levels.append(
            ConvergenceLevel(
                name,
                type(backend).__name__,
                str(getattr(backend, "version", "unknown")),
                float(value),
                frac_rms,
                rms,
                maxabs,
                n_points,
                spectrum,
            )
        )
        previous = spectrum
    last = levels[-1]
    accepted: bool | None
    if max_log_rms is None and max_log_abs is None:
        accepted = None
    else:
        accepted = bool(
            (
                max_log_rms is None
                or (
                    last.log_rms_vs_previous is not None and last.log_rms_vs_previous <= max_log_rms
                )
            )
            and (
                max_log_abs is None
                or (
                    last.log_max_abs_vs_previous is not None
                    and last.log_max_abs_vs_previous <= max_log_abs
                )
            )
        )
    return SolverConvergenceReport(
        parameters.to_dict(),
        float(redshift),
        (float(k[0]), float(k[-1])),
        tuple(levels),
        last.log_rms_vs_previous,
        last.log_max_abs_vs_previous,
        accepted,
        {
            kname: v
            for kname, v in (("max_log_rms", max_log_rms), ("max_log_abs", max_log_abs))
            if v is not None
        },
        (
            "Numerical convergence is conditional on this chosen parameter point, redshift, k support and observable.",
            "Changing accuracy settings can also change interpolation grids; comparisons are made on a common physical-k grid.",
            "A converged solver can still implement the wrong model or parameter mapping.",
        ),
    )


def write_solver_convergence_report(
    path: str | Path, report: SolverConvergenceReport, *, include_spectra: bool = False
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(include_spectra=include_spectra), indent=2, sort_keys=True) + "\n"
    )
    return out


__all__ = [
    "ConvergenceLevel",
    "SolverConvergenceReport",
    "run_solver_convergence_campaign",
    "write_solver_convergence_report",
]
