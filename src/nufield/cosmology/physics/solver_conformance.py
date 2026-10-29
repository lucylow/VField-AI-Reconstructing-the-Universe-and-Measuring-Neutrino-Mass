"""Cross-solver, accuracy-convergence and output-contract tests for cosmological theory backends.

A successful comparison on a single spectrum is not sufficient to validate a solver mapping. This
module creates durable reports over multiple redshifts and wavenumber ranges, makes unit assumptions
explicit, and separates numerical convergence from physical-model agreement. Backend comparisons
must match species treatment, neutrino hierarchy, nonlinear prescription, transfer variable, gauge
where relevant, and k/P units before interpreting a residual as an accuracy result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


class SpectrumBackend(Protocol):
    """Minimum interface consumed by the conformance harness."""

    @property
    def version(self) -> str: ...
    def matter_power(
        self, params: PhysicalCosmology, z: float, k_mpc_inv: np.ndarray
    ) -> MatterPowerSpectrum: ...


@dataclass(frozen=True, slots=True)
class SpectrumAgreement:
    """Summary of a comparison on the sampled common support."""

    redshift: float
    k_min_mpc_inv: float
    k_max_mpc_inv: float
    n_samples: int
    rms_fractional_error: float
    median_absolute_fractional_error: float
    p95_absolute_fractional_error: float
    max_absolute_fractional_error: float
    rms_log_power_difference: float
    max_abs_log_power_difference: float
    tolerance: float
    passed: bool
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RedshiftComparison:
    """Spectral arrays and metrics for one redshift."""

    redshift: float
    k_mpc_inv: np.ndarray
    reference_pk_mpc3: np.ndarray
    candidate_pk_mpc3: np.ndarray
    fractional_error: np.ndarray
    log_power_difference: np.ndarray
    agreement: SpectrumAgreement

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        row: dict[str, object] = {"redshift": self.redshift, "agreement": self.agreement.to_dict()}
        if include_arrays:
            row.update(
                {
                    "k_mpc_inv": self.k_mpc_inv.tolist(),
                    "reference_pk_mpc3": self.reference_pk_mpc3.tolist(),
                    "candidate_pk_mpc3": self.candidate_pk_mpc3.tolist(),
                    "fractional_error": self.fractional_error.tolist(),
                    "log_power_difference": self.log_power_difference.tolist(),
                }
            )
        return row


def compare_spectra(
    reference: MatterPowerSpectrum,
    candidate: MatterPowerSpectrum,
    *,
    tolerance: float = 0.01,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
    require_same_linearity: bool = True,
) -> RedshiftComparison:
    """Compare spectra using log interpolation, never extrapolating outside common support.

    The candidate is evaluated at reference samples that fall within the shared interval. The main
    pass/fail metric is maximum absolute fractional power difference. RMS log difference is included
    because it is symmetric under swapping reference and candidate for positive spectra.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isclose(reference.redshift, candidate.redshift, rel_tol=0, abs_tol=1e-10):
        raise ValueError("spectra must have matching redshifts")
    if require_same_linearity and reference.linear != candidate.linear:
        raise ValueError("cannot compare linear and nonlinear spectra as a solver-agreement test")
    lo = max(
        float(reference.k_mpc_inv[0]),
        float(candidate.k_mpc_inv[0]),
        float(k_min_mpc_inv) if k_min_mpc_inv is not None else 0.0,
    )
    hi = min(
        float(reference.k_mpc_inv[-1]),
        float(candidate.k_mpc_inv[-1]),
        float(k_max_mpc_inv) if k_max_mpc_inv is not None else float("inf"),
    )
    if hi <= lo:
        raise ValueError("requested spectra do not overlap in k")
    mask = (reference.k_mpc_inv >= lo) & (reference.k_mpc_inv <= hi)
    k = reference.k_mpc_inv[mask]
    if k.size < 3:
        raise ValueError("at least three reference samples must lie in common k support")
    p_ref = reference.evaluate(k)
    p_cand = candidate.evaluate(k)
    if np.any(p_ref <= 0) or np.any(p_cand <= 0):
        raise ValueError("positive power spectra are required for log-error metrics")
    frac = p_cand / p_ref - 1.0
    logdiff = np.log(p_cand) - np.log(p_ref)
    metrics = SpectrumAgreement(
        redshift=float(reference.redshift),
        k_min_mpc_inv=float(k.min()),
        k_max_mpc_inv=float(k.max()),
        n_samples=int(k.size),
        rms_fractional_error=float(np.sqrt(np.mean(frac**2))),
        median_absolute_fractional_error=float(np.median(np.abs(frac))),
        p95_absolute_fractional_error=float(np.quantile(np.abs(frac), 0.95)),
        max_absolute_fractional_error=float(np.max(np.abs(frac))),
        rms_log_power_difference=float(np.sqrt(np.mean(logdiff**2))),
        max_abs_log_power_difference=float(np.max(np.abs(logdiff))),
        tolerance=float(tolerance),
        passed=bool(np.max(np.abs(frac)) <= tolerance),
        caveats=(
            "Numerical agreement is meaningful only after verifying matching cosmological conventions.",
            "This comparison tests P(k), not transfer-function gauge choices or survey observables.",
        ),
    )
    return RedshiftComparison(
        float(reference.redshift), k.copy(), p_ref, p_cand, frac, logdiff, metrics
    )


@dataclass(frozen=True, slots=True)
class BackendConformanceReport:
    """Aggregate solver comparison manifest over redshift and k."""

    schema_version: str
    reference_backend: str
    candidate_backend: str
    reference_version: str
    candidate_version: str
    parameter_manifest: dict[str, object]
    k_requested_mpc_inv: tuple[float, ...]
    redshifts: tuple[float, ...]
    tolerance: float
    all_passed: bool
    comparisons: tuple[RedshiftComparison, ...]
    warnings: tuple[str, ...]

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "reference_version": self.reference_version,
            "candidate_version": self.candidate_version,
            "parameter_manifest": self.parameter_manifest,
            "k_requested_mpc_inv": list(self.k_requested_mpc_inv),
            "redshifts": list(self.redshifts),
            "tolerance": self.tolerance,
            "all_passed": self.all_passed,
            "comparisons": [c.to_dict(include_arrays=include_arrays) for c in self.comparisons],
            "warnings": list(self.warnings),
        }


def compare_backends(
    reference: SpectrumBackend,
    candidate: SpectrumBackend,
    params: PhysicalCosmology,
    *,
    redshifts: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    k_mpc_inv: np.ndarray | None = None,
    tolerance: float = 0.01,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
) -> BackendConformanceReport:
    """Run same-parameter multi-z spectrum comparisons between two backend implementations."""
    zs = tuple(float(z) for z in redshifts)
    if not zs or any(not math.isfinite(z) or z < 0 for z in zs):
        raise ValueError("redshifts must be a non-empty sequence of finite non-negative values")
    if len(set(zs)) != len(zs):
        raise ValueError("redshifts must be unique")
    kvals = np.geomspace(1e-3, 1.0, 256) if k_mpc_inv is None else np.asarray(k_mpc_inv, float)
    if kvals.ndim != 1 or kvals.size < 3 or np.any(kvals <= 0) or np.any(np.diff(kvals) <= 0):
        raise ValueError("k_mpc_inv must be a positive increasing vector with >=3 values")
    rows: list[RedshiftComparison] = []
    for z in zs:
        p_ref = reference.matter_power(params, z, kvals)
        p_cand = candidate.matter_power(params, z, kvals)
        if p_ref.source == "unspecified" or p_cand.source == "unspecified":
            # Do not fail solely on generic source metadata, but retain the warning below.
            pass
        rows.append(
            compare_spectra(
                p_ref,
                p_cand,
                tolerance=tolerance,
                k_min_mpc_inv=k_min_mpc_inv,
                k_max_mpc_inv=k_max_mpc_inv,
            )
        )
    warnings = [
        "Matching parameter values does not guarantee matching neutrino or dark-energy implementations.",
        "Record backend defaults, precision settings, massive species, gauge/transfer definitions and nonlinear prescriptions.",
        "The pass threshold is a project-specific numerical gate, not a universal cosmological accuracy standard.",
    ]
    if reference is candidate:
        warnings.append(
            "Reference and candidate are the same backend object; this is a self-consistency test only."
        )
    return BackendConformanceReport(
        "1.0",
        type(reference).__name__,
        type(candidate).__name__,
        str(reference.version),
        str(candidate.version),
        params.to_dict(),
        tuple(float(x) for x in kvals),
        zs,
        float(tolerance),
        all(row.agreement.passed for row in rows),
        tuple(rows),
        tuple(warnings),
    )


def write_conformance_report(
    path: str | Path, report: BackendConformanceReport, *, include_arrays: bool = False
) -> Path:
    """Serialize a solver-comparison report with stable key ordering."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            report.to_dict(include_arrays=include_arrays), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    return out


@dataclass(frozen=True, slots=True)
class AccuracyConvergenceRow:
    setting: float
    rms_fractional_from_highest: float
    max_fractional_from_highest: float
    passed: bool


def accuracy_convergence_study(
    compute: Callable[[float], MatterPowerSpectrum],
    settings: Sequence[float],
    *,
    tolerance: float = 0.005,
) -> tuple[AccuracyConvergenceRow, ...]:
    """Test numerical convergence as an accuracy setting is increased.

    ``compute`` should deterministically produce spectra with identical cosmological parameters,
    units, redshift, k support and physics options. The largest setting is the numerical reference;
    monotonic convergence is not required because numerical solvers can have non-monotonic errors.
    """
    vals = tuple(float(x) for x in settings)
    if len(vals) < 2 or any(not math.isfinite(x) or x <= 0 for x in vals):
        raise ValueError("at least two positive finite accuracy settings are required")
    if len(set(vals)) != len(vals):
        raise ValueError("accuracy settings must be unique")
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    computed = {s: compute(s) for s in sorted(vals)}
    reference = computed[max(vals)]
    rows: list[AccuracyConvergenceRow] = []
    for setting in sorted(vals):
        candidate = computed[setting]
        if candidate.linear != reference.linear or candidate.redshift != reference.redshift:
            raise ValueError(
                "accuracy study outputs must share redshift and linear/nonlinear status"
            )
        try:
            comp = compare_spectra(reference, candidate, tolerance=tolerance)
        except ValueError as exc:
            raise ValueError(
                f"cannot compare setting {setting:g} against highest setting: {exc}"
            ) from exc
        rows.append(
            AccuracyConvergenceRow(
                setting,
                comp.agreement.rms_fractional_error,
                comp.agreement.max_absolute_fractional_error,
                comp.agreement.passed,
            )
        )
    return tuple(rows)


__all__ = [
    "SpectrumBackend",
    "SpectrumAgreement",
    "RedshiftComparison",
    "BackendConformanceReport",
    "compare_spectra",
    "compare_backends",
    "write_conformance_report",
    "AccuracyConvergenceRow",
    "accuracy_convergence_study",
]
