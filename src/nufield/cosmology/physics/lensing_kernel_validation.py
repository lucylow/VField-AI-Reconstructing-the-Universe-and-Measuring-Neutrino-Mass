"""Numerical and bookkeeping diagnostics for weak-lensing source distributions and kernels.

This module audits source redshift distributions and kernel arrays before Limber projection. It does not
validate a survey's photo-z calibration, galaxy selection, intrinsic-alignment model, mask coupling, or
nonlinear matter power. The returned checks should be included in an inference provenance manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.lensing import (
    convergence_kernel_mpc_inv,
    lensing_efficiency,
    normalize_n_of_z,
)


@dataclass(frozen=True, slots=True)
class LensingKernelReport:
    source_normalization: float
    minimum_efficiency: float
    maximum_efficiency: float
    minimum_kernel_mpc_inv: float
    maximum_kernel_mpc_inv: float
    kernel_integral_per_sr: float
    zeros_at_high_redshift: int
    monotonicity_violations_efficiency: int
    monotonicity_interpretation: str
    passed_basic_checks: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_normalization": self.source_normalization,
            "minimum_efficiency": self.minimum_efficiency,
            "maximum_efficiency": self.maximum_efficiency,
            "minimum_kernel_mpc_inv": self.minimum_kernel_mpc_inv,
            "maximum_kernel_mpc_inv": self.maximum_kernel_mpc_inv,
            "kernel_integral_per_sr": self.kernel_integral_per_sr,
            "zeros_at_high_redshift": self.zeros_at_high_redshift,
            "monotonicity_violations_efficiency": self.monotonicity_violations_efficiency,
            "monotonicity_interpretation": self.monotonicity_interpretation,
            "passed_basic_checks": self.passed_basic_checks,
            "warnings": list(self.warnings),
            "units": {"kernel": "Mpc^-1", "redshift": "dimensionless"},
        }


def validate_lensing_kernel(
    background: BackgroundCosmology,
    z: np.ndarray,
    source_nz: np.ndarray,
    *,
    source_normalization_tolerance: float = 2e-4,
    negative_tolerance: float = 1e-10,
) -> LensingKernelReport:
    """Build and audit a flat-universe lensing efficiency and convergence kernel."""
    zz, nz = np.asarray(z, float), np.asarray(source_nz, float)
    if zz.ndim != 1 or len(zz) < 16 or nz.shape != zz.shape or np.any(~np.isfinite(zz + nz)):
        raise ValueError("z and source_nz must be finite matching vectors with >=16 samples")
    if np.any(zz < 0) or np.any(np.diff(zz) <= 0) or np.any(nz < 0):
        raise ValueError("redshift must be increasing/non-negative and source_nz non-negative")
    if source_normalization_tolerance <= 0 or negative_tolerance < 0:
        raise ValueError("invalid normalization/negative tolerances")
    raw_norm = float(simpson(nz, x=zz))
    normalized = normalize_n_of_z(zz, nz)
    efficiency = lensing_efficiency(background, zz, zz, normalized)
    kernel = convergence_kernel_mpc_inv(background, zz, efficiency)
    n_eff_violations = int(np.count_nonzero(np.diff(efficiency) > 1e-5))
    # Efficiency need not be globally monotonic for arbitrary multi-modal source n(z); report only.
    tail = max(1, len(zz) // 20)
    zeros_tail = int(np.count_nonzero(np.abs(kernel[-tail:]) <= negative_tolerance))
    passed = bool(
        abs(raw_norm - 1.0) <= source_normalization_tolerance
        or abs(float(simpson(normalized, x=zz)) - 1.0) <= source_normalization_tolerance
    )
    if np.min(efficiency) < -negative_tolerance or np.min(kernel) < -negative_tolerance:
        passed = False
    warnings = [
        "Efficiency monotonicity is not a universal physical constraint for a general multimodal n(z); violations are reported only.",
        "Kernel integral is descriptive and depends on the chosen redshift support; it is not expected to equal one.",
        "This checks numerical non-negativity and normalization, not survey selection or intrinsic-alignment correctness.",
    ]
    if abs(raw_norm - 1.0) > source_normalization_tolerance:
        warnings.append(
            "Input source_nz was not normalized; lensing_efficiency normalizes it internally."
        )
    if abs(background.params.omega_k) > 1e-6:
        warnings.append(
            "The implemented kernel is flat-universe only; non-zero curvature needs distance-ratio generalization."
        )
    return LensingKernelReport(
        raw_norm,
        float(efficiency.min()),
        float(efficiency.max()),
        float(kernel.min()),
        float(kernel.max()),
        float(simpson(kernel, x=zz)),
        zeros_tail,
        n_eff_violations,
        "Do not require monotonic efficiency for arbitrary source distributions.",
        passed,
        tuple(warnings),
    )


def lensing_kernel_resolution_study(
    background_factory,
    source_distribution_factory,
    *,
    z_min: float = 0.0,
    z_max: float = 3.0,
    sample_counts: tuple[int, ...] = (64, 128, 256, 512),
) -> list[dict[str, float | int]]:
    """Assess changes in a kernel integral across redshift-grid resolutions.

    Factories receive a strictly increasing redshift array and must return a background object and n(z)
    respectively. The study is useful for code tests, not a substitute for convergence of projected Cℓ.
    """
    counts = sorted(set(int(n) for n in sample_counts))
    if len(counts) < 2 or counts[0] < 16 or not 0 <= z_min < z_max:
        raise ValueError("at least two sample counts >=16 and 0<=z_min<z_max required")
    results = []
    previous = None
    for n in counts:
        z = np.linspace(z_min, z_max, n)
        bg = background_factory(z)
        nz = source_distribution_factory(z)
        report = validate_lensing_kernel(bg, z, nz)
        value = report.kernel_integral_per_sr
        results.append(
            {
                "n_z": n,
                "kernel_integral": value,
                "fractional_change_from_previous": float(
                    abs(value - previous) / max(abs(value), 1e-30)
                )
                if previous is not None
                else float("nan"),
            }
        )
        previous = value
    return results


__all__ = ["LensingKernelReport", "validate_lensing_kernel", "lensing_kernel_resolution_study"]
