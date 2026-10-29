"""Numerical acceptance checks for physical-theory backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class SpectrumAgreement:
    k_mpc_inv: np.ndarray
    fractional_difference: np.ndarray
    rms_fractional_difference: float
    median_absolute_fractional_difference: float
    max_absolute_fractional_difference: float
    within_tolerance: bool
    tolerance: float


def compare_power_spectra(
    reference: MatterPowerSpectrum,
    candidate: MatterPowerSpectrum,
    *,
    tolerance: float = 0.01,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
) -> SpectrumAgreement:
    """Compare two linear spectra on a shared k interval using log interpolation.

    The relative tolerance is a technical gate selected by the user. For a science claim, tolerance
    must be justified by observable sensitivity and verified against an independent reference setup.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    lo = max(reference.k_mpc_inv[0], candidate.k_mpc_inv[0], k_min_mpc_inv or 0.0)
    hi = min(reference.k_mpc_inv[-1], candidate.k_mpc_inv[-1], k_max_mpc_inv or float("inf"))
    mask = (reference.k_mpc_inv >= lo) & (reference.k_mpc_inv <= hi)
    k = reference.k_mpc_inv[mask]
    if k.size < 3 or hi <= lo:
        raise ValueError("fewer than three shared k samples in requested comparison interval")
    p_ref = reference.evaluate(k)
    p_cand = candidate.evaluate(k)
    frac = p_cand / p_ref - 1.0
    rms = float(np.sqrt(np.mean(frac**2)))
    med = float(np.median(np.abs(frac)))
    mx = float(np.max(np.abs(frac)))
    return SpectrumAgreement(k, frac, rms, med, mx, bool(mx <= tolerance), float(tolerance))


def check_expected_linear_spectrum(
    spectrum: MatterPowerSpectrum, k_min: float, k_max: float
) -> list[str]:
    """Return actionable warnings for basic linear-spectrum support and smoothness issues."""
    issues: list[str] = []
    if spectrum.k_mpc_inv[0] > k_min:
        issues.append("spectrum does not cover requested low-k boundary; avoid extrapolation")
    if spectrum.k_mpc_inv[-1] < k_max:
        issues.append("spectrum does not cover requested high-k boundary; avoid extrapolation")
    log_slope = np.diff(np.log(spectrum.pk_mpc3)) / np.diff(np.log(spectrum.k_mpc_inv))
    if np.percentile(np.abs(np.diff(log_slope)), 99) > 5:
        issues.append(
            "large high-percentile slope jumps detected; inspect interpolation or solver output"
        )
    if not spectrum.linear:
        issues.append("spectrum is tagged nonlinear; do not compare with a linear-only benchmark")
    return issues


__all__ = ["SpectrumAgreement", "compare_power_spectra", "check_expected_linear_spectrum"]
