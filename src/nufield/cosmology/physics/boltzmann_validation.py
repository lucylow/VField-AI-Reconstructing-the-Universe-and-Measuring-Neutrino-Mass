"""Numerical comparison helpers for CAMB/CLASS-produced theoretical spectra.

These routines do not choose the physical setup for you. Comparisons require the same cosmology,
neutrino hierarchy, redshift, units, nonlinear flag, gauge/convention and solver accuracy where relevant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class SpectrumComparison:
    k_mpc_inv: np.ndarray
    reference_pk_mpc3: np.ndarray
    candidate_pk_mpc3: np.ndarray
    fractional_difference: np.ndarray
    log_ratio_rms: float
    log_ratio_max_abs: float
    median_fractional_difference: float
    compared_points: int
    units: str
    caveats: tuple[str, ...]


def compare_matter_spectra(
    reference: MatterPowerSpectrum,
    candidate: MatterPowerSpectrum,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
    n_points: int = 256,
    max_extrapolation_fraction: float = 0.0,
) -> SpectrumComparison:
    """Interpolate both P(k) on their common log-k domain and summarize residuals.

    The function will not extrapolate by default. A nonzero extrapolation_fraction is an explicit
    percentage of the narrower spectrum's range allowed outside common support, but it should rarely
    be used for solver validation.
    """
    if not (np.isclose(reference.redshift, candidate.redshift, rtol=0, atol=1e-10)):
        raise ValueError("reference and candidate spectra must have the same redshift")
    if reference.linear != candidate.linear:
        raise ValueError("cannot compare linear and nonlinear spectra as a solver agreement test")
    if n_points < 8 or max_extrapolation_fraction < 0:
        raise ValueError("n_points must be >=8 and extrapolation allowance non-negative")
    lo = (
        max(float(reference.k_mpc_inv[0]), float(candidate.k_mpc_inv[0]))
        if k_min_mpc_inv is None
        else float(k_min_mpc_inv)
    )
    hi = (
        min(float(reference.k_mpc_inv[-1]), float(candidate.k_mpc_inv[-1]))
        if k_max_mpc_inv is None
        else float(k_max_mpc_inv)
    )
    if lo <= 0 or hi <= lo:
        raise ValueError("comparison interval must satisfy 0 < k_min < k_max")
    common_lo = max(float(reference.k_mpc_inv[0]), float(candidate.k_mpc_inv[0]))
    common_hi = min(float(reference.k_mpc_inv[-1]), float(candidate.k_mpc_inv[-1]))
    common_width = common_hi - common_lo
    allowed_lo = max(np.finfo(float).tiny, common_lo - max_extrapolation_fraction * common_width)
    allowed_hi = common_hi + max_extrapolation_fraction * common_width
    if lo < allowed_lo or hi > allowed_hi:
        raise ValueError(
            "requested interval exceeds common support and explicit extrapolation allowance"
        )
    if hi / lo <= 1.001:
        raise ValueError("requested interval is too narrow")
    k = np.geomspace(lo, hi, n_points)
    pref = PchipInterpolator(
        np.log(reference.k_mpc_inv), np.log(reference.pk_mpc3), extrapolate=True
    )(np.log(k))
    pcand = PchipInterpolator(
        np.log(candidate.k_mpc_inv), np.log(candidate.pk_mpc3), extrapolate=True
    )(np.log(k))
    logratio = pcand - pref
    frac = np.expm1(logratio)
    caveats = [
        "Interpolation is performed in ln(P) versus ln(k).",
        "Agreement is conditional on matched solver settings and physical conventions.",
    ]
    if lo < common_lo or hi > common_hi:
        caveats.append("Requested comparison includes explicitly permitted extrapolation.")
    return SpectrumComparison(
        k,
        np.exp(pref),
        np.exp(pcand),
        frac,
        float(np.sqrt(np.mean(logratio**2))),
        float(np.max(np.abs(logratio))),
        float(np.median(frac)),
        n_points,
        "k: Mpc^-1; P: Mpc^3",
        tuple(caveats),
    )


def assert_spectrum_agreement(
    comparison: SpectrumComparison, *, max_log_rms: float, max_log_abs: float | None = None
) -> None:
    """Raise AssertionError if interpolated log-spectrum discrepancy exceeds an explicit tolerance."""
    if max_log_rms <= 0 or (max_log_abs is not None and max_log_abs <= 0):
        raise ValueError("tolerances must be positive")
    failures = []
    if comparison.log_ratio_rms > max_log_rms:
        failures.append(f"log-ratio RMS {comparison.log_ratio_rms:g} > {max_log_rms:g}")
    if max_log_abs is not None and comparison.log_ratio_max_abs > max_log_abs:
        failures.append(f"max |log ratio| {comparison.log_ratio_max_abs:g} > {max_log_abs:g}")
    if failures:
        raise AssertionError("spectrum agreement failed: " + "; ".join(failures))


__all__ = ["SpectrumComparison", "compare_matter_spectra", "assert_spectrum_agreement"]
