"""Coverage and convergence diagnostics for common cosmological spectral integrals.

Many useful cosmological quantities are integrals over P(k) or transfer functions. An apparently
stable numerical answer can nevertheless be biased if the input spectrum does not cover the scales
that carry the integral. This module reports the fraction of variance accumulated over k ranges and
compares quadrature settings without silently extrapolating power beyond a solver's output grid.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum, top_hat_window


@dataclass(frozen=True, slots=True)
class VarianceIntegralReport:
    radius_mpc: float
    variance: float
    sigma: float
    k_min_mpc_inv: float
    k_max_mpc_inv: float
    median_contribution_k_mpc_inv: float
    k_10pct_contribution_mpc_inv: float
    k_90pct_contribution_mpc_inv: float
    low_k_variance_fraction: float
    high_k_variance_fraction: float
    n_samples: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def variance_integrand_dlnk(
    k_mpc_inv: np.ndarray, pk_mpc3: np.ndarray, radius_mpc: float
) -> np.ndarray:
    r"""Return Δ²(k)W²(kR), the integrand of σ²(R)=∫dlnk Δ²W²."""
    k, p = np.asarray(k_mpc_inv, float), np.asarray(pk_mpc3, float)
    if k.ndim != 1 or p.shape != k.shape or k.size < 3:
        raise ValueError("k and P must be matching 1-D arrays with >=3 samples")
    if np.any(~np.isfinite(k + p)) or np.any(k <= 0) or np.any(p < 0) or np.any(np.diff(k) <= 0):
        raise ValueError("k must increase, P be nonnegative, and all values finite")
    if not math.isfinite(radius_mpc) or radius_mpc <= 0:
        raise ValueError("radius_mpc must be positive and finite")
    return k**3 * p / (2 * math.pi**2) * top_hat_window(k * radius_mpc) ** 2


def variance_integral_report(
    spectrum: MatterPowerSpectrum,
    radius_mpc: float,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
    low_k_split_mpc_inv: float | None = None,
    high_k_split_mpc_inv: float | None = None,
) -> VarianceIntegralReport:
    """Compute σ²(R) and report how much is contributed by low/high-k portions of support."""
    k, p = spectrum.k_mpc_inv, spectrum.pk_mpc3
    lo = float(k[0]) if k_min_mpc_inv is None else float(k_min_mpc_inv)
    hi = float(k[-1]) if k_max_mpc_inv is None else float(k_max_mpc_inv)
    if lo < k[0] or hi > k[-1] or hi <= lo:
        raise ValueError("requested integration bounds must lie inside spectrum support")
    use = (k >= lo) & (k <= hi)
    if use.sum() < 3:
        raise ValueError("at least three spectral samples remain in requested range")
    kk, pp = k[use], p[use]
    integrand = variance_integrand_dlnk(kk, pp, radius_mpc)
    x = np.log(kk)
    variance = float(simpson(integrand, x=x))
    if variance <= 0:
        raise ValueError("variance integral is non-positive; inspect spectrum support and P(k)")
    # Contribution quantiles are calculated from trapezoid cell contributions in log-k.
    dx = np.diff(x)
    cell = 0.5 * (integrand[:-1] + integrand[1:]) * dx
    centers = np.sqrt(kk[:-1] * kk[1:])
    cumulative = np.concatenate(([0.0], np.cumsum(cell)))
    cumulative /= cumulative[-1] if cumulative[-1] > 0 else 1.0
    q10 = float(np.interp(0.1, cumulative, kk))
    q50 = float(np.interp(0.5, cumulative, kk))
    q90 = float(np.interp(0.9, cumulative, kk))
    low_split = lo * 2 if low_k_split_mpc_inv is None else float(low_k_split_mpc_inv)
    high_split = hi / 2 if high_k_split_mpc_inv is None else float(high_k_split_mpc_inv)
    low = kk <= low_split
    high = kk >= high_split
    low_v = float(simpson(integrand[low], x=x[low])) if low.sum() >= 2 else 0.0
    high_v = float(simpson(integrand[high], x=x[high])) if high.sum() >= 2 else 0.0
    return VarianceIntegralReport(
        float(radius_mpc),
        variance,
        math.sqrt(variance),
        float(lo),
        float(hi),
        q50,
        q10,
        q90,
        low_v / variance,
        high_v / variance,
        int(kk.size),
        (
            "Integral is limited to the supplied spectrum support; no extrapolation was applied.",
            "Contribution quantiles use a trapezoid cumulative integral in ln(k).",
            "Fractions based on separated subsets omit any interpolation across the split boundary.",
        ),
    )


def sigma8_support_report(
    spectrum: MatterPowerSpectrum, h: float, *, minimum_fractional_support: float = 0.01
) -> dict[str, object]:
    """Report whether a spectrum's available k range covers the main σ8 window contribution."""
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be positive")
    r8 = 8.0 / h
    report = variance_integral_report(spectrum, r8)
    supported = (
        report.k_10pct_contribution_mpc_inv >= spectrum.k_mpc_inv[0]
        and report.k_90pct_contribution_mpc_inv <= spectrum.k_mpc_inv[-1]
    )
    # The integral quantiles are defined inside available support, so this flags only a weak boundary
    # warning. Without extrapolation, external tail contribution cannot be known from this spectrum.
    low_boundary_x = spectrum.k_mpc_inv[0] * r8
    high_boundary_x = spectrum.k_mpc_inv[-1] * r8
    warnings = []
    if low_boundary_x > 0.1:
        warnings.append("low-k support begins at kR>0.1; the large-scale σ8 tail may be truncated")
    if high_boundary_x < 10:
        warnings.append("high-k support ends at kR<10; the small-scale σ8 tail may be truncated")
    if not supported:
        warnings.append(
            "internal contribution quantiles indicate a narrow/incomplete effective support"
        )
    return {
        "radius8_mpc": r8,
        "sigma8_within_available_support": report.sigma,
        "k_min_mpc_inv": float(spectrum.k_mpc_inv[0]),
        "k_max_mpc_inv": float(spectrum.k_mpc_inv[-1]),
        "kR_min": float(low_boundary_x),
        "kR_max": float(high_boundary_x),
        "available_support_variance_median_k": report.median_contribution_k_mpc_inv,
        "warnings": warnings,
        "warning": "This is σ8 over available P(k) support, not necessarily the fully converged cosmological σ8.",
    }


def compare_quadrature_resolution(
    k_mpc_inv: np.ndarray,
    pk_mpc3: np.ndarray,
    radius_mpc: float,
    *,
    sample_counts: tuple[int, ...] = (64, 128, 256, 512),
) -> list[dict[str, float | int]]:
    """Interpolate ln P and compare σ²(R) quadrature at several log-spaced resolutions.

    This diagnoses quadrature error for the *given interpolated spectrum*. It does not test whether
    the underlying Boltzmann calculation has enough k support or numerical accuracy.
    """
    k, p = np.asarray(k_mpc_inv, float), np.asarray(pk_mpc3, float)
    if (
        k.ndim != 1
        or p.shape != k.shape
        or len(k) < 3
        or np.any(k <= 0)
        or np.any(p <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError("k/P arrays must be positive, matching and increasing")
    counts = tuple(sorted(set(int(v) for v in sample_counts)))
    if len(counts) < 2 or counts[0] < 8:
        raise ValueError("at least two distinct quadrature counts >=8 required")
    interp = PchipInterpolator(np.log(k), np.log(p), extrapolate=False)
    results = []
    previous = None
    for n in counts:
        kk = np.geomspace(k[0], k[-1], n)
        pp = np.exp(interp(np.log(kk)))
        integrand = variance_integrand_dlnk(kk, pp, radius_mpc)
        value = float(simpson(integrand, x=np.log(kk)))
        frac = (
            float(abs(value - previous) / abs(value))
            if previous is not None and value != 0
            else float("nan")
        )
        results.append(
            {
                "n_samples": n,
                "variance": value,
                "sigma": math.sqrt(max(value, 0)),
                "fractional_change_from_previous": frac,
            }
        )
        previous = value
    return results


__all__ = [
    "VarianceIntegralReport",
    "variance_integrand_dlnk",
    "variance_integral_report",
    "sigma8_support_report",
    "compare_quadrature_resolution",
]
