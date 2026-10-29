"""Power-spectrum integration, sigma_R normalization, and convergence diagnostics.

The methods require physical units: k in Mpc^-1 and P(k) in Mpc^3. They perform consistency checks
on the supplied spectrum, not an independent transfer-function calculation. In particular, sigma8
is only a useful validation target if the spectrum's cosmological parameters, redshift, neutrino
prescription, and solver settings are matched to the reference value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


def top_hat_window(x: np.ndarray | float) -> np.ndarray:
    """Fourier-space spherical top-hat window, evaluated stably near x=0."""
    a = np.asarray(x, float)
    if np.any(~np.isfinite(a)):
        raise ValueError("x must be finite")
    out = np.empty_like(a)
    small = np.abs(a) < 1e-3
    x2 = a[small] ** 2
    out[small] = 1.0 - x2 / 10.0 + x2**2 / 280.0 - x2**3 / 15120.0
    regular = ~small
    xr = a[regular]
    out[regular] = 3.0 * (np.sin(xr) - xr * np.cos(xr)) / xr**3
    return out


def dimensionless_power(k_mpc_inv: np.ndarray, pk_mpc3: np.ndarray) -> np.ndarray:
    """Return Δ²(k)=k³P(k)/(2π²), dimensionless under physical Mpc units."""
    k, p = np.asarray(k_mpc_inv, float), np.asarray(pk_mpc3, float)
    if k.ndim != 1 or p.shape != k.shape or k.size < 2 or np.any(k <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError("k must be positive increasing 1-D and P(k) must match")
    if np.any(~np.isfinite(k)) or np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("k and P must be finite; P must be non-negative")
    return k**3 * p / (2 * math.pi**2)


def sigma_r_squared(
    spectrum: MatterPowerSpectrum,
    radius_mpc: float,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
    extrapolation: str = "raise",
) -> float:
    r"""Compute σ²(R)=∫dlnk Δ²(k) W²(kR) over the declared spectral interval.

    ``extrapolation='raise'`` rejects support extension. ``'zero'`` clips the integration interval to
    the spectrum support; it does not silently assume a power-law tail. Use a solver with broader
    k coverage for a publication-quality σ8 check.
    """
    if not np.isfinite(radius_mpc) or radius_mpc <= 0:
        raise ValueError("radius_mpc must be finite and positive")
    if extrapolation not in {"raise", "zero"}:
        raise ValueError("extrapolation must be 'raise' or 'zero'")
    k = spectrum.k_mpc_inv
    lo = float(k[0] if k_min_mpc_inv is None else k_min_mpc_inv)
    hi = float(k[-1] if k_max_mpc_inv is None else k_max_mpc_inv)
    if not 0 < lo < hi:
        raise ValueError("require 0 < k_min < k_max")
    if extrapolation == "raise" and (lo < k[0] or hi > k[-1]):
        raise ValueError("requested integration interval exceeds power-spectrum support")
    lo, hi = max(lo, float(k[0])), min(hi, float(k[-1]))
    use = (k >= lo) & (k <= hi)
    kk, pp = k[use], spectrum.pk_mpc3[use]
    if kk.size < 3:
        # Interpolate exact endpoints and retain enough nodes for numerical quadrature.
        logk = np.log(k)
        interp = PchipInterpolator(logk, spectrum.pk_mpc3, extrapolate=False)
        endpoints = np.array([lo, hi])
        interior = k[(k > lo) & (k < hi)]
        kk = np.unique(np.concatenate((endpoints, interior)))
        pp = interp(np.log(kk))
    else:
        interp = PchipInterpolator(np.log(k), spectrum.pk_mpc3, extrapolate=False)
        kk = np.unique(np.concatenate(([lo], kk, [hi])))
        pp = interp(np.log(kk))
    if np.any(~np.isfinite(pp)) or np.any(pp < 0):
        raise ValueError("interpolated power contains invalid values")
    integrand = dimensionless_power(kk, pp) * top_hat_window(kk * radius_mpc) ** 2
    value = float(trapezoid(integrand, x=np.log(kk)))
    if not np.isfinite(value) or value < 0:
        raise FloatingPointError("sigma_R integral is invalid")
    return value


def sigma_r(spectrum: MatterPowerSpectrum, radius_mpc: float, **kwargs: object) -> float:
    """Return sqrt of :func:`sigma_r_squared`."""
    return math.sqrt(sigma_r_squared(spectrum, radius_mpc, **kwargs))


def sigma8_from_spectrum(spectrum: MatterPowerSpectrum, h: float, **kwargs: object) -> float:
    """Compute σ8 using R=8/h Mpc, given physical-unit P(k) in Mpc³."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    return sigma_r(spectrum, 8.0 / h, **kwargs)


def spectral_slope(k: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return d ln P / d ln k from a positive spectrum using second-order gradients."""
    kk, pp = np.asarray(k, float), np.asarray(p, float)
    if kk.ndim != 1 or pp.shape != kk.shape or kk.size < 3:
        raise ValueError("k and P must be matching vectors of length >=3")
    if (
        np.any(kk <= 0)
        or np.any(pp <= 0)
        or np.any(~np.isfinite(kk + pp))
        or np.any(np.diff(kk) <= 0)
    ):
        raise ValueError("k and P must be finite, positive, and k increasing")
    return np.gradient(np.log(pp), np.log(kk), edge_order=2)


@dataclass(frozen=True, slots=True)
class Sigma8ConvergenceReport:
    estimates: tuple[float, ...]
    n_k_values: tuple[int, ...]
    fractional_change: tuple[float, ...]
    last_fractional_change: float
    tolerance: float
    converged: bool
    caveat: str

    def to_dict(self) -> dict[str, object]:
        return {
            "estimates": list(self.estimates),
            "n_k_values": list(self.n_k_values),
            "fractional_change": list(self.fractional_change),
            "last_fractional_change": self.last_fractional_change,
            "tolerance": self.tolerance,
            "converged": self.converged,
            "caveat": self.caveat,
        }


def sigma8_integration_convergence(
    spectrum: MatterPowerSpectrum,
    h: float,
    *,
    n_k_values: Sequence[int] = (64, 128, 256, 512),
    tolerance: float = 2e-3,
) -> Sigma8ConvergenceReport:
    """Re-sample a tabulated spectrum in log-k and quantify σ8 integration convergence."""
    counts = tuple(int(n) for n in n_k_values)
    if (
        len(counts) < 2
        or any(n < 16 for n in counts)
        or any(b <= a for a, b in zip(counts, counts[1:]))
    ):
        raise ValueError("n_k_values must be at least two strictly increasing values >=16")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    lo, hi = spectrum.k_mpc_inv[0], spectrum.k_mpc_inv[-1]
    interp = PchipInterpolator(np.log(spectrum.k_mpc_inv), spectrum.pk_mpc3, extrapolate=False)
    vals = []
    for n in counts:
        k = np.geomspace(lo, hi, n)
        p = interp(np.log(k))
        resampled = MatterPowerSpectrum(
            k,
            p,
            spectrum.redshift,
            linear=spectrum.linear,
            source=spectrum.source,
            metadata=spectrum.metadata,
        )
        vals.append(sigma8_from_spectrum(resampled, h, extrapolation="raise"))
    changes = [0.0]
    for a, b in zip(vals, vals[1:]):
        changes.append(abs(b - a) / max(abs(b), np.finfo(float).tiny))
    last = float(changes[-1])
    return Sigma8ConvergenceReport(
        tuple(map(float, vals)),
        counts,
        tuple(map(float, changes)),
        last,
        float(tolerance),
        bool(last <= tolerance),
        "This checks quadrature/resampling stability on the supplied k range; it cannot detect missing physical low/high-k power.",
    )


__all__ = [
    "top_hat_window",
    "dimensionless_power",
    "sigma_r_squared",
    "sigma_r",
    "sigma8_from_spectrum",
    "spectral_slope",
    "Sigma8ConvergenceReport",
    "sigma8_integration_convergence",
]
