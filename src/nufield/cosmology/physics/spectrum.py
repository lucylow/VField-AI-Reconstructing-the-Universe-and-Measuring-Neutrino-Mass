"""Unit-aware one-dimensional matter power spectra and derived variance integrals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator


def top_hat_window(x: float | np.ndarray) -> np.ndarray:
    r"""Fourier-space spherical top-hat window W(x)=3(sin x-x cos x)/x³."""
    xx = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(xx)):
        raise ValueError("x must be finite")
    out = np.empty_like(xx, dtype=float)
    small = np.abs(xx) < 1e-3
    xs = xx[small]
    # Series through O(x^6) to avoid cancellation near zero.
    out[small] = 1.0 - xs**2 / 10.0 + xs**4 / 280.0 - xs**6 / 15120.0
    xlarge = xx[~small]
    out[~small] = 3.0 * (np.sin(xlarge) - xlarge * np.cos(xlarge)) / xlarge**3
    return out


@dataclass(frozen=True, slots=True)
class MatterPowerSpectrum:
    """Matter power spectrum with physical units k [1/Mpc], P [Mpc^3]."""

    k_mpc_inv: np.ndarray
    pk_mpc3: np.ndarray
    redshift: float
    linear: bool = True
    source: str = "unspecified"
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        k = np.asarray(self.k_mpc_inv, dtype=float)
        p = np.asarray(self.pk_mpc3, dtype=float)
        if k.ndim != 1 or p.shape != k.shape or k.size < 3:
            raise ValueError("k and P must be same-shape 1-D arrays with at least 3 points")
        if np.any(~np.isfinite(k)) or np.any(~np.isfinite(p)):
            raise ValueError("spectrum contains non-finite values")
        if np.any(k <= 0) or np.any(p <= 0) or np.any(np.diff(k) <= 0):
            raise ValueError("k and P must be positive, with strictly increasing k")
        if self.redshift < 0 or not math.isfinite(self.redshift):
            raise ValueError("redshift must be finite and non-negative")
        # Freeze copied arrays so a frozen dataclass cannot be mutated through its array members.
        object.__setattr__(self, "k_mpc_inv", k.copy())
        object.__setattr__(self, "pk_mpc3", p.copy())

    @property
    def log_interpolator(self) -> PchipInterpolator:
        return PchipInterpolator(np.log(self.k_mpc_inv), np.log(self.pk_mpc3), extrapolate=False)

    def evaluate(self, k_mpc_inv: float | np.ndarray, *, extrapolate: bool = False) -> np.ndarray:
        k = np.asarray(k_mpc_inv, dtype=float)
        if np.any(k <= 0) or np.any(~np.isfinite(k)):
            raise ValueError("evaluation k must be finite and positive")
        if not extrapolate and (np.any(k < self.k_mpc_inv[0]) or np.any(k > self.k_mpc_inv[-1])):
            raise ValueError("requested k lies outside the spectrum; extrapolation is disabled")
        interp = PchipInterpolator(
            np.log(self.k_mpc_inv), np.log(self.pk_mpc3), extrapolate=extrapolate
        )
        return np.exp(interp(np.log(k)))

    def dimensionless_power(self) -> np.ndarray:
        return self.k_mpc_inv**3 * self.pk_mpc3 / (2 * math.pi**2)

    def sigma_r(self, radius_mpc: float, *, k_limits: tuple[float, float] | None = None) -> float:
        r"""Compute σ(R) from available P(k): integral dlnk Δ²(k) W(kR)².

        The integral is only over the sampled k range. Pass ``k_limits`` to restrict it; the method
        never silently extrapolates to unobserved scales.
        """
        if radius_mpc <= 0 or not math.isfinite(radius_mpc):
            raise ValueError("radius_mpc must be finite and positive")
        mask = np.ones_like(self.k_mpc_inv, dtype=bool)
        if k_limits is not None:
            lo, hi = k_limits
            if lo <= 0 or hi <= lo:
                raise ValueError("k_limits must satisfy 0 < k_min < k_max")
            mask &= (self.k_mpc_inv >= lo) & (self.k_mpc_inv <= hi)
        if mask.sum() < 3:
            raise ValueError("fewer than three k samples remain for sigma(R) integral")
        ln_k = np.log(self.k_mpc_inv[mask])
        delta2 = self.dimensionless_power()[mask]
        window = top_hat_window(self.k_mpc_inv[mask] * radius_mpc)
        variance = float(simpson(delta2 * window**2, x=ln_k))
        return math.sqrt(max(variance, 0.0))

    def sigma8(self, *, h: float) -> float:
        """Compute σ8 with radius 8 h^-1 Mpc (= 8/h Mpc in physical units)."""
        if h <= 0:
            raise ValueError("h must be positive")
        return self.sigma_r(8.0 / h)

    def to_h_units(self, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (k [h/Mpc], P [(Mpc/h)^3]) arrays."""
        if h <= 0:
            raise ValueError("h must be positive")
        return self.k_mpc_inv / h, self.pk_mpc3 * h**3


__all__ = ["MatterPowerSpectrum", "top_hat_window"]
