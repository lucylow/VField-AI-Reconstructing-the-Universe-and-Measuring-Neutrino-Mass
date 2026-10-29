"""Higher-order field diagnostics and cross-statistics."""

from __future__ import annotations

import numpy as np


def one_point_moments(field: np.ndarray) -> dict[str, float]:
    """Return mean, standard deviation, standardized skewness and excess kurtosis."""
    x = np.asarray(field, dtype=float)
    if x.size < 2 or not np.isfinite(x).all():
        raise ValueError("field must have at least two finite elements")
    mu = float(x.mean())
    centered = x - mu
    var = float(np.mean(centered**2))
    if var <= 0:
        skew, kurt = 0.0, 0.0
    else:
        skew = float(np.mean(centered**3) / var**1.5)
        kurt = float(np.mean(centered**4) / var**2 - 3.0)
    return {"mean": mu, "std": float(np.sqrt(var)), "skewness": skew, "excess_kurtosis": kurt}


def isotropic_autocorrelation(field: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Compute a periodic autocorrelation estimate by inverse FFT of the power."""
    x = np.asarray(field, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("field must be finite and 3-D")
    x = x - x.mean()
    pk = np.abs(np.fft.fftn(x)) ** 2
    corr = np.fft.ifftn(pk).real / x.size
    corr /= max(float(corr[0, 0, 0]), 1e-15)
    if max_lag is None:
        return corr.astype(np.float32)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    n = x.shape[0]
    idx = np.arange(min(max_lag + 1, n))
    return corr[np.ix_(idx, idx, idx)].astype(np.float32)


def cross_power_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Summarize the real-space cross correlation and Fourier coherence-like statistic."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape or x.ndim != 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("inputs must be finite, same-shaped 3-D arrays")
    x, y = x - x.mean(), y - y.mean()
    corr = float(np.mean(x * y)) / (float(x.std() * y.std()) + 1e-12)
    xk, yk = np.fft.fftn(x), np.fft.fftn(y)
    cross = np.real(xk * np.conj(yk))
    denom = np.sqrt(np.sum(np.abs(xk) ** 2) * np.sum(np.abs(yk) ** 2)) + 1e-12
    coherence = float(np.sum(cross) / denom)
    return {"zero_lag_correlation": corr, "global_cross_coherence_proxy": coherence}


def finite_difference_gradient(field: np.ndarray, spacing: float = 1.0) -> tuple[np.ndarray, ...]:
    """Periodic central-difference gradient for diagnostics."""
    x = np.asarray(field, dtype=np.float64)
    if x.ndim != 3 or spacing <= 0 or not np.isfinite(x).all():
        raise ValueError("field must be finite 3-D and spacing positive")
    return tuple(
        ((np.roll(x, -1, axis=i) - np.roll(x, 1, axis=i)) / (2 * spacing)).astype(np.float32)
        for i in range(3)
    )
