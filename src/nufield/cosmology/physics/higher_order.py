"""Tree-level perturbation-theory kernels and bispectrum diagnostics.

The F2 kernel below uses the Einstein--de Sitter approximation to second-order standard perturbation
theory. It is a reference teaching model for weakly nonlinear scales, not a nonlinear massive-neutrino
or survey-grade prediction.
"""

from __future__ import annotations

import numpy as np


def f2_eds(k1: float | np.ndarray, k2: float | np.ndarray, mu12: float | np.ndarray) -> np.ndarray:
    r"""Second-order density kernel F2(k1,k2)=5/7+μ/2(k1/k2+k2/k1)+2μ²/7."""
    a, b, mu = np.asarray(k1, float), np.asarray(k2, float), np.asarray(mu12, float)
    if (
        np.any(a <= 0)
        or np.any(b <= 0)
        or np.any(np.abs(mu) > 1)
        or np.any(~np.isfinite(a + b + mu))
    ):
        raise ValueError("k1,k2 must be positive and mu12 in [-1,1]")
    return 5 / 7 + 0.5 * mu * (a / b + b / a) + 2 / 7 * mu**2


def triangle_cosine(k1: np.ndarray, k2: np.ndarray) -> float:
    """Cosine of the angle between two finite nonzero wavevectors."""
    a, b = np.asarray(k1, float), np.asarray(k2, float)
    if a.shape != (3,) or b.shape != (3,) or np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        raise ValueError("k1 and k2 must be nonzero 3-vectors")
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))


def tree_level_bispectrum_eds(
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    power,
) -> float:
    r"""Compute B=2F2(k1,k2)P1P2+cyc for a closed Fourier triangle.

    ``power(k_abs)`` returns P(k) in a consistent volume unit. Input wavevectors must sum to zero
    within tolerance. The estimator does not include shot noise or nonlinear loop corrections.
    """
    vecs = [np.asarray(v, float) for v in (k1, k2, k3)]
    if any(v.shape != (3,) or np.any(~np.isfinite(v)) for v in vecs):
        raise ValueError("k1,k2,k3 must be finite 3-vectors")
    if np.linalg.norm(vecs[0] + vecs[1] + vecs[2]) > 1e-6 * max(
        1.0, *(np.linalg.norm(v) for v in vecs)
    ):
        raise ValueError("k vectors must form a closed triangle: k1+k2+k3≈0")
    kmag = [float(np.linalg.norm(v)) for v in vecs]
    p = [float(power(k)) for k in kmag]
    if any(v <= 0 or not np.isfinite(v) for v in kmag + p):
        raise ValueError("power callback and wavevector magnitudes must be finite and positive")
    mus = [
        triangle_cosine(vecs[0], vecs[1]),
        triangle_cosine(vecs[1], vecs[2]),
        triangle_cosine(vecs[2], vecs[0]),
    ]
    f = [
        float(f2_eds(kmag[0], kmag[1], mus[0])),
        float(f2_eds(kmag[1], kmag[2], mus[1])),
        float(f2_eds(kmag[2], kmag[0], mus[2])),
    ]
    return 2 * (f[0] * p[0] * p[1] + f[1] * p[1] * p[2] + f[2] * p[2] * p[0])


def reduced_bispectrum(bispectrum: float, p1: float, p2: float, p3: float) -> float:
    """Return Q=B/(P1P2+P2P3+P3P1), a shape-sensitive diagnostic."""
    if not np.isfinite(bispectrum) or any(p <= 0 or not np.isfinite(p) for p in (p1, p2, p3)):
        raise ValueError("bispectrum finite and all powers finite/positive required")
    return float(bispectrum / (p1 * p2 + p2 * p3 + p3 * p1))


__all__ = ["f2_eds", "triangle_cosine", "tree_level_bispectrum_eds", "reduced_bispectrum"]
