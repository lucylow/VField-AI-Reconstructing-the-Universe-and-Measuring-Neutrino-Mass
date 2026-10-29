"""Tree-level perturbation-theory bispectrum for a pressureless matter field.

The Einstein--de Sitter F2 kernel is a widely used approximation for late-time large-scale structure.
This module exposes the approximation and its limits; it is not a precision massive-neutrino bispectrum
solver, and should not be extrapolated into the nonlinear regime without validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


def f2_kernel_einstein_de_sitter(
    k1: np.ndarray | float, k2: np.ndarray | float, mu12: np.ndarray | float
) -> np.ndarray:
    r"""Evaluate F₂=5/7 + μ/2(k₁/k₂+k₂/k₁) + 2μ²/7 for positive magnitudes."""
    a, b, mu = np.broadcast_arrays(
        np.asarray(k1, float), np.asarray(k2, float), np.asarray(mu12, float)
    )
    if (
        np.any(~np.isfinite(a + b + mu))
        or np.any(a <= 0)
        or np.any(b <= 0)
        or np.any(np.abs(mu) > 1)
    ):
        raise ValueError("k1,k2 positive finite and mu12 in [-1,1] required")
    return 5.0 / 7.0 + 0.5 * mu * (a / b + b / a) + 2.0 / 7.0 * mu**2


def triangle_cosine(
    k1: np.ndarray | float, k2: np.ndarray | float, k3: np.ndarray | float
) -> np.ndarray:
    """Return the angle cosine between vectors k1 and k2 in a closed triangle with side k3."""
    a, b, c = np.broadcast_arrays(
        np.asarray(k1, float), np.asarray(k2, float), np.asarray(k3, float)
    )
    if np.any(~np.isfinite(a + b + c)) or np.any(a <= 0) or np.any(b <= 0) or np.any(c <= 0):
        raise ValueError("triangle sides must be finite and positive")
    if np.any(a + b < c - 1e-12) or np.any(a + c < b - 1e-12) or np.any(b + c < a - 1e-12):
        raise ValueError("sides do not form a triangle")
    return np.clip((c * c - a * a - b * b) / (2 * a * b), -1.0, 1.0)


def tree_bispectrum(
    p1: np.ndarray | float,
    p2: np.ndarray | float,
    p3: np.ndarray | float,
    k1: np.ndarray | float,
    k2: np.ndarray | float,
    k3: np.ndarray | float,
    *,
    mu12: np.ndarray | float | None = None,
    mu23: np.ndarray | float | None = None,
    mu31: np.ndarray | float | None = None,
) -> np.ndarray:
    r"""Compute B=2F₂(1,2)P₁P₂ + cyclic terms for a closed wavevector triangle.

    If cosines are not given, they are derived from triangle side lengths using the cosine law.
    Supplying cosines can be useful for explicit vectors, but each must be physically valid in [-1,1].
    """
    a, b, c, ka, kb, kc = np.broadcast_arrays(
        *[np.asarray(v, float) for v in (p1, p2, p3, k1, k2, k3)]
    )
    if (
        np.any(~np.isfinite(a + b + c + ka + kb + kc))
        or np.any(np.stack([a, b, c]) < 0)
        or np.any(np.stack([ka, kb, kc]) <= 0)
    ):
        raise ValueError("powers must be non-negative finite and side lengths positive finite")
    m12 = triangle_cosine(ka, kb, kc) if mu12 is None else np.asarray(mu12, float)
    m23 = triangle_cosine(kb, kc, ka) if mu23 is None else np.asarray(mu23, float)
    m31 = triangle_cosine(kc, ka, kb) if mu31 is None else np.asarray(mu31, float)
    if any(np.any(~np.isfinite(v)) or np.any(np.abs(v) > 1) for v in (m12, m23, m31)):
        raise ValueError("triangle cosines must be finite and lie in [-1,1]")
    f12 = f2_kernel_einstein_de_sitter(ka, kb, m12)
    f23 = f2_kernel_einstein_de_sitter(kb, kc, m23)
    f31 = f2_kernel_einstein_de_sitter(kc, ka, m31)
    return 2 * f12 * a * b + 2 * f23 * b * c + 2 * f31 * c * a


def reduced_tree_bispectrum(
    bispectrum: np.ndarray | float,
    p1: np.ndarray | float,
    p2: np.ndarray | float,
    p3: np.ndarray | float,
) -> np.ndarray:
    """Compute dimensionless reduced bispectrum Q=B/(P1P2+P2P3+P3P1)."""
    b, a, c, d = np.broadcast_arrays(*[np.asarray(v, float) for v in (bispectrum, p1, p2, p3)])
    if np.any(~np.isfinite(b + a + c + d)) or np.any(np.stack([a, c, d]) < 0):
        raise ValueError("inputs must be finite and powers non-negative")
    den = a * c + c * d + d * a
    if np.any(den <= 0):
        raise ValueError("reduced bispectrum denominator must be positive")
    return b / den


def tree_bispectrum_from_spectrum(
    spectrum: MatterPowerSpectrum,
    triangle_sides_mpc_inv: np.ndarray,
) -> np.ndarray:
    """Evaluate the tree bispectrum for an array with final axis ``(k1,k2,k3)``."""
    tri = np.asarray(triangle_sides_mpc_inv, float)
    if tri.shape[-1:] != (3,) or tri.ndim < 1 or np.any(~np.isfinite(tri)) or np.any(tri <= 0):
        raise ValueError("triangle_sides_mpc_inv must be finite positive with final dimension 3")
    if (
        np.any(tri[..., 0] + tri[..., 1] < tri[..., 2] - 1e-12)
        or np.any(tri[..., 0] + tri[..., 2] < tri[..., 1] - 1e-12)
        or np.any(tri[..., 1] + tri[..., 2] < tri[..., 0] - 1e-12)
    ):
        raise ValueError("one or more sides do not close to a triangle")
    interp = PchipInterpolator(
        np.log(spectrum.k_mpc_inv), np.log(spectrum.pk_mpc3), extrapolate=False
    )
    if np.any(tri < spectrum.k_mpc_inv[0]) or np.any(tri > spectrum.k_mpc_inv[-1]):
        raise ValueError("triangle k values exceed supplied spectrum support")
    p = np.exp(interp(np.log(tri)))
    return tree_bispectrum(p[..., 0], p[..., 1], p[..., 2], tri[..., 0], tri[..., 1], tri[..., 2])


@dataclass(frozen=True, slots=True)
class BispectrumValidityReport:
    k_max_mpc_inv: float
    redshift: float
    linear_spectrum: bool
    requested_validated_k_max: float
    within_declared_domain: bool
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "k_max_mpc_inv": self.k_max_mpc_inv,
            "redshift": self.redshift,
            "linear_spectrum": self.linear_spectrum,
            "requested_validated_k_max": self.requested_validated_k_max,
            "within_declared_domain": self.within_declared_domain,
            "caveats": list(self.caveats),
        }


def audit_tree_bispectrum_domain(
    spectrum: MatterPowerSpectrum, triangle_sides: np.ndarray, *, validated_k_max_mpc_inv: float
) -> BispectrumValidityReport:
    """Label whether all requested sides lie inside a user-declared perturbative validation range."""
    tri = np.asarray(triangle_sides, float)
    if tri.shape[-1:] != (3,) or np.any(~np.isfinite(tri)) or np.any(tri <= 0):
        raise ValueError("triangles must have positive finite sides in final dimension 3")
    if not np.isfinite(validated_k_max_mpc_inv) or validated_k_max_mpc_inv <= 0:
        raise ValueError("validated_k_max_mpc_inv must be positive")
    km = float(np.max(tri))
    inside = bool(
        km <= validated_k_max_mpc_inv
        and km <= spectrum.k_mpc_inv[-1]
        and np.min(tri) >= spectrum.k_mpc_inv[0]
    )
    return BispectrumValidityReport(
        km,
        spectrum.redshift,
        spectrum.linear,
        float(validated_k_max_mpc_inv),
        inside,
        (
            "The EdS F2 kernel is approximate for general cosmologies.",
            "Massive-neutrino scale-dependent growth is not modeled in F2.",
            "A user-declared k-domain is not a substitute for comparison to simulations.",
        ),
    )


__all__ = [
    "f2_kernel_einstein_de_sitter",
    "triangle_cosine",
    "tree_bispectrum",
    "reduced_tree_bispectrum",
    "tree_bispectrum_from_spectrum",
    "BispectrumValidityReport",
    "audit_tree_bispectrum_domain",
]
