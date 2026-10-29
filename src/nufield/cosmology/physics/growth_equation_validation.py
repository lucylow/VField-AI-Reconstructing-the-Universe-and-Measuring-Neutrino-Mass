"""Numerically audit supplied growth histories against the scale-independent GR growth equation.

For pressureless, scale-independent clustering in GR, D obeys a second-order ODE in ln(a). This is a
consistency check for a supplied D(a), H(a), and Ω_m(a), not a complete equation for massive-neutrino
perturbations. Massive neutrinos induce scale-dependent growth and require a Boltzmann or validated
simulation treatment for precision claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GrowthEquationReport:
    scale_factor: np.ndarray
    growth: np.ndarray
    first_derivative_ln_a: np.ndarray
    second_derivative_ln_a: np.ndarray
    friction_coefficient: np.ndarray
    gravity_source: np.ndarray
    residual: np.ndarray
    normalized_residual: np.ndarray
    max_abs_normalized_residual_interior: float
    rms_normalized_residual_interior: float
    points_used: int
    caveats: tuple[str, ...]

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "max_abs_normalized_residual_interior": self.max_abs_normalized_residual_interior,
            "rms_normalized_residual_interior": self.rms_normalized_residual_interior,
            "points_used": self.points_used,
            "caveats": list(self.caveats),
            "equation": "D'' + (2 + dlnH/dlna)D' - 1.5 Omega_m(a) D = 0",
        }
        if include_arrays:
            for key in (
                "scale_factor",
                "growth",
                "first_derivative_ln_a",
                "second_derivative_ln_a",
                "friction_coefficient",
                "gravity_source",
                "residual",
                "normalized_residual",
            ):
                out[key] = getattr(self, key).tolist()
        return out


def validate_growth_history(
    scale_factor: np.ndarray,
    growth: np.ndarray,
    H_km_s_mpc: np.ndarray,
    omega_m_clustering: np.ndarray,
    *,
    drop_boundary_points: int = 2,
) -> GrowthEquationReport:
    r"""Compute the finite-difference residual of the linear growth ODE.

    Inputs must be tabulated on a strictly increasing scale-factor grid. Derivatives are evaluated in
    x=ln(a), where primes denote d/dx. The normalized residual divides by the sum of absolute equation
    terms plus a small floor. Boundary points can be omitted because finite-difference derivatives are
    least accurate there. A small residual is necessary but not sufficient for a correct perturbation
    solution; a wrong smooth-matter source term can still make an internally consistent residual.
    """
    a, d, h, om = (
        np.asarray(v, float) for v in (scale_factor, growth, H_km_s_mpc, omega_m_clustering)
    )
    if a.ndim != 1 or len(a) < 9 or any(x.shape != a.shape for x in (d, h, om)):
        raise ValueError("a, growth, H and omega_m must be matching 1-D arrays with >=9 points")
    if any(np.any(~np.isfinite(x)) for x in (a, d, h, om)):
        raise ValueError("growth-history inputs must be finite")
    if np.any(a <= 0) or np.any(a > 1.0 + 1e-10) or np.any(np.diff(a) <= 0):
        raise ValueError("scale factor must increase strictly within (0, 1]")
    if np.any(d <= 0) or np.any(h <= 0) or np.any(om < 0):
        raise ValueError("growth and H must be positive and omega_m non-negative")
    if drop_boundary_points < 1 or 2 * drop_boundary_points >= len(a) - 2:
        raise ValueError("drop_boundary_points must be >=1 and leave enough interior samples")
    x = np.log(a)
    d1 = np.gradient(d, x, edge_order=2)
    d2 = np.gradient(d1, x, edge_order=2)
    dlnh = np.gradient(np.log(h), x, edge_order=2)
    friction = 2.0 + dlnh
    source = 1.5 * om * d
    residual = d2 + friction * d1 - source
    scale = np.abs(d2) + np.abs(friction * d1) + np.abs(source)
    normalized = residual / np.maximum(scale, 1e-12 * np.max(scale))
    core = slice(drop_boundary_points, -drop_boundary_points)
    max_res = float(np.max(np.abs(normalized[core])))
    rms_res = float(np.sqrt(np.mean(normalized[core] ** 2)))
    caveats = (
        "Finite-difference residual checks numerical consistency only; they do not prove the chosen model is physical.",
        "The equation assumes scale-independent clustering in GR and is not a neutrino-aware growth solution.",
        "Boundary derivatives are omitted from aggregate residuals; retain grid-resolution checks before claiming convergence.",
    )
    return GrowthEquationReport(
        a.copy(),
        d.copy(),
        d1,
        d2,
        friction,
        source,
        residual,
        normalized,
        max_res,
        rms_res,
        len(a) - 2 * drop_boundary_points,
        caveats,
    )


def einstein_de_sitter_reference(n: int = 257, *, a_min: float = 1e-3) -> dict[str, np.ndarray]:
    """Return analytic EdS arrays (D=a, H∝a^-3/2, Ω_m=1) for unit tests and demos."""
    if n < 9 or not 0 < a_min < 1:
        raise ValueError("n>=9 and 0<a_min<1 are required")
    a = np.geomspace(a_min, 1.0, int(n))
    return {
        "scale_factor": a,
        "growth": a.copy(),
        "H_km_s_mpc": a**-1.5,
        "omega_m_clustering": np.ones_like(a),
    }


def convergence_of_growth_residual(
    resolutions: tuple[int, ...] = (65, 129, 257, 513),
    *,
    a_min: float = 1e-3,
) -> list[dict[str, float | int]]:
    """Test an analytic EdS solution at multiple grid resolutions to expose derivative truncation error."""
    counts = sorted(set(int(n) for n in resolutions))
    if len(counts) < 2 or counts[0] < 9:
        raise ValueError("at least two resolutions >=9 are required")
    rows: list[dict[str, float | int]] = []
    for n in counts:
        data = einstein_de_sitter_reference(n, a_min=a_min)
        report = validate_growth_history(**data)
        rows.append(
            {
                "n_points": n,
                "max_abs_residual": report.max_abs_normalized_residual_interior,
                "rms_residual": report.rms_normalized_residual_interior,
            }
        )
    return rows


__all__ = [
    "GrowthEquationReport",
    "validate_growth_history",
    "einstein_de_sitter_reference",
    "convergence_of_growth_residual",
]
