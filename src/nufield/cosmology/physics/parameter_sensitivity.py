"""Numerical theory derivatives, local identifiability, and Fisher sensitivity diagnostics.

These utilities are model-agnostic and work with matter spectra, BAO distance vectors, weak-lensing
bandpowers, or concatenated multi-probe observables. Numerical derivatives are only meaningful when
finite-difference step studies demonstrate stability and the forward model is deterministic enough for
the requested precision. Fisher forecasts are local Gaussian approximations, not full posterior results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DerivativeResult:
    parameter_names: tuple[str, ...]
    observable: np.ndarray
    jacobian: np.ndarray
    steps: np.ndarray
    methods: tuple[str, ...]
    relative_step_diagnostics: dict[str, float]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        y, j, h = (
            np.asarray(self.observable, float),
            np.asarray(self.jacobian, float),
            np.asarray(self.steps, float),
        )
        if (
            y.ndim != 1
            or j.shape != (y.size, len(self.parameter_names))
            or h.shape != (len(self.parameter_names),)
        ):
            raise ValueError("derivative result dimensions are inconsistent")
        if np.any(~np.isfinite(y)) or np.any(~np.isfinite(j)) or np.any(h <= 0):
            raise ValueError("derivative outputs must be finite and steps positive")
        object.__setattr__(self, "observable", y.copy())
        object.__setattr__(self, "jacobian", j.copy())
        object.__setattr__(self, "steps", h.copy())

    def to_dict(self, *, include_jacobian: bool = True) -> dict[str, object]:
        return {
            "parameter_names": list(self.parameter_names),
            "observable_dimension": int(self.observable.size),
            "parameter_dimension": int(len(self.parameter_names)),
            "steps": self.steps.tolist(),
            "methods": list(self.methods),
            "relative_step_diagnostics": self.relative_step_diagnostics,
            "notes": list(self.notes),
            "jacobian": self.jacobian.tolist() if include_jacobian else None,
        }


@dataclass(frozen=True, slots=True)
class IdentifiabilityReport:
    singular_values: np.ndarray
    rank: int
    condition_number: float
    effective_rank: float
    near_degenerate_parameter_combinations: np.ndarray
    parameter_names: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        s = np.asarray(self.singular_values, float)
        v = np.asarray(self.near_degenerate_parameter_combinations, float)
        if s.ndim != 1 or v.ndim != 2 or v.shape[1] != len(self.parameter_names):
            raise ValueError("identifiability report shapes inconsistent")
        object.__setattr__(self, "singular_values", s.copy())
        object.__setattr__(self, "near_degenerate_parameter_combinations", v.copy())

    def to_dict(self) -> dict[str, object]:
        return {
            "singular_values": self.singular_values.tolist(),
            "rank": self.rank,
            "condition_number": self.condition_number,
            "effective_rank": self.effective_rank,
            "near_degenerate_parameter_combinations": self.near_degenerate_parameter_combinations.tolist(),
            "parameter_names": list(self.parameter_names),
            "warnings": list(self.warnings),
        }


def finite_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    parameter_names: Sequence[str],
    steps: float | np.ndarray,
    *,
    method: str = "central",
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    relative_scale: bool = False,
) -> DerivativeResult:
    r"""Estimate Jacobian J_ij=∂O_i/∂θ_j with explicit differencing convention.

    Central differences are preferred for smooth deterministic forward models. Forward differences are
    available near hard boundaries. If ``relative_scale`` is true, each supplied step is interpreted as
    a multiplier of max(|θ_j|,1) in the user's parameter units. Optional bounds are enforced before
    any model evaluation; a central stencil that crosses a bound is rejected instead of silently
    changing its order.
    """
    theta = np.asarray(parameters, float)
    names = tuple(str(x) for x in parameter_names)
    hraw = np.asarray(steps, float)
    h = np.full(theta.size, float(hraw)) if hraw.ndim == 0 else hraw.copy()
    if (
        theta.ndim != 1
        or theta.size == 0
        or len(names) != theta.size
        or len(set(names)) != len(names)
    ):
        raise ValueError("parameters must be 1-D with unique names of equal length")
    if h.shape != theta.shape or np.any(h <= 0) or np.any(~np.isfinite(theta + h)):
        raise ValueError("steps must be positive finite and match parameter shape")
    if relative_scale:
        h = h * np.maximum(np.abs(theta), 1.0)
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")
    lo = hi = None
    if bounds is not None:
        lo, hi = map(lambda v: np.asarray(v, float), bounds)
        if lo.shape != theta.shape or hi.shape != theta.shape or np.any(lo >= hi):
            raise ValueError("bounds must be matching lower/upper vectors with lo<hi")
        if np.any(theta < lo) or np.any(theta > hi):
            raise ValueError("parameters lie outside supplied bounds")
    base = np.asarray(function(theta.copy()), float)
    if base.ndim != 1 or base.size == 0 or np.any(~np.isfinite(base)):
        raise ValueError("function must return a finite non-empty 1-D observable vector")
    jac = np.empty((base.size, theta.size), float)
    actual_methods: list[str] = []
    for j, step in enumerate(h):
        plus = theta.copy()
        plus[j] += step
        minus = theta.copy()
        minus[j] -= step
        if lo is not None and (plus[j] > hi[j] or minus[j] < lo[j]):
            raise ValueError(f"finite-difference stencil for {names[j]} crosses a parameter bound")
        yp = np.asarray(function(plus), float)
        if yp.shape != base.shape or np.any(~np.isfinite(yp)):
            raise ValueError(f"function returned invalid observable for +step in {names[j]}")
        if method == "central":
            ym = np.asarray(function(minus), float)
            if ym.shape != base.shape or np.any(~np.isfinite(ym)):
                raise ValueError(f"function returned invalid observable for -step in {names[j]}")
            jac[:, j] = (yp - ym) / (2 * step)
            actual_methods.append("central")
        else:
            jac[:, j] = (yp - base) / step
            actual_methods.append("forward")
    # Dimensionless diagnostic columns use max(|O|, its RMS scale floor) only for reporting.
    scale = max(float(np.sqrt(np.mean(base**2))), 1e-30)
    relative_cols = np.linalg.norm(jac, axis=0) * h / scale
    return DerivativeResult(
        names,
        base,
        jac,
        h,
        tuple(actual_methods),
        {name: float(value) for name, value in zip(names, relative_cols)},
        (
            "Finite differences inherit numerical noise and solver interpolation errors.",
            "Repeat with at least two step sizes before interpreting Fisher information.",
        ),
    )


def finite_difference_step_stability(
    function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    parameter_names: Sequence[str],
    steps_small: np.ndarray,
    steps_large: np.ndarray,
    *,
    tolerance: float = 0.1,
    parameter_scales: np.ndarray | None = None,
) -> dict[str, object]:
    """Compare Jacobians computed with two finite-difference step vectors."""
    small = finite_difference_jacobian(function, parameters, parameter_names, steps_small)
    large = finite_difference_jacobian(function, parameters, parameter_names, steps_large)
    if small.jacobian.shape != large.jacobian.shape:
        raise ArithmeticError("finite-difference Jacobian shapes differ")
    scale = (
        np.ones(small.jacobian.shape[0])
        if parameter_scales is None
        else np.asarray(parameter_scales, float)
    )
    if scale.shape != (small.jacobian.shape[0],) or np.any(scale <= 0):
        raise ValueError("parameter_scales must be positive and match observable dimension")
    denom = np.maximum(
        np.abs(large.jacobian),
        1e-10 * np.max(np.abs(large.jacobian), axis=0, keepdims=True) + 1e-20,
    )
    rel = (small.jacobian - large.jacobian) / denom
    rms_each = np.sqrt(np.mean(rel**2, axis=0))
    max_each = np.max(np.abs(rel), axis=0)
    return {
        "parameter_names": list(small.parameter_names),
        "rms_relative_difference": rms_each.tolist(),
        "max_absolute_relative_difference": max_each.tolist(),
        "tolerance": float(tolerance),
        "passed": bool(np.all(max_each <= tolerance)),
        "warning": "Stability is necessary but not sufficient: both step sizes can share the same modeling bias.",
    }


def whitened_jacobian(jacobian: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Whiten derivative rows by Cholesky covariance: J_w=L⁻¹J for C=LLᵀ."""
    j, c = np.asarray(jacobian, float), np.asarray(covariance, float)
    if j.ndim != 2 or c.shape != (j.shape[0], j.shape[0]) or np.any(~np.isfinite(j)):
        raise ValueError("jacobian and covariance shapes/values invalid")
    if not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return np.linalg.solve(chol, j)


def identifiability_report(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    parameter_names: Sequence[str],
    *,
    relative_singular_threshold: float = 1e-6,
) -> IdentifiabilityReport:
    """Diagnose locally degenerate parameter combinations from the whitened Jacobian SVD."""
    j = np.asarray(jacobian, float)
    names = tuple(str(n) for n in parameter_names)
    if j.ndim != 2 or j.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("jacobian shape must match unique parameter names")
    if not 0 < relative_singular_threshold < 1:
        raise ValueError("relative_singular_threshold must be in (0,1)")
    jw = whitened_jacobian(j, covariance)
    # Column scaling prevents parameter units alone from dominating numerical rank diagnostics.
    colnorm = np.linalg.norm(jw, axis=0)
    scaled = jw / np.where(colnorm > 0, colnorm, 1.0)
    _, s, vt = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(s > relative_singular_threshold * s[0])) if s.size and s[0] > 0 else 0
    cond = float(s[0] / s[-1]) if s.size and s[-1] > 0 else float("inf")
    weights = s**2
    effective_rank = (
        float(np.sum(weights) ** 2 / np.sum(weights**2)) if np.sum(weights**2) > 0 else 0.0
    )
    near = vt[s <= relative_singular_threshold * s[0]] if s.size and s[0] > 0 else vt
    warnings = []
    if rank < len(names):
        warnings.append(
            "some parameter combinations are locally unidentifiable for this data vector and covariance"
        )
    if cond > 1e8:
        warnings.append(
            "whitened Jacobian is poorly conditioned; priors or additional probes may be needed"
        )
    if np.any(colnorm == 0):
        warnings.append(
            "one or more parameters have zero local response at the chosen fiducial point"
        )
    return IdentifiabilityReport(s, rank, cond, effective_rank, near, names, tuple(warnings))


def fisher_from_jacobian(
    jacobian: np.ndarray, covariance: np.ndarray, prior_fisher: np.ndarray | None = None
) -> np.ndarray:
    """Compute local Fisher matrix JᵀC⁻¹J, optionally adding prior Fisher information."""
    jw = whitened_jacobian(jacobian, covariance)
    f = jw.T @ jw
    if prior_fisher is not None:
        pf = np.asarray(prior_fisher, float)
        if pf.shape != f.shape or np.any(~np.isfinite(pf)) or not np.allclose(pf, pf.T, atol=1e-10):
            raise ValueError("prior_fisher must be a finite symmetric matrix matching parameters")
        f = f + pf
    return 0.5 * (f + f.T)


def fisher_covariance(
    fisher: np.ndarray, *, allow_pseudoinverse: bool = False
) -> tuple[np.ndarray, dict[str, object]]:
    """Invert Fisher matrix and report condition; pseudoinverse must be explicitly requested."""
    f = np.asarray(fisher, float)
    if (
        f.ndim != 2
        or f.shape[0] != f.shape[1]
        or np.any(~np.isfinite(f))
        or not np.allclose(f, f.T, atol=1e-10)
    ):
        raise ValueError("Fisher matrix must be finite, square and symmetric")
    eig = np.linalg.eigvalsh(f)
    if eig.min() <= 0 and not allow_pseudoinverse:
        raise ValueError(
            "Fisher matrix is not positive definite; include justified priors or opt in to pseudoinverse"
        )
    if eig.min() > 0:
        cov = np.linalg.inv(f)
        mode = "inverse"
    else:
        cov = np.linalg.pinv(f, hermitian=True)
        mode = "pseudoinverse"
    report = {
        "mode": mode,
        "min_eigenvalue": float(eig.min()),
        "max_eigenvalue": float(eig.max()),
        "condition_number": float(eig.max() / eig.min()) if eig.min() > 0 else float("inf"),
        "marginalized_sigma": np.sqrt(np.maximum(np.diag(cov), 0)).tolist(),
        "warning": "Fisher forecasts are local Gaussian approximations and may fail for boundary parameters.",
    }
    return cov, report


__all__ = [
    "DerivativeResult",
    "IdentifiabilityReport",
    "finite_difference_jacobian",
    "finite_difference_step_stability",
    "whitened_jacobian",
    "identifiability_report",
    "fisher_from_jacobian",
    "fisher_covariance",
]
