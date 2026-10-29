"""Local identifiability diagnostics for cosmological parameters and observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.derivatives import numerical_jacobian


@dataclass(frozen=True, slots=True)
class IdentifiabilityReport:
    parameter_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    fisher_matrix: np.ndarray
    parameter_correlation: np.ndarray
    effective_rank: int
    condition_number: float
    warnings: tuple[str, ...]


def whiten_jacobian(jacobian: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Whiten observable derivatives with a Cholesky factor of the data covariance."""
    j, c = np.asarray(jacobian, float), np.asarray(covariance, float)
    if j.ndim != 2 or c.shape != (j.shape[0], j.shape[0]) or np.any(~np.isfinite(j)):
        raise ValueError("jacobian/covariance dimensions mismatch or non-finite values")
    if not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    try:
        lower = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return np.linalg.solve(lower, j)


def diagnose_identifiability(
    parameter_names: tuple[str, ...] | list[str],
    observable_names: tuple[str, ...] | list[str],
    theta_fiducial: np.ndarray,
    steps: np.ndarray,
    observable_fn,
    covariance: np.ndarray,
    *,
    relative_rank_tolerance: float = 1e-6,
) -> IdentifiabilityReport:
    """Estimate parameter response directions and identify near-degenerate combinations.

    Derivatives are local finite differences. Their interpretation depends on parameter scaling,
    selected observables, covariance, nuisance assumptions and finite-difference step choices.
    """
    pnames, onames = tuple(parameter_names), tuple(observable_names)
    theta, step = np.asarray(theta_fiducial, float), np.asarray(steps, float)
    if len(pnames) != theta.size or len(pnames) != step.size:
        raise ValueError("parameter names, theta and steps must have equal lengths")
    y = np.asarray(observable_fn(theta), float)
    if y.ndim != 1 or len(onames) != y.size:
        raise ValueError("observable_names length must match observable_fn output")
    jac = numerical_jacobian(observable_fn, theta, step, method="central")
    jw = whiten_jacobian(jac, covariance)
    _, s, vh = np.linalg.svd(jw, full_matrices=False)
    threshold = relative_rank_tolerance * s[0] if s.size else 0.0
    rank = int(np.count_nonzero(s > threshold))
    f = jw.T @ jw
    covp = np.linalg.pinv(f, rcond=relative_rank_tolerance)
    std = np.sqrt(np.maximum(np.diag(covp), 0))
    denom = std[:, None] * std[None, :]
    corr = np.divide(covp, denom, out=np.zeros_like(covp), where=denom > 0)
    condition = float(s[0] / s[-1]) if s.size and s[-1] > threshold else float("inf")
    warnings = []
    if rank < len(pnames):
        warnings.append(
            "local response is rank-deficient; at least one parameter combination is not identified"
        )
    if condition > 1e8:
        warnings.append(
            "local response is ill-conditioned; inspect scaling and parameter degeneracies"
        )
    return IdentifiabilityReport(pnames, onames, s, vh, f, corr, rank, condition, tuple(warnings))


__all__ = ["IdentifiabilityReport", "whiten_jacobian", "diagnose_identifiability"]
