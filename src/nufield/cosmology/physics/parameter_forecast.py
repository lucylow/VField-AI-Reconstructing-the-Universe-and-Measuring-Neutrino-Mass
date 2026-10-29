"""General Gaussian Fisher forecasting including parameter-dependent covariance.

These forecasts are local quadratic approximations, not posterior samples. They are useful for
checking information flow and degeneracies before committing to expensive simulation campaigns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.derivatives import numerical_jacobian


@dataclass(frozen=True, slots=True)
class CovarianceFisherResult:
    parameter_names: tuple[str, ...]
    mean_jacobian: np.ndarray
    covariance_derivatives: np.ndarray
    fisher_matrix: np.ndarray
    parameter_covariance: np.ndarray
    marginalized_sigma: np.ndarray
    correlation: np.ndarray
    rank: int
    condition_number: float
    warnings: tuple[str, ...]


def _validated_spd(matrix: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(matrix, float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or np.any(~np.isfinite(value)):
        raise ValueError(f"{label} must be a finite square matrix")
    if not np.allclose(value, value.T, rtol=1e-9, atol=1e-11):
        raise ValueError(f"{label} must be symmetric")
    try:
        np.linalg.cholesky(value)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label} must be positive definite") from exc
    return value


def gaussian_fisher_with_covariance_derivatives(
    parameter_names: tuple[str, ...] | list[str],
    theta_fiducial: np.ndarray,
    step_sizes: np.ndarray,
    mean_fn,
    covariance_fn,
    *,
    prior_precision: np.ndarray | None = None,
    relative_rank_tolerance: float = 1e-10,
) -> CovarianceFisherResult:
    r"""Compute the Gaussian Fisher matrix including both mean and covariance information.

    .. math::
       F_{ij}=\mu_{,i}^T C^{-1}\mu_{,j}
       +\tfrac12\operatorname{Tr}(C^{-1}C_{,i}C^{-1}C_{,j})+F_{ij}^{\rm prior}.

    All derivatives use centered finite differences. The covariance term can be noisy and can overstate
    information if the covariance model is estimated or misspecified; use convergence tests in step size.
    """
    names = tuple(parameter_names)
    theta, steps = np.asarray(theta_fiducial, float), np.asarray(step_sizes, float)
    if theta.ndim != 1 or steps.shape != theta.shape or len(names) != theta.size:
        raise ValueError(
            "parameter_names, theta_fiducial and step_sizes must have matching lengths"
        )
    if np.any(steps <= 0) or np.any(~np.isfinite(theta + steps)):
        raise ValueError("finite theta and strictly positive finite steps are required")
    mu0 = np.asarray(mean_fn(theta), float)
    cov0 = _validated_spd(np.asarray(covariance_fn(theta), float), "fiducial covariance")
    if mu0.ndim != 1 or np.any(~np.isfinite(mu0)) or cov0.shape != (mu0.size, mu0.size):
        raise ValueError("mean_fn and covariance_fn returned incompatible output shapes")
    dmu = numerical_jacobian(mean_fn, theta, steps)
    dcov = np.empty((theta.size, mu0.size, mu0.size), float)
    for i, step in enumerate(steps):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        cp = _validated_spd(np.asarray(covariance_fn(plus), float), f"covariance plus step {i}")
        cm = _validated_spd(np.asarray(covariance_fn(minus), float), f"covariance minus step {i}")
        if cp.shape != cov0.shape or cm.shape != cov0.shape:
            raise ValueError("covariance shape changed while taking derivatives")
        dcov[i] = (cp - cm) / (2.0 * step)
    invc = np.linalg.solve(cov0, np.eye(cov0.shape[0]))
    fisher = dmu.T @ invc @ dmu
    for i in range(theta.size):
        for j in range(theta.size):
            fisher[i, j] += 0.5 * np.trace(invc @ dcov[i] @ invc @ dcov[j])
    if prior_precision is not None:
        prior = np.asarray(prior_precision, float)
        if (
            prior.shape != fisher.shape
            or np.any(~np.isfinite(prior))
            or not np.allclose(prior, prior.T)
        ):
            raise ValueError("prior_precision must be finite, symmetric and parameter-sized")
        if np.linalg.eigvalsh(prior).min() < -1e-10:
            raise ValueError("prior_precision must be positive semidefinite")
        fisher += prior
    fisher = 0.5 * (fisher + fisher.T)
    values, vectors = np.linalg.eigh(fisher)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if values.min() < -1e-9 * scale:
        raise ValueError(
            "Fisher matrix has a substantial negative eigenvalue; check derivatives/covariance"
        )
    cutoff = relative_rank_tolerance * max(float(values.max()), 0.0)
    valid = values > cutoff
    rank = int(valid.sum())
    inv_values = np.divide(1.0, values, out=np.zeros_like(values), where=valid)
    pcov = (vectors * inv_values) @ vectors.T
    sig = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    denom = sig[:, None] * sig[None, :]
    corr = np.divide(pcov, denom, out=np.zeros_like(pcov), where=denom > 0)
    cond = float(values.max() / values[valid].min()) if rank and valid.any() else float("inf")
    warnings: list[str] = []
    if rank < len(names):
        warnings.append(
            "Fisher matrix is rank deficient; unconstrained combinations need priors or new observables"
        )
    if cond > 1e10:
        warnings.append(
            "Fisher matrix is ill conditioned; inspect scaled parameter coordinates and derivatives"
        )
    warnings.append(
        "Forecast is a local Gaussian approximation and omits non-Gaussian posterior structure"
    )
    return CovarianceFisherResult(
        names, dmu, dcov, fisher, pcov, sig, corr, rank, cond, tuple(warnings)
    )


def figure_of_merit(covariance: np.ndarray, parameter_indices: tuple[int, int]) -> float:
    """Return inverse 2-D covariance-ellipse area proxy, 1/sqrt(det(C_2D))."""
    c = np.asarray(covariance, float)
    i, j = parameter_indices
    if (
        c.ndim != 2
        or c.shape[0] != c.shape[1]
        or not (0 <= i < c.shape[0] and 0 <= j < c.shape[0])
        or i == j
    ):
        raise ValueError("covariance must be square and parameter indices distinct/in range")
    sub = c[np.ix_([i, j], [i, j])]
    det = float(np.linalg.det(sub))
    if not np.isfinite(det) or det <= 0:
        raise ValueError("selected covariance block must have positive determinant")
    return float(1.0 / np.sqrt(det))


__all__ = [
    "CovarianceFisherResult",
    "gaussian_fisher_with_covariance_derivatives",
    "figure_of_merit",
]
