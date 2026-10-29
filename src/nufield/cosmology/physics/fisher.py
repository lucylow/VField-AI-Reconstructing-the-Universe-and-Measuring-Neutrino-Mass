"""Finite-difference Fisher forecasts with conditioning and convention checks.

Fisher matrices are local Gaussian approximations. They are not substitutes for posterior sampling,
non-Gaussian likelihood checks, survey-window treatment, or systematic uncertainty modeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FisherResult:
    parameter_names: tuple[str, ...]
    matrix: np.ndarray
    covariance: np.ndarray
    marginalized_sigma: np.ndarray
    jacobian: np.ndarray
    condition_number: float
    rank: int
    warnings: tuple[str, ...]


def finite_difference_jacobian(
    observable_fn,
    theta: np.ndarray,
    step_sizes: np.ndarray,
) -> np.ndarray:
    """Compute centered finite-difference derivative matrix d observable / d theta."""
    p = np.asarray(theta, dtype=float)
    steps = np.asarray(step_sizes, dtype=float)
    if p.ndim != 1 or steps.shape != p.shape or np.any(steps <= 0) or np.any(~np.isfinite(p)):
        raise ValueError("theta and positive step_sizes must be same-shape 1-D arrays")
    baseline = np.asarray(observable_fn(p), dtype=float)
    if baseline.ndim != 1 or np.any(~np.isfinite(baseline)):
        raise ValueError("observable_fn must return a finite 1-D vector")
    jac = np.empty((baseline.size, p.size), dtype=float)
    for j, step in enumerate(steps):
        plus, minus = p.copy(), p.copy()
        plus[j] += step
        minus[j] -= step
        fp = np.asarray(observable_fn(plus), dtype=float)
        fm = np.asarray(observable_fn(minus), dtype=float)
        if fp.shape != baseline.shape or fm.shape != baseline.shape:
            raise ValueError("observable_fn output shape changed under finite differences")
        if np.any(~np.isfinite(fp)) or np.any(~np.isfinite(fm)):
            raise ValueError("observable_fn returned non-finite values in finite differences")
        jac[:, j] = (fp - fm) / (2 * step)
    return jac


def fisher_forecast(
    parameter_names: tuple[str, ...] | list[str],
    theta_fiducial: np.ndarray,
    step_sizes: np.ndarray,
    observable_fn,
    covariance_data: np.ndarray,
    *,
    prior_precision: np.ndarray | None = None,
    rcond: float = 1e-10,
) -> FisherResult:
    """Build F=Jᵀ C⁻¹ J + prior precision and return pseudo-inverse covariance."""
    names = tuple(parameter_names)
    theta = np.asarray(theta_fiducial, float)
    cov = np.asarray(covariance_data, float)
    if len(names) != theta.size or len(names) != len(step_sizes):
        raise ValueError("parameter_names, theta_fiducial and step_sizes lengths differ")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or np.any(~np.isfinite(cov)):
        raise ValueError("covariance_data must be a finite square matrix")
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10):
        raise ValueError("covariance_data must be symmetric")
    eig = np.linalg.eigvalsh(cov)
    if eig.min() <= 0:
        raise ValueError("covariance_data must be positive definite")
    jac = finite_difference_jacobian(observable_fn, theta, np.asarray(step_sizes))
    if cov.shape != (jac.shape[0], jac.shape[0]):
        raise ValueError("covariance dimension must match observable vector length")
    invcov = np.linalg.inv(cov)
    fmat = jac.T @ invcov @ jac
    if prior_precision is not None:
        prior = np.asarray(prior_precision, float)
        if (
            prior.shape != fmat.shape
            or np.any(~np.isfinite(prior))
            or not np.allclose(prior, prior.T)
        ):
            raise ValueError(
                "prior_precision must be a finite symmetric parameter-by-parameter matrix"
            )
        if np.linalg.eigvalsh(prior).min() < -1e-10:
            raise ValueError("prior_precision must be positive semidefinite")
        fmat = fmat + prior
    u, s, vh = np.linalg.svd(fmat)
    threshold = rcond * s.max() if s.size and s.max() else 0.0
    rank = int(np.sum(s > threshold))
    inv_s = np.array([1.0 / x if x > threshold else 0.0 for x in s])
    posterior_cov = (vh.T * inv_s) @ u.T
    sigmas = np.sqrt(np.maximum(np.diag(posterior_cov), 0.0))
    cond = float(s.max() / s.min()) if s.size and s.min() > 0 else float("inf")
    warn = []
    if rank < len(names):
        warn.append(
            "Fisher matrix is rank-deficient; some parameter combinations are not identified"
        )
    if cond > 1e10:
        warn.append("Fisher matrix is ill-conditioned; inspect parameter scaling and degeneracies")
    return FisherResult(names, fmat, posterior_cov, sigmas, jac, cond, rank, tuple(warn))


def gaussian_logdet(covariance: np.ndarray) -> float:
    """Stable log determinant for a symmetric positive-definite covariance matrix."""
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("covariance must be square and symmetric")
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0:
        raise ValueError("covariance determinant must be positive")
    return float(logdet)


__all__ = ["FisherResult", "finite_difference_jacobian", "fisher_forecast", "gaussian_logdet"]
