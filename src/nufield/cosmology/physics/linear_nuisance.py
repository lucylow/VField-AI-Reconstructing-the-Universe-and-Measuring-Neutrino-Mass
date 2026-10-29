"""Generalized least-squares templates and Gaussian marginalization for linear nuisance parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearNuisanceFit:
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    residual: np.ndarray
    chi_squared: float
    log_determinant_normal: float
    rank: int


def fit_linear_nuisance(
    data: np.ndarray,
    base_model: np.ndarray,
    templates: np.ndarray,
    covariance: np.ndarray,
    *,
    prior_mean: np.ndarray | None = None,
    prior_covariance: np.ndarray | None = None,
) -> LinearNuisanceFit:
    r"""Fit data ≈ base_model + templates @ a under Gaussian errors and optional Gaussian prior.

    When a prior is supplied, the returned coefficients are the posterior mean under the linear-Gaussian
    model. Marginal likelihood determinants are not automatically assembled here; use the returned
    normal-matrix logdet with the data covariance and prior normalization in the full evidence formula.
    """
    y, base, t, c = map(lambda x: np.asarray(x, float), (data, base_model, templates, covariance))
    if y.ndim != 1 or base.shape != y.shape or t.ndim != 2 or t.shape[0] != y.size:
        raise ValueError(
            "data/base_model must match and templates must have shape (n_data,n_template)"
        )
    if c.shape != (y.size, y.size) or any(np.any(~np.isfinite(a)) for a in (y, base, t, c)):
        raise ValueError(
            "data, model, templates and covariance must be finite with matching shapes"
        )
    if not np.allclose(c, c.T, atol=1e-10) or np.linalg.eigvalsh(c).min() <= 0:
        raise ValueError("data covariance must be symmetric positive definite")
    cinv_t = np.linalg.solve(c, t)
    cinv_r = np.linalg.solve(c, y - base)
    normal = t.T @ cinv_t
    rhs = t.T @ cinv_r
    ntemp = t.shape[1]
    if prior_mean is not None or prior_covariance is not None:
        if prior_mean is None or prior_covariance is None:
            raise ValueError("prior_mean and prior_covariance must be provided together")
        pm, pc = np.asarray(prior_mean, float), np.asarray(prior_covariance, float)
        if (
            pm.shape != (ntemp,)
            or pc.shape != (ntemp, ntemp)
            or not np.allclose(pc, pc.T, atol=1e-10)
        ):
            raise ValueError("nuisance prior dimensions are invalid")
        if np.linalg.eigvalsh(pc).min() <= 0:
            raise ValueError("prior covariance must be positive definite")
        pinv = np.linalg.inv(pc)
        normal += pinv
        rhs += pinv @ pm
    sign, logdet = np.linalg.slogdet(normal)
    if sign <= 0:
        raise ValueError("nuisance normal matrix is not positive definite")
    cov_a = np.linalg.inv(normal)
    a = cov_a @ rhs
    residual = y - base - t @ a
    chi2 = float(residual @ np.linalg.solve(c, residual))
    rank = int(np.linalg.matrix_rank(normal))
    return LinearNuisanceFit(a, cov_a, residual, chi2, float(logdet), rank)


def polynomial_nuisance_templates(
    x: np.ndarray, degree: int, *, center: float = 0.0, scale: float = 1.0
) -> np.ndarray:
    """Build a numerically scaled polynomial design matrix with columns 1,x,...,x^degree."""
    xx = np.asarray(x, float)
    if xx.ndim != 1 or np.any(~np.isfinite(xx)) or degree < 0 or scale <= 0:
        raise ValueError("x finite vector, degree>=0 and scale>0 required")
    u = (xx - center) / scale
    return np.column_stack([u**d for d in range(degree + 1)])


def marginalized_gaussian_loglike_linear_templates(
    data: np.ndarray,
    base_model: np.ndarray,
    templates: np.ndarray,
    covariance: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
) -> float:
    """Evaluate Gaussian likelihood analytically marginalized over Gaussian linear-template priors.

    The marginalized data covariance is C + T Λ Tᵀ and the mean is m + T a_prior. This exact identity
    is preferred to optimizing nuisance amplitudes and then treating them as fixed without accounting
    for uncertainty.
    """
    y, m, t, c = map(lambda x: np.asarray(x, float), (data, base_model, templates, covariance))
    a0, lam = np.asarray(prior_mean, float), np.asarray(prior_covariance, float)
    if t.ndim != 2 or t.shape[0] != y.size or m.shape != y.shape or c.shape != (y.size, y.size):
        raise ValueError("inconsistent data/template/covariance dimensions")
    if a0.shape != (t.shape[1],) or lam.shape != (t.shape[1], t.shape[1]):
        raise ValueError("prior shape does not match template count")
    cm = c + t @ lam @ t.T
    mu = m + t @ a0
    sign, logdet = np.linalg.slogdet(cm)
    if sign <= 0:
        raise ValueError("marginalized covariance must be positive definite")
    r = y - mu
    return float(-0.5 * (r @ np.linalg.solve(cm, r) + logdet + y.size * math.log(2 * math.pi)))


__all__ = [
    "LinearNuisanceFit",
    "fit_linear_nuisance",
    "polynomial_nuisance_templates",
    "marginalized_gaussian_loglike_linear_templates",
]
