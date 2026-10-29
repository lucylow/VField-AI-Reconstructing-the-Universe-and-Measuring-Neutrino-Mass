"""Covariance-weighted parameter degeneracy and local Fisher diagnostics.

This module quantifies degeneracies conditional on a supplied observable Jacobian and covariance. Fisher
matrices are local Gaussian approximations: they do not include nonlinear posterior boundaries, prior
volume effects, simulator uncertainty, or the full non-Gaussian shape of a neutrino-mass posterior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ParameterDegeneracyReport:
    parameter_names: tuple[str, ...]
    fisher_data: np.ndarray
    fisher_prior: np.ndarray
    fisher_total: np.ndarray
    parameter_correlation: np.ndarray
    marginalized_sigma: np.ndarray
    conditional_sigma: np.ndarray
    derivative_cosine: np.ndarray
    singular_values_whitened_jacobian: np.ndarray
    rank: int
    condition_number: float
    n_data: int
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_names": list(self.parameter_names),
            "fisher_data": self.fisher_data.tolist(),
            "fisher_prior": self.fisher_prior.tolist(),
            "fisher_total": self.fisher_total.tolist(),
            "parameter_correlation": self.parameter_correlation.tolist(),
            "marginalized_sigma": self.marginalized_sigma.tolist(),
            "conditional_sigma": self.conditional_sigma.tolist(),
            "derivative_cosine": self.derivative_cosine.tolist(),
            "singular_values_whitened_jacobian": self.singular_values_whitened_jacobian.tolist(),
            "rank": self.rank,
            "condition_number": self.condition_number,
            "n_data": self.n_data,
            "caveats": list(self.caveats),
        }


def _symmetric_psd(matrix: np.ndarray, name: str, *, allow_semidefinite: bool = True) -> np.ndarray:
    x = np.asarray(matrix, float)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or np.any(~np.isfinite(x)):
        raise ValueError(f"{name} must be a finite square matrix")
    if not np.allclose(x, x.T, atol=1e-10, rtol=0):
        raise ValueError(f"{name} must be symmetric")
    e = np.linalg.eigvalsh(0.5 * (x + x.T))
    if e.min() < -1e-10 * max(float(np.max(np.abs(e))), 1.0):
        raise ValueError(f"{name} must be positive semidefinite")
    if not allow_semidefinite and e.min() <= 0:
        raise ValueError(f"{name} must be positive definite")
    return 0.5 * (x + x.T)


def parameter_degeneracy_report(
    jacobian: np.ndarray,
    data_covariance: np.ndarray,
    *,
    parameter_names: tuple[str, ...] | None = None,
    prior_covariance: np.ndarray | None = None,
    svd_rcond: float = 1e-10,
) -> ParameterDegeneracyReport:
    """Return Fisher, correlation, derivative angle and marginalized error diagnostics.

    The Jacobian uses rows as observables and columns as parameters. The data covariance is expected to
    include the declared observable units and all covariances intended for the local forecast. A prior
    covariance, if provided, contributes its inverse to the Fisher matrix; infinite-width priors should
    be omitted rather than represented by huge arbitrary variances.
    """
    j, c = np.asarray(jacobian, float), np.asarray(data_covariance, float)
    if j.ndim != 2 or j.shape[0] < 1 or j.shape[1] < 1 or np.any(~np.isfinite(j)):
        raise ValueError("jacobian must be a finite 2-D matrix with non-zero dimensions")
    ndata, npar = j.shape
    c = _symmetric_psd(c, "data_covariance", allow_semidefinite=False)
    if c.shape != (ndata, ndata):
        raise ValueError("data covariance dimension must match Jacobian rows")
    if not 0 < svd_rcond < 1:
        raise ValueError("svd_rcond must be in (0,1)")
    names = tuple(parameter_names or [f"theta_{i}" for i in range(npar)])
    if len(names) != npar or len(set(names)) != npar:
        raise ValueError("parameter_names must be unique and match Jacobian columns")
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("data covariance is not positive definite") from exc
    # Solve L X = J instead of explicitly materializing C^-1.
    jw = np.linalg.solve(chol, j)
    fisher_data = jw.T @ jw
    fisher_data = 0.5 * (fisher_data + fisher_data.T)
    if prior_covariance is None:
        fisher_prior = np.zeros((npar, npar))
    else:
        prior = _symmetric_psd(prior_covariance, "prior_covariance", allow_semidefinite=False)
        if prior.shape != (npar, npar):
            raise ValueError("prior covariance dimension must match Jacobian columns")
        fisher_prior = np.linalg.inv(prior)
    fisher_total = 0.5 * (fisher_data + fisher_prior + (fisher_data + fisher_prior).T)
    # Cosine in the data-covariance metric: a value near ±1 indicates similar/opposite local effects.
    norm = np.sqrt(np.maximum(np.diag(fisher_data), 0.0))
    denom = norm[:, None] * norm[None, :]
    cosine = np.divide(fisher_data, denom, out=np.zeros_like(fisher_data), where=denom > 0)
    cosine = np.clip(cosine, -1, 1)
    u, singular, vt = np.linalg.svd(jw, full_matrices=False)
    cutoff = svd_rcond * singular[0] if singular.size and singular[0] > 0 else 0
    rank = int(np.count_nonzero(singular > cutoff))
    positive = singular[singular > cutoff]
    cond = float((positive[0] / positive[-1]) ** 2) if len(positive) else math.inf
    eig_f = np.linalg.eigvalsh(fisher_total)
    if eig_f.min() <= 0:
        raise ValueError(
            "combined Fisher matrix is singular; add justified priors or reduce parameter set"
        )
    covp = np.linalg.inv(fisher_total)
    corr_den = np.sqrt(np.outer(np.diag(covp), np.diag(covp)))
    corr = covp / corr_den
    marg = np.sqrt(np.diag(covp))
    # Conditional uncertainties assume all other parameters are fixed: inverse sqrt diagonal Fisher.
    diagonal = np.diag(fisher_total)
    cond_sigma = np.divide(1.0, np.sqrt(diagonal), out=np.full(npar, np.inf), where=diagonal > 0)
    return ParameterDegeneracyReport(
        names,
        fisher_data,
        fisher_prior,
        fisher_total,
        corr,
        marg,
        cond_sigma,
        cosine,
        singular,
        rank,
        cond,
        ndata,
        (
            "Fisher quantities are local Gaussian approximations around the selected fiducial model.",
            "Near-collinear derivatives can make constraints prior-dominated and forecast errors unstable.",
            "The result is conditional on the provided covariance, model derivatives and omitted nuisance parameters.",
        ),
    )


def derivative_cosine_from_responses(
    response_a: np.ndarray, response_b: np.ndarray, covariance: np.ndarray
) -> float:
    """Return covariance-weighted cosine between two observable-response vectors."""
    a, b, c = (
        np.asarray(response_a, float),
        np.asarray(response_b, float),
        np.asarray(covariance, float),
    )
    if a.ndim != 1 or b.shape != a.shape or c.shape != (a.size, a.size):
        raise ValueError(
            "responses must be matching vectors and covariance must match their dimension"
        )
    c = _symmetric_psd(c, "covariance", allow_semidefinite=False)
    if np.any(~np.isfinite(a + b)):
        raise ValueError("responses must be finite")
    ia, ib = np.linalg.solve(c, a), np.linalg.solve(c, b)
    aa, bb = float(a @ ia), float(b @ ib)
    if aa <= 0 or bb <= 0:
        raise ValueError("response norm must be positive")
    return float(np.clip(a @ ib / math.sqrt(aa * bb), -1, 1))


__all__ = [
    "ParameterDegeneracyReport",
    "parameter_degeneracy_report",
    "derivative_cosine_from_responses",
]
