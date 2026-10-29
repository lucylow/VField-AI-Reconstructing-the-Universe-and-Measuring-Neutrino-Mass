"""Likelihoods that account for covariance estimated from a finite number of simulations.

A plug-in Gaussian likelihood with an estimated covariance can underestimate parameter uncertainty.
This module implements common finite-mock diagnostics and the Sellentin--Heavens multivariate-t form.
Assumptions on independent Gaussian mocks must be checked for the actual survey/mocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass(frozen=True, slots=True)
class CovarianceMockDiagnostics:
    n_mocks: int
    n_bins: int
    hartlap_factor: float | None
    condition_number: float
    eigenvalue_min: float
    eigenvalue_max: float
    can_hartlap_correct: bool
    can_use_t_likelihood: bool
    warnings: tuple[str, ...]


def hartlap_factor(n_mocks: int, n_bins: int) -> float:
    """Return the Hartlap inverse-covariance correction (n_mock - n_bin - 2)/(n_mock - 1)."""
    if n_mocks <= 1 or n_bins < 1:
        raise ValueError("n_mocks must exceed 1 and n_bins must be positive")
    factor = (n_mocks - n_bins - 2) / (n_mocks - 1)
    if factor <= 0:
        raise ValueError("Hartlap correction is non-positive: require n_mocks > n_bins + 2")
    return float(factor)


def diagnose_mock_covariance(covariance: np.ndarray, n_mocks: int) -> CovarianceMockDiagnostics:
    """Report covariance conditioning and finite-mock applicability without modifying the matrix."""
    c = np.asarray(covariance, float)
    if (
        c.ndim != 2
        or c.shape[0] != c.shape[1]
        or np.any(~np.isfinite(c))
        or not np.allclose(c, c.T, atol=1e-10)
    ):
        raise ValueError("covariance must be finite, square and symmetric")
    if n_mocks < 2:
        raise ValueError("at least two mocks required")
    eig = np.linalg.eigvalsh(c)
    p = c.shape[0]
    if eig.min() <= 0:
        raise ValueError("covariance must be positive definite")
    can_h = n_mocks > p + 2
    can_t = n_mocks > p
    warn = []
    if n_mocks <= p:
        warn.append(
            "number of mocks does not exceed number of data bins; covariance inverse is unstable/undefined"
        )
    elif not can_h:
        warn.append("too few mocks for a positive Hartlap factor")
    if n_mocks < 10 * p:
        warn.append(
            "mock count is small relative to dimension; covariance uncertainty may dominate"
        )
    return CovarianceMockDiagnostics(
        n_mocks,
        p,
        hartlap_factor(n_mocks, p) if can_h else None,
        float(eig.max() / eig.min()),
        float(eig.min()),
        float(eig.max()),
        can_h,
        can_t,
        tuple(warn),
    )


def multivariate_t_loglike_from_sample_covariance(
    data: np.ndarray, model: np.ndarray, sample_covariance: np.ndarray, n_mocks: int
) -> float:
    r"""Sellentin--Heavens likelihood for covariance estimated from independent Gaussian mocks.

    .. math::
      \log L = \log\Gamma(N_s/2)-\log\Gamma((N_s-p)/2)
      -\tfrac p2\log[\pi(N_s-1)]-\tfrac12\log|S|
      -\tfrac{N_s}2\log[1+\chi^2/(N_s-1)].

    Here S is the sample covariance (not the Hartlap-rescaled inverse). Requires N_s > p. It does not
    fix non-independent mocks, non-Gaussian covariance sampling or covariance-estimator bias.
    """
    y, m = np.asarray(data, float), np.asarray(model, float)
    c = np.asarray(sample_covariance, float)
    if y.ndim != 1 or m.shape != y.shape or np.any(~np.isfinite(y + m)):
        raise ValueError("data and model must be matching finite vectors")
    p = y.size
    if n_mocks <= p:
        raise ValueError("Sellentin-Heavens likelihood requires n_mocks > number of data bins")
    if c.shape != (p, p) or not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("sample covariance shape/symmetry invalid")
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("sample covariance must be positive definite") from exc
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0:
        raise ValueError("sample covariance determinant must be positive")
    r = y - m
    chi2 = float(r @ np.linalg.solve(c, r))
    return float(
        gammaln(n_mocks / 2)
        - gammaln((n_mocks - p) / 2)
        - 0.5 * p * np.log(np.pi * (n_mocks - 1))
        - 0.5 * logdet
        - 0.5 * n_mocks * np.log1p(chi2 / (n_mocks - 1))
    )


def estimate_covariance_from_mocks(mock_data_vectors: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    """Estimate covariance from array shaped (n_mock,n_bin), with explicit ddof."""
    x = np.asarray(mock_data_vectors, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or np.any(~np.isfinite(x)):
        raise ValueError("mock_data_vectors must be finite shape (n_mock>=2,n_bin>=1)")
    if not 0 <= ddof < x.shape[0]:
        raise ValueError("ddof must be in [0,n_mock)")
    return np.atleast_2d(np.cov(x, rowvar=False, ddof=ddof))


__all__ = [
    "CovarianceMockDiagnostics",
    "hartlap_factor",
    "diagnose_mock_covariance",
    "multivariate_t_loglike_from_sample_covariance",
    "estimate_covariance_from_mocks",
]
