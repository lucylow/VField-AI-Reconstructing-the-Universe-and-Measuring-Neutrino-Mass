"""Finite-mock-aware covariance and likelihood tools for cosmological data vectors.

This module makes several commonly hidden choices explicit: the number of independent mocks,
finite-sample precision correction, covariance shrinkage target, and the likelihood family. It is a
statistical component, not a substitute for a correctly derived cosmological covariance. The sample
covariance must be estimated from mocks that represent the same estimator, binning, survey window,
and nuisance assumptions as the measured data vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammaln

LikelihoodKind = Literal["gaussian", "sellentin_heavens"]


def sample_covariance_from_mocks(mock_data: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    """Estimate covariance from mocks of shape ``(n_mock, n_data)``.

    Rows are independent mock realizations and columns are the *same ordered* observables. This
    helper cannot detect duplicate parent boxes; that provenance check must happen before the call.
    """
    x = np.asarray(mock_data, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("mock_data must have shape (n_mock>=2, n_data>=1)")
    if np.any(~np.isfinite(x)):
        raise ValueError("mock_data contains non-finite values")
    if ddof < 0 or ddof >= x.shape[0]:
        raise ValueError("ddof must be non-negative and smaller than n_mock")
    cov = np.cov(x, rowvar=False, ddof=ddof)
    return np.atleast_2d(np.asarray(cov, float))


def hartlap_precision_factor(n_mocks: int, n_data: int) -> float:
    """Hartlap et al. multiplicative precision correction ``(Nmock-Ndata-2)/(Nmock-1)``.

    The factor is defined only for ``Nmock > Ndata + 2``. It applies to an inverse *sample*
    covariance under assumptions described in the literature; it is not a generic correction for
    arbitrary covariances or likelihoods.
    """
    if n_mocks < 1 or n_data < 1:
        raise ValueError("n_mocks and n_data must be positive integers")
    if n_mocks <= n_data + 2:
        raise ValueError("Hartlap correction requires n_mocks > n_data + 2")
    return float((n_mocks - n_data - 2) / (n_mocks - 1))


def invert_sample_covariance(
    covariance: np.ndarray,
    n_mocks: int,
    *,
    apply_hartlap: bool = True,
) -> np.ndarray:
    """Invert a positive-definite sample covariance and optionally apply its Hartlap factor."""
    c = _validate_covariance(covariance)
    if n_mocks < 2:
        raise ValueError("n_mocks must be at least two")
    try:
        precision = np.linalg.inv(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("sample covariance is singular; diagnose or shrink explicitly") from exc
    if apply_hartlap:
        precision *= hartlap_precision_factor(n_mocks, c.shape[0])
    return precision


def _validate_covariance(covariance: np.ndarray, *, positive_definite: bool = True) -> np.ndarray:
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 1 or np.any(~np.isfinite(c)):
        raise ValueError("covariance must be a finite square matrix")
    if not np.allclose(c, c.T, rtol=1e-9, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    c = 0.5 * (c + c.T)
    eig = np.linalg.eigvalsh(c)
    if positive_definite and eig.min() <= 0:
        raise ValueError("covariance must be positive definite")
    if not positive_definite and eig.min() < -1e-10 * max(1.0, abs(eig.max())):
        raise ValueError("covariance must be positive semidefinite")
    return c


def diagonal_target_shrinkage(
    covariance: np.ndarray,
    alpha: float,
    *,
    preserve_variances: bool = True,
) -> np.ndarray:
    """Shrink covariance toward its diagonal by a fixed declared alpha.

    ``alpha=0`` leaves the covariance unchanged; ``alpha=1`` removes all off-diagonal terms. This
    deterministic shrinkage is deliberately not advertised as an optimal Ledoit-Wolf estimator.
    Choose alpha by a documented validation strategy, never by looking at final test performance.
    """
    c = _validate_covariance(covariance, positive_definite=False)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    target = np.diag(np.diag(c)) if preserve_variances else np.eye(c.shape[0]) * np.mean(np.diag(c))
    out = (1.0 - alpha) * c + alpha * target
    _validate_covariance(out, positive_definite=True)
    return out


@dataclass(frozen=True, slots=True)
class LikelihoodDiagnostics:
    family: str
    n_data: int
    n_mocks: int | None
    chi_squared: float
    log_determinant: float
    log_likelihood: float
    precision_factor: float
    covariance_condition_number: float
    residual_rms_sigma: float
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "n_data": self.n_data,
            "n_mocks": self.n_mocks,
            "chi_squared": self.chi_squared,
            "log_determinant": self.log_determinant,
            "log_likelihood": self.log_likelihood,
            "precision_factor": self.precision_factor,
            "covariance_condition_number": self.covariance_condition_number,
            "residual_rms_sigma": self.residual_rms_sigma,
            "caveats": list(self.caveats),
        }


def evaluate_cosmology_likelihood(
    observed: np.ndarray,
    prediction: np.ndarray,
    covariance: np.ndarray,
    *,
    family: LikelihoodKind = "gaussian",
    n_mocks: int | None = None,
    apply_hartlap: bool = False,
) -> LikelihoodDiagnostics:
    """Evaluate a multivariate Gaussian or Sellentin--Heavens finite-mock t likelihood.

    For Sellentin--Heavens, ``covariance`` must be the unbiased sample covariance from ``n_mocks``
    realizations and ``n_mocks > n_data``. For a Gaussian using sample covariance, optional Hartlap
    scaling changes the chi-square precision but not the determinant; this reports the exact chosen
    convention rather than pretending it is a fully normalized finite-mock likelihood.
    """
    y, m = np.asarray(observed, float), np.asarray(prediction, float)
    if y.ndim != 1 or m.shape != y.shape or y.size == 0 or np.any(~np.isfinite(y + m)):
        raise ValueError("observed and prediction must be finite matching 1-D vectors")
    c = _validate_covariance(covariance, positive_definite=True)
    if c.shape != (y.size, y.size):
        raise ValueError("covariance shape must match data vector")
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    residual = y - m
    white = np.linalg.solve(chol, residual)
    chi2 = float(white @ white)
    logdet = float(2 * np.log(np.diag(chol)).sum())
    p = y.size
    factor = 1.0
    if family == "gaussian":
        if apply_hartlap:
            if n_mocks is None:
                raise ValueError("n_mocks is required for Hartlap correction")
            factor = hartlap_precision_factor(n_mocks, p)
            chi2 *= factor
        loglike = -0.5 * (chi2 + logdet + p * math.log(2 * math.pi))
        caveats = ("Gaussian likelihood assumes the covariance is treated as known.",)
        if apply_hartlap:
            caveats += (
                "Hartlap factor corrected chi-square precision; determinant was not rescaled, so this is not a fully normalized finite-mock likelihood.",
            )
    elif family == "sellentin_heavens":
        if n_mocks is None or n_mocks <= p:
            raise ValueError("Sellentin-Heavens likelihood requires n_mocks > n_data")
        if apply_hartlap:
            raise ValueError("do not combine Hartlap scaling with the Sellentin-Heavens likelihood")
        ns = float(n_mocks)
        loglike = (
            gammaln(ns / 2)
            - gammaln((ns - p) / 2)
            - (p / 2) * math.log(math.pi * (ns - 1))
            - 0.5 * logdet
            - (ns / 2) * math.log1p(chi2 / (ns - 1))
        )
        caveats = (
            "Requires an unbiased sample covariance from representative independent mocks.",
            "Finite-mock likelihood does not repair an incorrect covariance or mismatched mock pipeline.",
        )
    else:
        raise ValueError("family must be 'gaussian' or 'sellentin_heavens'")
    eig = np.linalg.eigvalsh(c)
    return LikelihoodDiagnostics(
        family,
        p,
        n_mocks,
        chi2,
        logdet,
        float(loglike),
        factor,
        float(eig.max() / eig.min()),
        float(np.sqrt(np.mean(white**2))),
        caveats,
    )


def covariance_regularization_scan(
    covariance: np.ndarray,
    shrinkage_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
) -> list[dict[str, float | bool]]:
    """Summarize conditioning and inversion stability across declared diagonal shrinkage values."""
    c = _validate_covariance(covariance, positive_definite=False)
    rows = []
    for alpha in shrinkage_values:
        shrunk = diagonal_target_shrinkage(c, float(alpha))
        eig = np.linalg.eigvalsh(shrunk)
        rows.append(
            {
                "alpha": float(alpha),
                "min_eigenvalue": float(eig.min()),
                "max_eigenvalue": float(eig.max()),
                "condition_number": float(eig.max() / eig.min()),
                "positive_definite": bool(eig.min() > 0),
                "inverse_finite": bool(np.all(np.isfinite(np.linalg.inv(shrunk)))),
            }
        )
    return rows


__all__ = [
    "LikelihoodDiagnostics",
    "sample_covariance_from_mocks",
    "hartlap_precision_factor",
    "invert_sample_covariance",
    "diagonal_target_shrinkage",
    "evaluate_cosmology_likelihood",
    "covariance_regularization_scan",
]
