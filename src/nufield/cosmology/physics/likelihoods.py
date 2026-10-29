"""Small, explicit likelihood functions for observable-level cosmology benchmarks."""

from __future__ import annotations

import math

import numpy as np


def gaussian_log_likelihood(
    observed: np.ndarray,
    predicted: np.ndarray,
    covariance: np.ndarray,
    *,
    include_normalization: bool = True,
) -> float:
    """Evaluate a multivariate Gaussian log likelihood with a fixed covariance matrix."""
    y, mu, cov = (
        np.asarray(observed, float),
        np.asarray(predicted, float),
        np.asarray(covariance, float),
    )
    if y.ndim != 1 or mu.shape != y.shape or cov.shape != (y.size, y.size):
        raise ValueError("observed/predicted must be vectors and covariance must match length")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(mu)) or np.any(~np.isfinite(cov)):
        raise ValueError("likelihood inputs must be finite")
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")
    residual = y - mu
    chi2 = float(residual @ np.linalg.solve(cov, residual))
    norm = y.size * math.log(2 * math.pi) + float(logdet) if include_normalization else 0.0
    return -0.5 * (chi2 + norm)


def gaussian_chi_squared(
    observed: np.ndarray, predicted: np.ndarray, covariance: np.ndarray
) -> float:
    """Return residual chi-squared; covariance normalization is not included."""
    y, mu, cov = (
        np.asarray(observed, float),
        np.asarray(predicted, float),
        np.asarray(covariance, float),
    )
    if y.ndim != 1 or mu.shape != y.shape or cov.shape != (y.size, y.size):
        raise ValueError("observed/predicted/covariance shapes differ")
    residual = y - mu
    return float(residual @ np.linalg.solve(cov, residual))


def covariance_shrinkage(
    covariance: np.ndarray, shrinkage: float, target: np.ndarray | None = None
) -> np.ndarray:
    """Linear covariance shrinkage C_reg=(1-lambda)C+lambda T.

    The target defaults to the diagonal of C. This is a numerical regularizer, not a substitute for
    estimating covariance uncertainty or using an appropriate finite-mock likelihood correction.
    """
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("covariance must be square and symmetric")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0,1]")
    t = np.diag(np.diag(c)) if target is None else np.asarray(target, float)
    if t.shape != c.shape or not np.allclose(t, t.T, atol=1e-10):
        raise ValueError("target must be symmetric and match covariance shape")
    result = (1 - shrinkage) * c + shrinkage * t
    if np.linalg.eigvalsh(result).min() <= 0:
        raise ValueError("shrinkage target/result is not positive definite")
    return result


__all__ = ["gaussian_log_likelihood", "gaussian_chi_squared", "covariance_shrinkage"]
