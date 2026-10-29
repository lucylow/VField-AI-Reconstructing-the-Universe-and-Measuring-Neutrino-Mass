"""Multivariate Gaussian summary-likelihood baseline.

This is useful as a transparent benchmark only when the summary-vector distribution and covariance
assumptions are justified. The starter does not supply a physical covariance model.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp


class GaussianSummaryLikelihood:
    def __init__(self, mean: np.ndarray, covariance: np.ndarray, jitter: float = 1e-8):
        mu, cov = np.asarray(mean, float), np.asarray(covariance, float)
        if (
            mu.ndim != 1
            or cov.shape != (len(mu), len(mu))
            or not np.isfinite(mu).all()
            or not np.isfinite(cov).all()
        ):
            raise ValueError("invalid mean or covariance shape/values")
        if jitter < 0 or not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError("covariance must be symmetric and jitter non-negative")
        self.mean = mu
        self.covariance = cov + jitter * np.eye(len(mu))
        self._factor = cho_factor(self.covariance, lower=True, check_finite=True)
        self._logdet = float(2 * np.log(np.diag(self._factor[0])).sum())

    def logpdf(self, value: np.ndarray) -> float:
        x = np.asarray(value, float)
        if x.shape != self.mean.shape or not np.isfinite(x).all():
            raise ValueError("value must match summary dimension and be finite")
        residual = x - self.mean
        quad = float(residual @ cho_solve(self._factor, residual))
        return float(-0.5 * (len(x) * np.log(2 * np.pi) + self._logdet + quad))

    def whitened_residual(self, value: np.ndarray) -> np.ndarray:
        x = np.asarray(value, float)
        if x.shape != self.mean.shape:
            raise ValueError("value shape mismatch")
        # Triangular solve is used only on the lower Cholesky factor.
        from scipy.linalg import solve_triangular

        return solve_triangular(self._factor[0], x - self.mean, lower=True).astype(np.float32)


def gaussian_log_mixture(
    log_components: np.ndarray, log_weights: np.ndarray | None = None
) -> float:
    """Stable log-sum-exp aggregation over mixture components."""
    logs = np.asarray(log_components, float)
    if logs.ndim != 1 or len(logs) == 0 or np.isnan(logs).any():
        raise ValueError("log_components must be a non-empty vector without NaNs")
    if log_weights is None:
        lw = np.full(len(logs), -np.log(len(logs)))
    else:
        w = np.asarray(log_weights, float)
        if w.shape != logs.shape or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("log_weights input should be non-negative unnormalized weights")
        w = w / w.sum()
        lw = np.log(np.maximum(w, np.finfo(float).tiny))
    return float(logsumexp(logs + lw))
