"""Stable covariance estimation utilities for summary-statistic likelihood baselines.

These helpers are not a replacement for a survey covariance analysis. They are intended for controlled
simulation experiments with explicit independent realization units and transparent regularization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CovarianceReport:
    n_observations: int
    n_features: int
    shrinkage: float
    diagonal_floor: float
    empirical_rank: int
    raw_condition_number: float | None
    regularized_condition_number: float
    minimum_eigenvalue: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_regularized_covariance(
    residuals: np.ndarray, *, shrinkage: float = 0.1, diagonal_floor: float = 1e-6
) -> tuple[np.ndarray, CovarianceReport]:
    """Shrink a sample covariance toward its diagonal and floor marginal variances.

    Parameters must already be in the fixed feature coordinates chosen from the training partition.
    The shrinkage coefficient is a declared hyperparameter unless selected by an inner validation
    protocol. It must not be tuned using the final test data.
    """
    x = np.asarray(residuals, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("residuals must be finite [N>=2,K>=1]")
    if not 0 <= shrinkage <= 1 or diagonal_floor <= 0:
        raise ValueError("shrinkage must be [0,1] and diagonal_floor positive")
    centered = x - x.mean(axis=0, keepdims=True)
    raw = np.atleast_2d(np.cov(centered, rowvar=False, ddof=1))
    if raw.shape != (x.shape[1], x.shape[1]):
        raise RuntimeError("covariance shape mismatch")
    raw = (raw + raw.T) / 2
    diag = np.maximum(np.diag(raw), diagonal_floor)
    target = np.diag(diag)
    cov = (1 - shrinkage) * raw + shrinkage * target
    cov = (cov + cov.T) / 2
    # Add only enough scale-aware jitter to make a Cholesky factorization possible.
    scale = max(float(np.mean(np.diag(cov))), diagonal_floor)
    jitter = diagonal_floor * scale
    for _ in range(8):
        eig = np.linalg.eigvalsh(cov)
        if eig.min() > 0:
            break
        cov = cov + np.eye(len(diag)) * jitter
        jitter *= 10
    eig = np.linalg.eigvalsh(cov)
    if eig.min() <= 0:
        raise np.linalg.LinAlgError("could not regularize covariance to positive definite")
    raw_eigs = np.linalg.eigvalsh(raw)
    raw_condition = (
        float(raw_eigs.max() / max(raw_eigs.min(), 1e-30)) if raw_eigs.min() > 0 else None
    )
    report = CovarianceReport(
        len(x),
        x.shape[1],
        float(shrinkage),
        float(diagonal_floor),
        int(np.linalg.matrix_rank(raw)),
        raw_condition,
        float(eig.max() / eig.min()),
        float(eig.min()),
    )
    return cov, report


def multivariate_gaussian_logpdf(
    observation: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> float:
    """Numerically stable log N(x | mean, covariance) via Cholesky, without explicit inverse."""
    x = np.asarray(observation, dtype=np.float64).reshape(-1)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    c = np.asarray(covariance, dtype=np.float64)
    if x.shape != mu.shape or c.shape != (len(x), len(x)):
        raise ValueError("observation/mean/covariance dimensions differ")
    if not np.isfinite(x).all() or not np.isfinite(mu).all() or not np.isfinite(c).all():
        raise ValueError("logpdf inputs must be finite")
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    residual = np.linalg.solve(chol, x - mu)
    return float(
        -0.5 * (len(x) * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + residual @ residual)
    )


def normalized_discrete_grid_prior(grid: np.ndarray, low: float, high: float) -> np.ndarray:
    """Approximate a continuous uniform prior on a one-dimensional grid using Voronoi cell widths.

    This makes the discrete approximation explicit. It is inaccurate near a poorly covered or
    truncated grid; the report should flag when grid endpoints do not cover the prior support.
    """
    g = np.asarray(grid, dtype=float).reshape(-1)
    if not len(g) or not np.isfinite(g).all() or np.any(np.diff(g) <= 0):
        raise ValueError("grid must be finite, strictly increasing, and nonempty")
    if not low < high or g[0] < low - 1e-10 or g[-1] > high + 1e-10:
        raise ValueError("grid must lie within ordered prior bounds")
    if len(g) == 1:
        return np.ones(1)
    edges = np.empty(len(g) + 1)
    edges[0] = low
    edges[-1] = high
    edges[1:-1] = (g[:-1] + g[1:]) / 2
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("grid/prior widths invalid")
    return widths / widths.sum()
