"""Likelihood-space diagnostics for theory vectors and correlated cosmological observations.

These routines help identify model failures and data-vector sensitivity. They do not select a preferred
likelihood family automatically: data transformations, covariance estimates, nonlinear corrections and
nuisance assumptions must be specified by the analysis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Convert a finite positive-variance covariance matrix to its dimensionless correlation matrix."""
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or np.any(~np.isfinite(c)):
        raise ValueError("covariance must be finite and square")
    if not np.allclose(c, c.T, rtol=1e-9, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    diag = np.diag(c)
    if np.any(diag <= 0):
        raise ValueError("all covariance diagonal entries must be positive")
    scale = np.sqrt(diag)
    corr = c / np.outer(scale, scale)
    return np.clip(0.5 * (corr + corr.T), -1.0, 1.0)


def whiten_residuals(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return Cholesky-whitened residual coordinates ``L^-1 r`` for C=LLᵀ."""
    r, c = np.asarray(residual, float), np.asarray(covariance, float)
    if r.ndim != 1 or c.shape != (r.size, r.size) or np.any(~np.isfinite(r)):
        raise ValueError("residual and covariance dimensions do not match")
    if not np.allclose(c, c.T, atol=1e-11, rtol=1e-9):
        raise ValueError("covariance must be symmetric")
    try:
        l = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return solve_triangular(l, r, lower=True, check_finite=False)


def leave_one_bin_out_sensitivity(
    residual: np.ndarray,
    covariance: np.ndarray,
) -> list[dict[str, float | int]]:
    """Measure how much the full chi-square changes if each single bin is omitted.

    A reduced covariance is formed by deleting the corresponding row and column; the statistic is
    therefore a leave-one-coordinate-out diagnostic, not an independent-bin decomposition of χ².
    """
    r, c = np.asarray(residual, float), np.asarray(covariance, float)
    if r.ndim != 1 or c.shape != (r.size, r.size) or r.size < 2:
        raise ValueError("need residual and matching covariance with at least two coordinates")
    if (
        np.any(~np.isfinite(r))
        or np.any(~np.isfinite(c))
        or not np.allclose(c, c.T, atol=1e-11, rtol=1e-9)
    ):
        raise ValueError("residual/covariance must be finite and covariance symmetric")
    full_w = whiten_residuals(r, c)
    full = float(full_w @ full_w)
    rows = []
    for idx in range(r.size):
        keep = np.arange(r.size) != idx
        try:
            w = whiten_residuals(r[keep], c[np.ix_(keep, keep)])
            reduced = float(w @ w)
        except ValueError:
            reduced = float("nan")
        rows.append(
            {
                "omitted_index": int(idx),
                "full_chi_squared": full,
                "chi_squared_without_bin": reduced,
                "delta_chi_squared": float(full - reduced)
                if np.isfinite(reduced)
                else float("nan"),
            }
        )
    return rows


def covariance_eigenmode_report(
    covariance: np.ndarray, *, max_modes: int = 12
) -> dict[str, object]:
    """Summarize principal covariance modes from largest to smallest variance."""
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 1 or np.any(~np.isfinite(c)):
        raise ValueError("covariance must be a finite square matrix")
    if not np.allclose(c, c.T, atol=1e-11, rtol=1e-9):
        raise ValueError("covariance must be symmetric")
    if max_modes < 1:
        raise ValueError("max_modes must be positive")
    values, vectors = np.linalg.eigh(c)
    if values.min() <= 0:
        raise ValueError("covariance must be positive definite")
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    frac = values / values.sum()
    return {
        "eigenvalues_descending": values[:max_modes].tolist(),
        "fractional_variance_descending": frac[:max_modes].tolist(),
        "cumulative_fraction": np.cumsum(frac[:max_modes]).tolist(),
        "condition_number": float(values[0] / values[-1]),
        "eigenvectors_columns": vectors[:, :max_modes].tolist(),
        "number_of_modes": int(values.size),
    }


@dataclass(frozen=True, slots=True)
class ResidualBinDiagnostic:
    label: str
    residual: float
    marginal_sigma: float
    marginal_pull: float
    whitened_pull: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "label": self.label,
            "residual": self.residual,
            "marginal_sigma": self.marginal_sigma,
            "marginal_pull": self.marginal_pull,
            "whitened_pull": self.whitened_pull,
        }


def residual_bin_diagnostics(
    observed: np.ndarray,
    predicted: np.ndarray,
    covariance: np.ndarray,
    *,
    labels: Sequence[str] | None = None,
) -> tuple[ResidualBinDiagnostic, ...]:
    """Return marginal and whitened residual diagnostics in the given vector ordering."""
    y, m, c = (
        np.asarray(observed, float),
        np.asarray(predicted, float),
        np.asarray(covariance, float),
    )
    if y.ndim != 1 or m.shape != y.shape or c.shape != (y.size, y.size) or y.size == 0:
        raise ValueError("observed, predicted and covariance dimensions must agree")
    if np.any(~np.isfinite(y + m)):
        raise ValueError("observed and predicted must be finite")
    labs = tuple(
        str(x) for x in (labels if labels is not None else [f"bin_{i}" for i in range(y.size)])
    )
    if len(labs) != y.size:
        raise ValueError("label count must match data vector")
    residual = y - m
    sigma = np.sqrt(np.diag(c))
    if np.any(sigma <= 0):
        raise ValueError("covariance variances must be positive")
    whitened = whiten_residuals(residual, c)
    return tuple(
        ResidualBinDiagnostic(
            labs[i],
            float(residual[i]),
            float(sigma[i]),
            float(residual[i] / sigma[i]),
            float(whitened[i]),
        )
        for i in range(y.size)
    )


__all__ = [
    "correlation_from_covariance",
    "whiten_residuals",
    "leave_one_bin_out_sensitivity",
    "covariance_eigenmode_report",
    "ResidualBinDiagnostic",
    "residual_bin_diagnostics",
]
