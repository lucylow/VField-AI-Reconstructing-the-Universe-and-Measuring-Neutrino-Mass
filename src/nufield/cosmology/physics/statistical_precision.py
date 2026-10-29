"""Finite-mock covariance diagnostics for cosmological summary-statistic likelihoods."""

from __future__ import annotations

import numpy as np


def sample_covariance(samples: np.ndarray, *, rowvar: bool = False) -> np.ndarray:
    """Estimate an unbiased sample covariance from independent mock vectors.

    Input rows are independent simulations by default. Parent cosmological simulations or related
    phase pairs must not be treated as independent merely because filenames differ.
    """
    x = np.asarray(samples, float)
    if x.ndim != 2 or x.shape[0] < 2 or np.any(~np.isfinite(x)):
        raise ValueError("samples must be a finite 2-D matrix with at least two rows")
    return np.cov(x, rowvar=rowvar, ddof=1)


def hartlap_precision_factor(n_mocks: int, n_data_vector: int) -> float:
    r"""Return Hartlap factor (N_s-p-2)/(N_s-1), only when N_s>p+2.

    This correction concerns an inverse Wishart covariance estimate under specific assumptions. It is
    not a universal finite-mock correction and must not be applied outside its validity conditions.
    """
    if n_mocks < 2 or n_data_vector < 1:
        raise ValueError("n_mocks>=2 and n_data_vector>=1 required")
    if n_mocks <= n_data_vector + 2:
        raise ValueError("Hartlap correction undefined: need n_mocks > n_data_vector + 2")
    return float((n_mocks - n_data_vector - 2) / (n_mocks - 1))


def covariance_estimation_warning(n_mocks: int, n_data_vector: int) -> list[str]:
    """Return warnings that can be attached to experiment manifests."""
    warnings = []
    if n_mocks <= n_data_vector:
        warnings.append(
            "sample covariance is rank-deficient or singular when N_mocks <= N_data_vector"
        )
    if n_mocks <= n_data_vector + 2:
        warnings.append("Hartlap precision correction is undefined for this mock/data-vector ratio")
    elif n_mocks < 10 * n_data_vector:
        warnings.append("finite-mock covariance uncertainty may materially affect parameter errors")
    warnings.append("Check independence, simulation design, and likelihood correction assumptions.")
    return warnings


def chi_square_summary(
    residual: np.ndarray, covariance: np.ndarray, n_parameters_fit: int = 0
) -> dict[str, float]:
    """Summarize chi² and nominal degrees of freedom for a supplied covariance."""
    r, c = np.asarray(residual, float), np.asarray(covariance, float)
    if r.ndim != 1 or c.shape != (r.size, r.size) or n_parameters_fit < 0:
        raise ValueError("residual/covariance dimensions or fitted parameter count invalid")
    if not np.allclose(c, c.T, atol=1e-10) or np.linalg.eigvalsh(c).min() <= 0:
        raise ValueError("covariance must be symmetric positive definite")
    chi2 = float(r @ np.linalg.solve(c, r))
    dof = max(0, r.size - n_parameters_fit)
    return {
        "chi2": chi2,
        "dof": float(dof),
        "reduced_chi2": chi2 / dof if dof > 0 else float("nan"),
    }


__all__ = [
    "sample_covariance",
    "hartlap_precision_factor",
    "covariance_estimation_warning",
    "chi_square_summary",
]
