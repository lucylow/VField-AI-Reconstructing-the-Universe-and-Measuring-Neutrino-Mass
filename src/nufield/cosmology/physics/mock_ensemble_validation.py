"""Ensemble-level validation for simulation/mock power spectra.

These tests compare repeated realization summaries with a supplied reference spectrum and quantify
finite-ensemble uncertainty. They do not claim that the reference is correct or that simulation boxes
are independent. Keep paired seeds, shared initial conditions, redshift, volume and estimator metadata
explicit when comparing cosmological simulations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True, slots=True)
class MockEnsembleReport:
    k_mpc_inv: np.ndarray
    reference_power_mpc3: np.ndarray
    mean_mock_power_mpc3: np.ndarray
    fractional_mean_bias: np.ndarray
    standard_error_of_mean_mpc3: np.ndarray
    covariance_mpc6: np.ndarray
    correlation: np.ndarray
    chi2_mean_residual: float
    dof: int
    chi2_p_value: float
    n_mocks: int
    n_bins: int
    covariance_condition_number: float
    caveats: tuple[str, ...]

    def to_dict(self, *, include_matrices: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "chi2_mean_residual": self.chi2_mean_residual,
            "dof": self.dof,
            "chi2_p_value": self.chi2_p_value,
            "n_mocks": self.n_mocks,
            "n_bins": self.n_bins,
            "covariance_condition_number": self.covariance_condition_number,
            "caveats": list(self.caveats),
            "units": {"k": "Mpc^-1", "P": "Mpc^3", "Cov(P)": "Mpc^6"},
        }
        out["k_mpc_inv"] = self.k_mpc_inv.tolist()
        out["reference_power_mpc3"] = self.reference_power_mpc3.tolist()
        out["mean_mock_power_mpc3"] = self.mean_mock_power_mpc3.tolist()
        out["fractional_mean_bias"] = self.fractional_mean_bias.tolist()
        out["standard_error_of_mean_mpc3"] = self.standard_error_of_mean_mpc3.tolist()
        if include_matrices:
            out["covariance_mpc6"] = self.covariance_mpc6.tolist()
            out["correlation"] = self.correlation.tolist()
        return out


def validate_mock_power_ensemble(
    k_mpc_inv: np.ndarray,
    mock_power_mpc3: np.ndarray,
    reference_power_mpc3: np.ndarray,
    *,
    covariance_regularization: float = 0.0,
) -> MockEnsembleReport:
    """Estimate ensemble mean/covariance and evaluate residual of reference against sample mean.

    ``mock_power_mpc3`` must have shape (n_mock,n_k). We use the unbiased sample covariance with
    denominator N-1. The optional diagonal regularization is a fraction of the mean diagonal variance;
    it is reported as a numerical stabilization, not a physical covariance contribution.
    """
    k, mocks, ref = (
        np.asarray(k_mpc_inv, float),
        np.asarray(mock_power_mpc3, float),
        np.asarray(reference_power_mpc3, float),
    )
    if (
        k.ndim != 1
        or len(k) < 2
        or mocks.ndim != 2
        or mocks.shape[1] != k.size
        or ref.shape != k.shape
    ):
        raise ValueError("k, mock power (n_mock,n_k), and reference must have consistent shapes")
    if (
        mocks.shape[0] < 3
        or np.any(~np.isfinite(k))
        or np.any(~np.isfinite(mocks))
        or np.any(~np.isfinite(ref))
    ):
        raise ValueError("need at least three finite mock spectra and finite reference/k")
    if np.any(k <= 0) or np.any(np.diff(k) <= 0) or np.any(mocks <= 0) or np.any(ref <= 0):
        raise ValueError("k and all powers must be positive, with k strictly increasing")
    if not np.isfinite(covariance_regularization) or covariance_regularization < 0:
        raise ValueError("covariance_regularization must be finite and non-negative")
    mean = mocks.mean(axis=0)
    cov = np.cov(mocks, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    shrink = covariance_regularization * float(np.mean(np.diag(cov)))
    if shrink:
        cov = cov + np.eye(k.size) * shrink
    eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    if eig.min() <= 0:
        raise ValueError(
            "mock covariance is not positive definite; reduce bins, increase mock count, or justify shrinkage"
        )
    diag = np.diag(cov)
    corr = cov / np.sqrt(np.outer(diag, diag))
    residual = mean - ref
    chi_value = float(residual @ np.linalg.solve(cov, residual))
    dof = int(np.linalg.matrix_rank(cov))
    pvalue = float(chi2.sf(chi_value, dof))
    return MockEnsembleReport(
        k.copy(),
        ref.copy(),
        mean,
        (mean - ref) / ref,
        np.sqrt(diag / mocks.shape[0]),
        cov,
        corr,
        chi_value,
        dof,
        pvalue,
        mocks.shape[0],
        k.size,
        float(eig[-1] / eig[0]),
        (
            "The p-value is conditional on the supplied reference and empirical covariance.",
            "Mocks must represent the same target distribution and estimator as the reference.",
            "A finite mock ensemble can give a singular or noisy covariance; shrinkage changes the likelihood model.",
            "If mock spectra share initial conditions or parent boxes, independence assumptions must be documented.",
        ),
    )


def paired_mock_fractional_differences(
    power_a: np.ndarray,
    power_b: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    """Compare matched mock spectra, preserving pairing rather than treating samples as independent."""
    a, b = np.asarray(power_a, float), np.asarray(power_b, float)
    if a.ndim != 2 or b.shape != a.shape or a.shape[0] < 2 or np.any(~np.isfinite(a + b)):
        raise ValueError("power_a and power_b must be matching finite (n_mock>=2,n_k) arrays")
    if np.any(a <= 0) or np.any(b <= 0) or epsilon <= 0:
        raise ValueError("powers must be positive and epsilon positive")
    log_ratio = np.log(b) - np.log(a)
    return {
        "mean_log_ratio": np.mean(log_ratio, axis=0),
        "std_log_ratio": np.std(log_ratio, axis=0, ddof=1),
        "paired_rms_log_ratio": float(np.sqrt(np.mean(log_ratio**2))),
        "paired_max_abs_log_ratio": float(np.max(np.abs(log_ratio))),
        "n_paired_mocks": a.shape[0],
    }


__all__ = [
    "MockEnsembleReport",
    "validate_mock_power_ensemble",
    "paired_mock_fractional_differences",
]
