"""First-order parameter shifts induced by a declared theory/data systematic.

For local Gaussian inference with Jacobian J=∂m/∂θ and covariance C, this module computes the linear
parameter bias Δθ ≈ (JᵀC⁻¹J)⁻¹ JᵀC⁻¹ Δm. This diagnoses sensitivity to a specified mismatch; it is not
a replacement for rerunning the full nonlinear posterior when the shift is large.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass(frozen=True, slots=True)
class ParameterBiasReport:
    parameter_names: tuple[str, ...]
    parameter_shift: np.ndarray
    fisher_matrix: np.ndarray
    parameter_covariance_local: np.ndarray
    systematic_delta_data: np.ndarray
    delta_chi2_at_unshifted_parameters: float
    bias_significance: float
    fisher_condition_number: float
    rank: int
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_names": list(self.parameter_names),
            "parameter_shift": self.parameter_shift.tolist(),
            "fisher_matrix": self.fisher_matrix.tolist(),
            "parameter_covariance_local": self.parameter_covariance_local.tolist(),
            "systematic_delta_data": self.systematic_delta_data.tolist(),
            "delta_chi2_at_unshifted_parameters": self.delta_chi2_at_unshifted_parameters,
            "bias_significance": self.bias_significance,
            "fisher_condition_number": self.fisher_condition_number,
            "rank": self.rank,
            "interpretation": self.interpretation,
            "caveat": "Linearized Gaussian approximation; rerun the full inference for substantial shifts.",
        }


def linear_parameter_bias(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    systematic_delta_data: np.ndarray,
    *,
    parameter_names: tuple[str, ...] | None = None,
    svd_rcond: float = 1e-10,
    require_full_rank: bool = True,
) -> ParameterBiasReport:
    """Compute first-order parameter bias with Cholesky whitening and rank-aware Fisher inversion."""
    j, c, delta = (
        np.asarray(jacobian, float),
        np.asarray(covariance, float),
        np.asarray(systematic_delta_data, float),
    )
    if j.ndim != 2 or delta.ndim != 1 or j.shape[0] != delta.size:
        raise ValueError("jacobian must be (n_data,n_params) and delta_data must match n_data")
    ndata, npar = j.shape
    if (
        c.shape != (ndata, ndata)
        or npar < 1
        or np.any(~np.isfinite(j))
        or np.any(~np.isfinite(c))
        or np.any(~np.isfinite(delta))
    ):
        raise ValueError("invalid or non-finite Jacobian, covariance or systematic vector")
    if not np.allclose(c, c.T, atol=1e-10, rtol=0):
        raise ValueError("covariance must be symmetric")
    if not 0 < svd_rcond < 1:
        raise ValueError("svd_rcond must lie in (0,1)")
    names = tuple(parameter_names or tuple(f"theta_{i}" for i in range(npar)))
    if len(names) != npar or len(set(names)) != npar:
        raise ValueError("parameter_names must be unique and match the Jacobian columns")
    try:
        cf = cho_factor(c, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("data covariance must be positive definite") from exc
    c_inv_j = cho_solve(cf, j, check_finite=False)
    c_inv_delta = cho_solve(cf, delta, check_finite=False)
    fisher = j.T @ c_inv_j
    fisher = 0.5 * (fisher + fisher.T)
    u, s, vt = np.linalg.svd(fisher, full_matrices=False)
    cutoff = svd_rcond * s[0] if len(s) and s[0] > 0 else 0.0
    rank = int(np.count_nonzero(s > cutoff))
    if require_full_rank and rank < npar:
        raise ValueError(
            f"Fisher matrix is rank-deficient ({rank}/{npar}); inspect parameter degeneracies"
        )
    inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > cutoff)
    fisher_inv = (vt.T * inv_s) @ u.T
    rhs = j.T @ c_inv_delta
    shift = fisher_inv @ rhs
    delta_chi2 = float(delta @ c_inv_delta)
    sig2 = float(shift @ np.linalg.pinv(fisher_inv, rcond=svd_rcond) @ shift) if rank else 0.0
    significance = math.sqrt(max(sig2, 0.0))
    condition = float(s[0] / s[-1]) if len(s) and s[-1] > cutoff else math.inf
    return ParameterBiasReport(
        names,
        shift,
        fisher,
        fisher_inv,
        delta.copy(),
        delta_chi2,
        significance,
        condition,
        rank,
        "Full-rank local Gaussian approximation"
        if rank == npar
        else "Rank-truncated approximation",
    )


def systematic_bias_fractional_envelope(
    theory_vector: np.ndarray,
    fractional_envelope: np.ndarray | float,
    jacobian: np.ndarray,
    covariance: np.ndarray,
    *,
    parameter_names: tuple[str, ...] | None = None,
) -> ParameterBiasReport:
    """Evaluate bias for one sign of an amplitude-envelope systematic, as a stress test."""
    m, env = np.asarray(theory_vector, float), np.asarray(fractional_envelope, float)
    if m.ndim != 1 or np.any(~np.isfinite(m)):
        raise ValueError("theory_vector must be a finite 1-D array")
    if env.ndim == 0:
        env = np.full_like(m, float(env))
    if env.shape != m.shape or np.any(~np.isfinite(env)) or np.any(env < 0):
        raise ValueError("fractional envelope must be non-negative and match theory")
    delta = m * env
    return linear_parameter_bias(jacobian, covariance, delta, parameter_names=parameter_names)


__all__ = ["ParameterBiasReport", "linear_parameter_bias", "systematic_bias_fractional_envelope"]
