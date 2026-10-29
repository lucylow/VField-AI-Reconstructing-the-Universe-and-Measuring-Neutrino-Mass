"""Explicit theory-discrepancy models for cosmological likelihood stress tests.

The covariance kernels here represent a declared uncertainty model for a missing/approximate theory
term. They do not discover the correct theory-error amplitude from data. Use only after defining a
scientifically defensible error envelope and checking sensitivity to its hyperparameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TheoryErrorCovariance:
    covariance: np.ndarray
    fractional_envelope: np.ndarray
    correlation_length_ln_k: float
    kernel: str
    mean_fractional_error: float
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        c = np.asarray(self.covariance, float)
        e = np.asarray(self.fractional_envelope, float)
        if c.ndim != 2 or c.shape != (len(e), len(e)) or e.ndim != 1:
            raise ValueError("theory-error covariance/envelope shape mismatch")
        if np.any(~np.isfinite(c)) or np.any(~np.isfinite(e)) or np.any(e < 0):
            raise ValueError("theory-error arrays must be finite; envelope must be non-negative")
        if not np.allclose(c, c.T, rtol=0, atol=1e-10):
            raise ValueError("theory-error covariance must be symmetric")
        if np.linalg.eigvalsh(c).min() < -1e-10 * max(float(np.max(np.diag(c))), 1.0):
            raise ValueError("theory-error covariance must be positive semidefinite")
        object.__setattr__(self, "covariance", c.copy())
        object.__setattr__(self, "fractional_envelope", e.copy())


def rbf_correlation_ln_k(k_mpc_inv: np.ndarray, correlation_length_ln_k: float) -> np.ndarray:
    r"""Return R_ij=exp[-(ln k_i-ln k_j)^2/(2ℓ²)] for a positive physical k grid."""
    k = np.asarray(k_mpc_inv, float)
    if (
        k.ndim != 1
        or len(k) < 2
        or np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError("k must be finite, positive, increasing and contain >=2 values")
    if not math.isfinite(correlation_length_ln_k) or correlation_length_ln_k <= 0:
        raise ValueError("correlation length in ln k must be positive and finite")
    x = np.log(k)
    delta = x[:, None] - x[None, :]
    r = np.exp(-0.5 * (delta / correlation_length_ln_k) ** 2)
    return 0.5 * (r + r.T)


def fractional_theory_error_covariance(
    k_mpc_inv: np.ndarray,
    theory_vector: np.ndarray,
    fractional_envelope: float | np.ndarray,
    *,
    correlation_length_ln_k: float = 0.25,
    kernel: str = "rbf",
) -> TheoryErrorCovariance:
    r"""Construct an absolute theory-error covariance from a fractional envelope.

    ``C_th,ij = P_i σ_i R_ij σ_j P_j``. This is most naturally used for a positive P(k) data vector;
    if the observable includes signed cross-bandpowers, supply an appropriate absolute envelope and do
    not interpret the fractional parameter as a percentage of a negative theory value.
    """
    k, p = np.asarray(k_mpc_inv, float), np.asarray(theory_vector, float)
    if k.ndim != 1 or p.shape != k.shape or len(k) < 2 or np.any(~np.isfinite(k + p)):
        raise ValueError("k and theory_vector must be matching finite vectors of length >=2")
    if np.any(k <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError("k must be positive and strictly increasing")
    env = np.asarray(fractional_envelope, float)
    if env.ndim == 0:
        env = np.full_like(k, float(env))
    if env.shape != k.shape or np.any(~np.isfinite(env)) or np.any(env < 0):
        raise ValueError(
            "fractional error envelope must be scalar or match k with non-negative values"
        )
    if kernel != "rbf":
        raise ValueError("only the explicit 'rbf' correlation kernel is currently implemented")
    corr = rbf_correlation_ln_k(k, correlation_length_ln_k)
    sigma_abs = np.abs(p) * env
    cov = sigma_abs[:, None] * corr * sigma_abs[None, :]
    return TheoryErrorCovariance(
        cov,
        env,
        float(correlation_length_ln_k),
        kernel,
        float(np.mean(env)),
        (
            "The envelope is analyst-specified and is not inferred from this function.",
            "A smooth RBF correlation in ln(k) is a modeling choice, not a derived physical covariance.",
            "Repeat inference over multiple envelope amplitudes and correlation lengths.",
        ),
    )


def add_covariance_components(
    *components: np.ndarray, symmetry_tolerance: float = 1e-9
) -> np.ndarray:
    """Validate and sum covariance matrices without silently repairing invalid input."""
    mats = [np.asarray(c, float) for c in components]
    if not mats:
        raise ValueError("at least one covariance component is required")
    shape = mats[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("covariances must be square")
    for c in mats:
        if (
            c.shape != shape
            or np.any(~np.isfinite(c))
            or not np.allclose(c, c.T, atol=symmetry_tolerance, rtol=0)
        ):
            raise ValueError(
                "all covariance components must share square shape, finite entries and symmetry"
            )
        eig = np.linalg.eigvalsh(c)
        scale = max(float(np.max(np.abs(np.diag(c)))), 1.0)
        if eig.min() < -symmetry_tolerance * scale:
            raise ValueError("a covariance component is not positive semidefinite")
    out = sum(mats[1:], start=mats[0].copy())
    return 0.5 * (out + out.T)


def covariance_sensitivity_grid(
    data_covariance: np.ndarray,
    theory_error_covariance: np.ndarray,
    amplitudes: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> list[dict[str, float | int]]:
    """Report log-determinant and effective whitened information as theory error is rescaled."""
    c, t = np.asarray(data_covariance, float), np.asarray(theory_error_covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or t.shape != c.shape:
        raise ValueError("data and theory-error covariance must be same-shape square matrices")
    if not np.allclose(c, c.T, atol=1e-10) or not np.allclose(t, t.T, atol=1e-10):
        raise ValueError("covariance matrices must be symmetric")
    if np.linalg.eigvalsh(c).min() <= 0 or np.linalg.eigvalsh(t).min() < -1e-10:
        raise ValueError("data covariance must be positive definite and theory covariance PSD")
    amps = tuple(float(x) for x in amplitudes)
    if not amps or any(not math.isfinite(a) or a < 0 for a in amps):
        raise ValueError("amplitudes must be a non-empty sequence of finite non-negative scalars")
    out = []
    n = c.shape[0]
    for amp in amps:
        ct = c + amp * amp * t
        sign, logdet = np.linalg.slogdet(ct)
        if sign <= 0:
            raise ArithmeticError("combined covariance lost positive determinant")
        eig = np.linalg.eigvalsh(ct)
        out.append(
            {
                "amplitude_scale": amp,
                "logdet": float(logdet),
                "condition_number": float(eig[-1] / eig[0]),
                "n_data_bins": n,
                "mean_diagonal_variance": float(np.mean(np.diag(ct))),
            }
        )
    return out


__all__ = [
    "TheoryErrorCovariance",
    "rbf_correlation_ln_k",
    "fractional_theory_error_covariance",
    "add_covariance_components",
    "covariance_sensitivity_grid",
]
