"""Super-sample covariance (SSC) and covariance-component diagnostics.

SSC captures modulation of short-scale observables by density modes larger than an analyzed window.
The response formalism used here is C_SSC = R S_b Rᵀ, where R is the derivative of a data vector
with respect to a set of long-wavelength background modes and S_b is their covariance. For a single
background-density mode this reduces to σ_b² r rᵀ. The implementation does not supply calibrated
nonlinear responses; those must come from a validated theory model or simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson


@dataclass(frozen=True, slots=True)
class CovarianceComponentReport:
    component_names: tuple[str, ...]
    component_traces: dict[str, float]
    component_diagonal_mean: dict[str, float]
    total_trace: float
    total_min_eigenvalue: float
    total_condition_number: float
    positive_semidefinite: bool
    warnings: tuple[str, ...]


def spherical_tophat_fourier_window(k_radius: float | np.ndarray) -> np.ndarray:
    r"""Fourier transform of a unit-normalized real-space spherical top-hat."""
    x = np.asarray(k_radius, float)
    if np.any(~np.isfinite(x)):
        raise ValueError("k_radius must be finite")
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = 1 - xs**2 / 10 + xs**4 / 280 - xs**6 / 15120
    xx = x[~small]
    out[~small] = 3 * (np.sin(xx) - xx * np.cos(xx)) / xx**3
    return out


def background_density_variance(
    k_mpc_inv: np.ndarray,
    pk_mpc3: np.ndarray,
    window_fourier: np.ndarray,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
) -> float:
    r"""Compute σ_b²=∫dlnk Δ²(k)|W(k)|² over an explicitly supported interval.

    The provided ``window_fourier`` is already the Fourier-space survey-window response evaluated at
    the k samples; the function does not assume spherical geometry. It expects an isotropically
    averaged window power and a linear-theory spectrum where long modes are linear. A real survey
    window may be anisotropic and can require a multidimensional integration.
    """
    k, p, w = map(lambda v: np.asarray(v, float), (k_mpc_inv, pk_mpc3, window_fourier))
    if k.ndim != 1 or p.shape != k.shape or w.shape != k.shape or k.size < 3:
        raise ValueError("k, P and window must be matching 1-D arrays with >=3 points")
    if (
        np.any(~np.isfinite(k + p + w))
        or np.any(k <= 0)
        or np.any(p < 0)
        or np.any(np.abs(w) > 1 + 1e-6)
    ):
        raise ValueError("k must be positive, P non-negative and window magnitude <=1")
    if np.any(np.diff(k) <= 0):
        raise ValueError("k must be strictly increasing")
    lo = float(k[0]) if k_min_mpc_inv is None else float(k_min_mpc_inv)
    hi = float(k[-1]) if k_max_mpc_inv is None else float(k_max_mpc_inv)
    if lo < k[0] or hi > k[-1] or hi <= lo:
        raise ValueError("requested integration interval must be contained in k support")
    use = (k >= lo) & (k <= hi)
    if use.sum() < 3:
        raise ValueError("at least three samples must remain in the requested k interval")
    delta2 = k[use] ** 3 * p[use] / (2 * math.pi**2)
    result = float(simpson(delta2 * w[use] ** 2, x=np.log(k[use])))
    return max(result, 0.0)


def super_sample_covariance(
    response_matrix: np.ndarray,
    background_mode_covariance: np.ndarray,
    *,
    enforce_psd: bool = True,
) -> np.ndarray:
    r"""Propagate long-mode covariance via C_SSC=R S Rᵀ.

    ``response_matrix`` has shape (n_data, n_long_modes); ``background_mode_covariance`` has shape
    (n_long_modes,n_long_modes). This form permits multiple window-weighted density/tidal modes.
    """
    r = np.asarray(response_matrix, float)
    s = np.asarray(background_mode_covariance, float)
    if r.ndim != 2 or r.shape[0] < 1 or r.shape[1] < 1 or np.any(~np.isfinite(r)):
        raise ValueError("response_matrix must be finite 2-D with nonzero dimensions")
    if (
        s.shape != (r.shape[1], r.shape[1])
        or np.any(~np.isfinite(s))
        or not np.allclose(s, s.T, atol=1e-10)
    ):
        raise ValueError(
            "background mode covariance must be finite symmetric and match response columns"
        )
    eig_s = np.linalg.eigvalsh(s)
    if eig_s.min() < -1e-10 * max(1.0, float(np.max(np.abs(eig_s)))):
        raise ValueError("background mode covariance must be positive semidefinite")
    c = r @ s @ r.T
    c = 0.5 * (c + c.T)
    if enforce_psd and np.linalg.eigvalsh(c).min() < -1e-9 * max(1.0, float(np.max(np.abs(c)))):
        raise ArithmeticError("constructed SSC covariance is unexpectedly non-PSD")
    return c


def single_mode_ssc(response: np.ndarray, variance: float) -> np.ndarray:
    """Convenience rank-one SSC covariance for one background mode."""
    r = np.asarray(response, float)
    if r.ndim != 1 or r.size < 1 or np.any(~np.isfinite(r)):
        raise ValueError("response must be a finite 1-D vector")
    if variance < 0 or not math.isfinite(variance):
        raise ValueError("background variance must be finite and non-negative")
    return variance * np.outer(r, r)


def finite_difference_response(
    observable_plus: np.ndarray,
    observable_minus: np.ndarray,
    delta_plus: float,
    delta_minus: float,
) -> np.ndarray:
    """Estimate response ∂O/∂δ_b from asymmetrically spaced separate-universe runs."""
    op, om = np.asarray(observable_plus, float), np.asarray(observable_minus, float)
    if op.shape != om.shape or np.any(~np.isfinite(op + om)):
        raise ValueError("plus/minus observables must be finite and matching")
    if not delta_plus > delta_minus:
        raise ValueError("delta_plus must exceed delta_minus")
    return (op - om) / (delta_plus - delta_minus)


def combine_covariance_components(
    components: dict[str, np.ndarray],
    *,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, CovarianceComponentReport]:
    """Sum covariance components after strict matrix and positive-semidefinite checks."""
    if not components:
        raise ValueError("at least one covariance component is required")
    mats: dict[str, np.ndarray] = {}
    dim: int | None = None
    warnings: list[str] = []
    for name, value in components.items():
        c = np.asarray(value, float)
        if c.ndim != 2 or c.shape[0] != c.shape[1] or np.any(~np.isfinite(c)):
            raise ValueError(f"component {name!r} must be a finite square matrix")
        if dim is None:
            dim = c.shape[0]
        if c.shape != (dim, dim) or not np.allclose(c, c.T, atol=1e-10):
            raise ValueError(f"component {name!r} has incompatible shape or is non-symmetric")
        eig = np.linalg.eigvalsh(c)
        if eig.min() < -tolerance * max(1.0, abs(float(eig.max()))):
            raise ValueError(f"component {name!r} is not positive semidefinite")
        if np.linalg.matrix_rank(c, tol=tolerance) < dim:
            warnings.append(
                f"component {name!r} is rank-deficient; this is expected for rank-one SSC but should be documented"
            )
        mats[name] = c
    total = np.sum(np.stack(list(mats.values())), axis=0)
    total = 0.5 * (total + total.T)
    eig = np.linalg.eigvalsh(total)
    psd = bool(eig.min() >= -tolerance * max(1.0, abs(float(eig.max()))))
    cond = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
    report = CovarianceComponentReport(
        tuple(mats),
        {n: float(np.trace(c)) for n, c in mats.items()},
        {n: float(np.mean(np.diag(c))) for n, c in mats.items()},
        float(np.trace(total)),
        float(eig.min()),
        cond,
        psd,
        tuple(warnings),
    )
    return total, report


def jackknife_covariance(replicate_vectors: np.ndarray) -> np.ndarray:
    """Estimate delete-one jackknife covariance from (n_replicates,n_data) vectors.

    Replicate vectors must correspond to a consistent delete-one partition estimator. This formula
    must not be applied to arbitrary independent mocks or to bootstrap resamples without changing the
    normalization. Jackknife covariance does not remove survey-window or nonlinear covariance bias.
    """
    x = np.asarray(replicate_vectors, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or np.any(~np.isfinite(x)):
        raise ValueError("replicate_vectors must be finite shape (n_replicates>=2,n_data>=1)")
    n = x.shape[0]
    mean = np.mean(x, axis=0)
    d = x - mean
    return (n - 1) / n * (d.T @ d)


def covariance_correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """Convert positive-diagonal covariance to a correlation matrix."""
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or np.any(~np.isfinite(c)):
        raise ValueError("covariance must be finite square")
    if not np.allclose(c, c.T, atol=1e-10) or np.any(np.diag(c) <= 0):
        raise ValueError("covariance must be symmetric with positive diagonal")
    s = np.sqrt(np.diag(c))
    result = c / np.outer(s, s)
    return np.clip(0.5 * (result + result.T), -1.0, 1.0)


__all__ = [
    "CovarianceComponentReport",
    "spherical_tophat_fourier_window",
    "background_density_variance",
    "super_sample_covariance",
    "single_mode_ssc",
    "finite_difference_response",
    "combine_covariance_components",
    "jackknife_covariance",
    "covariance_correlation_matrix",
]
