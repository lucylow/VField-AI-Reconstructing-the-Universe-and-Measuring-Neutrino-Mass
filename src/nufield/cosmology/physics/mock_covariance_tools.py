"""Covariance estimation, shrinkage and mock-stability diagnostics for cosmological data vectors.

Covariance uncertainty can dominate high-dimensional field-level analyses. The utilities here measure
rank and conditioning, compare mock subsets, and offer one transparent Ledoit--Wolf-style shrinkage
estimate toward an isotropic target. They do not repair correlated or non-representative mocks and must
not be confused with a finite-mock likelihood such as the Sellentin--Heavens t likelihood.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CovarianceStabilityReport:
    n_mocks: int
    n_data: int
    matrix_rank: int
    condition_number: float
    min_eigenvalue: float
    max_eigenvalue: float
    effective_rank: float
    relative_diagonal_error: float | None
    positive_definite: bool
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def covariance_diagnostics(
    covariance: np.ndarray,
    *,
    n_mocks: int | None = None,
    reference_covariance: np.ndarray | None = None,
    eigenvalue_floor: float = 1e-12,
) -> CovarianceStabilityReport:
    """Summarize eigenmodes and optional differences to a reference covariance."""
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.size == 0 or np.any(~np.isfinite(c)):
        raise ValueError("covariance must be a finite non-empty square matrix")
    if not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive")
    eig = np.linalg.eigvalsh(0.5 * (c + c.T))
    positive = bool(eig.min() > 0)
    rank = int(np.linalg.matrix_rank(c, tol=eigenvalue_floor * max(1.0, abs(float(eig.max())))))
    cond = float(eig.max() / eig.min()) if positive else float("inf")
    pos_eig = np.clip(eig, 0, None)
    effrank = float(pos_eig.sum() ** 2 / np.sum(pos_eig**2)) if np.sum(pos_eig**2) > 0 else 0.0
    diag_error = None
    warnings = []
    if not positive:
        warnings.append(
            "covariance is not positive definite; a standard Gaussian likelihood cannot use it directly"
        )
    if rank < c.shape[0]:
        warnings.append(
            "covariance is rank-deficient; often n_mocks<=n_data or data bins are redundant"
        )
    if n_mocks is not None:
        if n_mocks < 2:
            raise ValueError("n_mocks must be >=2")
        if n_mocks <= c.shape[0]:
            warnings.append("number of mocks does not exceed data-vector dimension")
        if n_mocks < 10 * c.shape[0]:
            warnings.append(
                "mock count is small relative to data dimension; inverse covariance may be noisy"
            )
    if reference_covariance is not None:
        ref = np.asarray(reference_covariance, float)
        if ref.shape != c.shape or np.any(~np.isfinite(ref)) or np.any(np.diag(ref) <= 0):
            raise ValueError(
                "reference covariance must be finite, shape matched, with positive diagonal"
            )
        denom = np.diag(ref)
        diag_error = float(np.sqrt(np.mean(((np.diag(c) - denom) / denom) ** 2)))
    return CovarianceStabilityReport(
        int(n_mocks or 0),
        c.shape[0],
        rank,
        cond,
        float(eig.min()),
        float(eig.max()),
        effrank,
        diag_error,
        positive,
        tuple(warnings),
    )


def empirical_covariance(mock_vectors: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    """Return sample covariance from independent mock data vectors of shape (n_mock,n_data)."""
    x = np.asarray(mock_vectors, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or np.any(~np.isfinite(x)):
        raise ValueError("mock_vectors must be finite (n_mock>=2,n_data>=1)")
    if not 0 <= ddof < x.shape[0]:
        raise ValueError("ddof must be in [0,n_mocks)")
    c = np.cov(x, rowvar=False, ddof=ddof)
    return np.atleast_2d(np.asarray(c, float))


def ledoit_wolf_identity_shrinkage(mock_vectors: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    r"""Estimate linear covariance shrinkage toward μI from centered mock vectors.

    Uses the simple large-sample Ledoit--Wolf-style estimator for independent row samples. Covariance
    uses the 1/N normalization here to match the intensity estimate. The resulting shrinkage intensity
    is clipped to [0,1]. This is an automatic numerical estimator, not guaranteed optimal for correlated
    mocks, non-Gaussian likelihoods, or a target structure known to be non-isotropic.
    """
    x = np.asarray(mock_vectors, float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1 or np.any(~np.isfinite(x)):
        raise ValueError("mock_vectors must be finite (n_mock>=3,n_data>=1)")
    x = x - np.mean(x, axis=0, keepdims=True)
    n, p = x.shape
    s = x.T @ x / n
    mu = float(np.trace(s) / p)
    target = np.eye(p) * mu
    delta2 = float(np.sum((s - target) ** 2))
    if delta2 <= 0:
        alpha = 1.0
    else:
        # Estimate E||x_i x_i^T - S||_F² / N for sample covariance fluctuations.
        phi = 0.0
        for row in x:
            outer = np.outer(row, row)
            phi += float(np.sum((outer - s) ** 2))
        beta = phi / (n * n)
        alpha = float(np.clip(beta / delta2, 0, 1))
    shrunk = (1 - alpha) * s + alpha * target
    shrunk = 0.5 * (shrunk + shrunk.T)
    return shrunk, {
        "shrinkage_intensity": alpha,
        "target_mean_variance": mu,
        "raw_trace": float(np.trace(s)),
        "shrunk_trace": float(np.trace(shrunk)),
        "n_mocks": float(n),
        "n_data": float(p),
        "warning": "Assumes independent mocks and an isotropic shrinkage target.",
    }


def covariance_subset_stability(
    mock_vectors: np.ndarray, *, n_subsets: int = 4, seed: int = 0
) -> dict[str, object]:
    """Compare covariance diagonals and eigenvalues across disjoint mock subsets.

    This is a diagnostic for sensitivity to the available mock ensemble, not an unbiased covariance
    estimator. Every subset must contain at least two mocks; use more mocks if the split is unstable.
    """
    x = np.asarray(mock_vectors, float)
    if x.ndim != 2 or x.shape[0] < 2 * n_subsets or x.shape[1] < 1 or np.any(~np.isfinite(x)):
        raise ValueError("mock_vectors must have >=2*n_subsets finite rows")
    if n_subsets < 2:
        raise ValueError("n_subsets must be >=2")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])
    pieces = np.array_split(idx, n_subsets)
    covariances = [empirical_covariance(x[piece]) for piece in pieces]
    diag = np.stack([np.diag(c) for c in covariances])
    eigs = np.stack([np.linalg.eigvalsh(c) for c in covariances])
    full = empirical_covariance(x)
    full_diag = np.diag(full)
    rel_diag = (diag - full_diag[None, :]) / np.maximum(full_diag[None, :], 1e-30)
    return {
        "n_subsets": n_subsets,
        "subset_sizes": [len(p) for p in pieces],
        "relative_diagonal_rms_by_subset": np.sqrt(np.mean(rel_diag**2, axis=1)).tolist(),
        "eigenvalue_min_by_subset": eigs[:, 0].tolist(),
        "eigenvalue_max_by_subset": eigs[:, -1].tolist(),
        "median_absolute_relative_diagonal_spread": float(np.median(np.std(rel_diag, axis=0))),
        "seed": int(seed),
        "warning": "Subset stability does not demonstrate mock representativeness or account for common simulation systematics.",
    }


def whiten_residuals(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return z=L⁻¹r for C=LLᵀ; ideal Gaussian simulations should have approximately unit covariance."""
    r, c = np.asarray(residual, float), np.asarray(covariance, float)
    if (
        r.ndim != 1
        or c.shape != (r.size, r.size)
        or np.any(~np.isfinite(r))
        or np.any(~np.isfinite(c))
    ):
        raise ValueError("residual and covariance shapes/values invalid")
    try:
        l = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return np.linalg.solve(l, r)


def whitened_ensemble_diagnostics(
    mock_vectors: np.ndarray, reference_mean: np.ndarray, covariance: np.ndarray
) -> dict[str, object]:
    """Inspect the sample mean and covariance after Cholesky whitening by a reference covariance."""
    x = np.asarray(mock_vectors, float)
    mu = np.asarray(reference_mean, float)
    c = np.asarray(covariance, float)
    if x.ndim != 2 or mu.shape != (x.shape[1],) or c.shape != (x.shape[1], x.shape[1]):
        raise ValueError("mock vectors, mean and covariance dimensions are inconsistent")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(mu)):
        raise ValueError("mock vectors and mean must be finite")
    l = np.linalg.cholesky(c)
    z = np.linalg.solve(l, (x - mu).T).T
    sample_cov = np.cov(z, rowvar=False, ddof=1)
    sample_cov = np.atleast_2d(sample_cov)
    return {
        "n_mocks": int(x.shape[0]),
        "n_data": int(x.shape[1]),
        "mean_whitened_rms": float(np.sqrt(np.mean(np.mean(z, axis=0) ** 2))),
        "mean_whitened_max_abs": float(np.max(np.abs(np.mean(z, axis=0)))),
        "whitened_covariance_diagonal_mean": float(np.mean(np.diag(sample_cov))),
        "whitened_covariance_diagonal_rms_error_from_one": float(
            np.sqrt(np.mean((np.diag(sample_cov) - 1) ** 2))
        ),
        "whitened_offdiagonal_rms": float(
            np.sqrt(np.mean((sample_cov - np.diag(np.diag(sample_cov))) ** 2))
        ),
        "interpretation": "Descriptive finite-mock diagnostic; errors shrink slowly with data-vector dimension.",
    }


__all__ = [
    "CovarianceStabilityReport",
    "covariance_diagnostics",
    "empirical_covariance",
    "ledoit_wolf_identity_shrinkage",
    "covariance_subset_stability",
    "whiten_residuals",
    "whitened_ensemble_diagnostics",
]
