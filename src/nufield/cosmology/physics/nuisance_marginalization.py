"""Analytical treatment of linear nuisance templates in cosmological data vectors.

A broad class of survey nuisance models can be written y = μ(θ) + Aβ + ε, where ε is Gaussian with
covariance C. With a Gaussian prior β~N(β0,Λ), integrating β out gives an exact Gaussian model with
mean μ+Aβ0 and covariance C+AΛAᵀ. With an unbounded/flat prior, the nuisance parameters can instead
be profiled by generalized least squares; a normalized marginalized likelihood is not defined for an
improper flat prior without an explicit measure convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nufield.cosmology.physics.joint_likelihood import JointGaussianLikelihood, LikelihoodResult


@dataclass(frozen=True, slots=True)
class NuisanceFit:
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    residual_after_fit: np.ndarray
    chi_squared: float
    log_likelihood_profiled: float
    rank: int
    condition_number: float
    names: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        b = np.asarray(self.coefficients, float)
        c = np.asarray(self.coefficient_covariance, float)
        r = np.asarray(self.residual_after_fit, float)
        if b.ndim != 1 or c.shape != (b.size, b.size) or r.ndim != 1:
            raise ValueError("nuisance fit output shapes are invalid")
        object.__setattr__(self, "coefficients", b.copy())
        object.__setattr__(self, "coefficient_covariance", c.copy())
        object.__setattr__(self, "residual_after_fit", r.copy())


@dataclass(frozen=True, slots=True)
class MarginalizedGaussian:
    mean: np.ndarray
    covariance: np.ndarray
    nuisance_mean: np.ndarray
    nuisance_covariance: np.ndarray
    template_matrix: np.ndarray
    names: tuple[str, ...]
    warnings: tuple[str, ...]


def _validate_templates(
    template_matrix: np.ndarray,
    n_data: int,
    names: tuple[str, ...] = (),
) -> np.ndarray:
    a = np.asarray(template_matrix, float)
    if a.ndim != 2 or a.shape[0] != n_data or a.shape[1] < 1 or np.any(~np.isfinite(a)):
        raise ValueError("template_matrix must be finite with shape (n_data, n_nuisance>=1)")
    if names and len(names) != a.shape[1]:
        raise ValueError("nuisance names must match template columns")
    if np.linalg.matrix_rank(a) < a.shape[1]:
        raise ValueError(
            "nuisance templates are linearly dependent; remove or reparameterize redundant columns"
        )
    return a


def marginalize_gaussian_linear_nuisance(
    theory_mean: np.ndarray,
    data_covariance: np.ndarray,
    template_matrix: np.ndarray,
    nuisance_prior_mean: np.ndarray | None = None,
    nuisance_prior_covariance: np.ndarray | None = None,
    *,
    names: tuple[str, ...] = (),
) -> MarginalizedGaussian:
    r"""Integrate Gaussian-prior linear nuisance coefficients exactly.

    Parameters
    ----------
    theory_mean : (n_data,)
        Predicted cosmological data vector before nuisance contributions.
    data_covariance : (n_data,n_data)
        Statistical/systematic covariance excluding the modeled nuisance uncertainty.
    template_matrix : (n_data,n_nuisance)
        Columns are explicit nuisance templates, e.g. additive broadband basis vectors.
    nuisance_prior_mean, nuisance_prior_covariance
        Mean and covariance of the nuisance coefficients. Omitting both gives the zero-mean proper
        prior with unit covariance; pass physically justified prior widths for scientific work.

    Returns the exact marginal Gaussian N(μ+Aβ0, C+AΛAᵀ). The caller must avoid adding the same
    nuisance uncertainty a second time through data_covariance.
    """
    mu, cov = np.asarray(theory_mean, float), np.asarray(data_covariance, float)
    if mu.ndim != 1 or np.any(~np.isfinite(mu)):
        raise ValueError("theory_mean must be finite and one-dimensional")
    if (
        cov.shape != (mu.size, mu.size)
        or np.any(~np.isfinite(cov))
        or not np.allclose(cov, cov.T, atol=1e-10)
    ):
        raise ValueError("data_covariance must be finite, symmetric, and dimension-matched")
    np.linalg.cholesky(cov)
    a = _validate_templates(template_matrix, mu.size, names)
    n_beta = a.shape[1]
    beta0 = (
        np.zeros(n_beta) if nuisance_prior_mean is None else np.asarray(nuisance_prior_mean, float)
    )
    lam = (
        np.eye(n_beta)
        if nuisance_prior_covariance is None
        else np.asarray(nuisance_prior_covariance, float)
    )
    if beta0.shape != (n_beta,) or np.any(~np.isfinite(beta0)):
        raise ValueError("nuisance_prior_mean must be finite and match template count")
    if (
        lam.shape != (n_beta, n_beta)
        or np.any(~np.isfinite(lam))
        or not np.allclose(lam, lam.T, atol=1e-10)
    ):
        raise ValueError(
            "nuisance_prior_covariance must be finite symmetric and match template count"
        )
    np.linalg.cholesky(lam)
    mean = mu + a @ beta0
    marg_cov = cov + a @ lam @ a.T
    marg_cov = 0.5 * (marg_cov + marg_cov.T)
    np.linalg.cholesky(marg_cov)
    return MarginalizedGaussian(
        mean,
        marg_cov,
        beta0.copy(),
        lam.copy(),
        a.copy(),
        tuple(names),
        (
            "Analytical integration assumes Gaussian data residuals and a proper Gaussian nuisance prior.",
            "Ensure nuisance covariance is not already included in the supplied data covariance.",
        ),
    )


def profile_linear_nuisance(
    observed: np.ndarray,
    theory_mean: np.ndarray,
    data_covariance: np.ndarray,
    template_matrix: np.ndarray,
    *,
    names: tuple[str, ...] = (),
    rcond: float = 1e-10,
) -> NuisanceFit:
    r"""Find generalized-least-squares nuisance coefficients and the profiled likelihood.

    The fitted coefficients solve (AᵀC⁻¹A)β=AᵀC⁻¹(y-μ). Their covariance conditional on the model is
    (AᵀC⁻¹A)⁻¹. The reported likelihood is the ordinary normalized Gaussian likelihood evaluated at
    the best fit. It is a profile likelihood, not a marginalized posterior over β.
    """
    y, mu, c = (
        np.asarray(observed, float),
        np.asarray(theory_mean, float),
        np.asarray(data_covariance, float),
    )
    if y.ndim != 1 or mu.shape != y.shape or np.any(~np.isfinite(y + mu)):
        raise ValueError("observed and theory_mean must be matching finite vectors")
    if (
        c.shape != (y.size, y.size)
        or np.any(~np.isfinite(c))
        or not np.allclose(c, c.T, atol=1e-10)
    ):
        raise ValueError("covariance must be finite symmetric and dimension-matched")
    a = _validate_templates(template_matrix, y.size, names)
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("data covariance must be positive definite") from exc
    # Compute C^-1 A and C^-1 residual through triangular solves, avoiding explicit C^-1.
    cinv_a = np.linalg.solve(chol.T, np.linalg.solve(chol, a))
    residual = y - mu
    cinv_r = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
    fisher = a.T @ cinv_a
    rank = int(np.linalg.matrix_rank(fisher, tol=rcond * np.linalg.norm(fisher, ord=2)))
    if rank < a.shape[1]:
        raise ValueError("nuisance design is not identifiable under the data covariance")
    cond = float(np.linalg.cond(fisher))
    cov_beta = np.linalg.inv(fisher)
    beta = cov_beta @ (a.T @ cinv_r)
    resid_fit = residual - a @ beta
    cinv_resid_fit = np.linalg.solve(chol.T, np.linalg.solve(chol, resid_fit))
    chi2 = float(resid_fit @ cinv_resid_fit)
    logdet = 2 * float(np.sum(np.log(np.diag(chol))))
    ll = -0.5 * (chi2 + y.size * math.log(2 * math.pi) + logdet)
    warnings = []
    if cond > 1e10:
        warnings.append(
            "nuisance normal matrix is ill-conditioned; coefficient estimates may be unstable"
        )
    warnings.append("Profile likelihood is not equivalent to marginalizing nuisance coefficients.")
    return NuisanceFit(
        beta, cov_beta, resid_fit, chi2, ll, rank, cond, tuple(names), tuple(warnings)
    )


def gaussian_marginalized_likelihood(
    observed: np.ndarray,
    theory_mean: np.ndarray,
    data_covariance: np.ndarray,
    template_matrix: np.ndarray,
    nuisance_prior_mean: np.ndarray,
    nuisance_prior_covariance: np.ndarray,
    *,
    names: tuple[str, ...] = (),
) -> tuple[LikelihoodResult, MarginalizedGaussian]:
    """Convenience wrapper to integrate linear nuisance terms and evaluate the resulting likelihood."""
    marginalized = marginalize_gaussian_linear_nuisance(
        theory_mean,
        data_covariance,
        template_matrix,
        nuisance_prior_mean,
        nuisance_prior_covariance,
        names=names,
    )
    likelihood = JointGaussianLikelihood(
        observed, marginalized.covariance, covariance_label="linear-nuisance-marginalized"
    )
    return likelihood.evaluate(marginalized.mean), marginalized


def polynomial_template_matrix(
    x: np.ndarray,
    order: int,
    *,
    basis: str = "centered_power",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Construct additive nuisance columns from a dimensionless coordinate vector.

    ``centered_power`` uses x^0,...,x^order. ``legendre`` uses Legendre polynomials on [-1,1]. The
    x scaling is the caller's responsibility: for cosmological k bins use log(k/k_pivot) or a
    normalized coordinate instead of raw tiny/large physical units.
    """
    xx = np.asarray(x, float)
    if xx.ndim != 1 or xx.size < order + 2 or np.any(~np.isfinite(xx)) or order < 0:
        raise ValueError("x must be finite 1-D and have more rows than polynomial order")
    if basis == "centered_power":
        a = np.column_stack([xx**j for j in range(order + 1)])
    elif basis == "legendre":
        if np.min(xx) < -1 - 1e-8 or np.max(xx) > 1 + 1e-8:
            raise ValueError("Legendre basis requires x in [-1,1]")
        a = np.polynomial.legendre.legvander(xx, order)
    else:
        raise ValueError("basis must be 'centered_power' or 'legendre'")
    if weights is not None:
        w = np.asarray(weights, float)
        if w.shape != xx.shape or np.any(w <= 0) or np.any(~np.isfinite(w)):
            raise ValueError("weights must be finite and positive, matching x")
        a = a / np.sqrt(w)[:, None]
    if np.linalg.matrix_rank(a) < a.shape[1]:
        raise ValueError("template matrix is rank-deficient; rescale x or reduce order")
    return a


def fisher_marginalize_parameters(
    fisher: np.ndarray, interest_indices: Sequence[int]
) -> np.ndarray:
    """Return marginalized Fisher matrix for selected parameters via Schur complement.

    This is equivalent to removing nuisance-parameter degrees of freedom in a local Gaussian
    approximation. It is not reliable for strongly non-Gaussian or multi-modal posteriors.
    """
    f = np.asarray(fisher, float)
    keep = np.asarray(tuple(int(i) for i in interest_indices), dtype=int)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or np.any(~np.isfinite(f)):
        raise ValueError("fisher matrix must be finite square")
    if (
        keep.size == 0
        or len(set(keep.tolist())) != keep.size
        or np.any(keep < 0)
        or np.any(keep >= f.shape[0])
    ):
        raise ValueError("interest_indices must contain valid unique indices")
    if not np.allclose(f, f.T, atol=1e-10):
        raise ValueError("fisher matrix must be symmetric")
    nuis = np.array([i for i in range(f.shape[0]) if i not in set(keep.tolist())], dtype=int)
    fii = f[np.ix_(keep, keep)]
    if nuis.size == 0:
        return fii.copy()
    fin = f[np.ix_(keep, nuis)]
    fnn = f[np.ix_(nuis, nuis)]
    try:
        result = fii - fin @ np.linalg.solve(fnn, fin.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "nuisance Fisher submatrix is singular; add justified priors or reparameterize"
        ) from exc
    result = 0.5 * (result + result.T)
    if np.linalg.eigvalsh(result).min() <= 0:
        raise ValueError("marginalized Fisher matrix is not positive definite")
    return result


__all__ = [
    "NuisanceFit",
    "MarginalizedGaussian",
    "marginalize_gaussian_linear_nuisance",
    "profile_linear_nuisance",
    "gaussian_marginalized_likelihood",
    "polynomial_template_matrix",
    "fisher_marginalize_parameters",
]
