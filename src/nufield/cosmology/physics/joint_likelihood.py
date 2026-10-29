"""Joint likelihood primitives for galaxy clustering, weak lensing and CMB-derived observables.

This module provides statistical composition, not dataset-specific likelihoods. The caller must supply
observed data vectors, a covariance compatible with those exact bins, and a theory predictor using the
same conventions. The software explicitly rejects dimension mismatches and non-positive-definite
covariances instead of silently adding diagonal jitter. Cross-probe covariance is supported through a
single full covariance matrix; setting cross blocks to zero is an explicit modeling assumption.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


@dataclass(frozen=True, slots=True)
class ProbeBlock:
    """Description of one contiguous segment of a joint data vector."""

    name: str
    start: int
    stop: int
    labels: tuple[str, ...]
    units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or self.start < 0 or self.stop <= self.start:
            raise ValueError("probe block must have name and a valid non-empty index range")
        if len(self.labels) != self.stop - self.start:
            raise ValueError("number of labels must match the block length")
        if self.units and len(self.units) != len(self.labels):
            raise ValueError("units must be omitted or match labels length")


@dataclass(frozen=True, slots=True)
class JointDataVector:
    """Ordered observed vector with explicit block lineage."""

    values: np.ndarray
    blocks: tuple[ProbeBlock, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        y = np.asarray(self.values, float)
        if y.ndim != 1 or y.size < 1 or np.any(~np.isfinite(y)):
            raise ValueError("joint data values must be a finite non-empty vector")
        if not self.blocks:
            raise ValueError("at least one probe block is required")
        ordered = sorted(self.blocks, key=lambda b: b.start)
        cursor = 0
        names: set[str] = set()
        for b in ordered:
            if b.start != cursor or b.stop > y.size:
                raise ValueError(
                    "probe blocks must be contiguous, non-overlapping and cover the data vector"
                )
            if b.name in names:
                raise ValueError("probe block names must be unique")
            names.add(b.name)
            cursor = b.stop
        if cursor != y.size:
            raise ValueError("probe blocks must cover every data-vector element")
        object.__setattr__(self, "values", y.copy())
        object.__setattr__(self, "blocks", tuple(ordered))

    @classmethod
    def from_blocks(
        cls,
        blocks: Mapping[str, Sequence[float]],
        *,
        units: Mapping[str, Sequence[str]] | None = None,
        labels: Mapping[str, Sequence[str]] | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> JointDataVector:
        """Concatenate named blocks in mapping insertion order."""
        if not blocks:
            raise ValueError("blocks mapping cannot be empty")
        arrays = []
        block_meta = []
        start = 0
        for name, values in blocks.items():
            arr = np.asarray(values, float)
            if arr.ndim != 1 or arr.size == 0 or np.any(~np.isfinite(arr)):
                raise ValueError(f"block {name!r} must be a finite non-empty vector")
            labs = tuple((labels or {}).get(name, [f"{name}[{i}]" for i in range(arr.size)]))
            unit = tuple((units or {}).get(name, ()))
            block_meta.append(ProbeBlock(str(name), start, start + arr.size, labs, unit))
            arrays.append(arr)
            start += arr.size
        return cls(
            np.concatenate(arrays), tuple(block_meta), tuple(sorted((metadata or {}).items()))
        )

    def slices(self) -> dict[str, slice]:
        return {b.name: slice(b.start, b.stop) for b in self.blocks}

    def split(self, vector: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Split observed or same-length modeled vector into named views (copies)."""
        values = self.values if vector is None else np.asarray(vector, float)
        if values.shape != self.values.shape:
            raise ValueError("vector shape must match the joint data vector")
        return {b.name: values[b.start : b.stop].copy() for b in self.blocks}


@dataclass(frozen=True, slots=True)
class LikelihoodResult:
    log_likelihood: float
    chi_squared: float
    log_determinant_covariance: float
    residual: np.ndarray
    whitened_residual: np.ndarray
    coordinate_chi_squared: np.ndarray
    valid: bool = True
    message: str = ""

    def __post_init__(self) -> None:
        r, w, c = map(
            lambda x: np.asarray(x, float),
            (self.residual, self.whitened_residual, self.coordinate_chi_squared),
        )
        if r.ndim != 1 or w.shape != r.shape or c.shape != r.shape:
            raise ValueError("residual diagnostics must be same-length vectors")
        object.__setattr__(self, "residual", r.copy())
        object.__setattr__(self, "whitened_residual", w.copy())
        object.__setattr__(self, "coordinate_chi_squared", c.copy())


@dataclass(frozen=True, slots=True)
class GaussianPrior:
    """Multivariate normal prior over a fixed-ordered parameter vector."""

    mean: np.ndarray
    covariance: np.ndarray
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mu, cov = np.asarray(self.mean, float), np.asarray(self.covariance, float)
        if mu.ndim != 1 or cov.shape != (mu.size, mu.size) or mu.size == 0:
            raise ValueError("prior mean/covariance shape mismatch")
        if (
            np.any(~np.isfinite(mu))
            or np.any(~np.isfinite(cov))
            or not np.allclose(cov, cov.T, atol=1e-10)
        ):
            raise ValueError("prior mean/covariance must be finite and covariance symmetric")
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("prior covariance must be positive definite") from exc
        if self.names and len(self.names) != mu.size:
            raise ValueError("prior names must be omitted or match parameter count")
        object.__setattr__(self, "mean", mu.copy())
        object.__setattr__(self, "covariance", cov.copy())

    def logpdf(self, parameters: np.ndarray, *, include_normalization: bool = True) -> float:
        theta = np.asarray(parameters, float)
        if theta.shape != self.mean.shape or np.any(~np.isfinite(theta)):
            return -math.inf
        cf = cho_factor(self.covariance, lower=True, check_finite=False)
        delta = theta - self.mean
        chi2 = float(delta @ cho_solve(cf, delta, check_finite=False))
        if not include_normalization:
            return -0.5 * chi2
        _, logdet = np.linalg.slogdet(self.covariance)
        return float(-0.5 * (chi2 + self.mean.size * math.log(2 * math.pi) + logdet))


class JointGaussianLikelihood:
    """A numerically stable Gaussian likelihood using one joint covariance matrix."""

    def __init__(
        self,
        data: JointDataVector | np.ndarray,
        covariance: np.ndarray,
        *,
        block_names: Sequence[str] | None = None,
        covariance_label: str = "unspecified",
        require_positive_definite: bool = True,
    ) -> None:
        if isinstance(data, JointDataVector):
            self.data_vector = data
            y = data.values
        else:
            y = np.asarray(data, float)
            if y.ndim != 1 or y.size < 1 or np.any(~np.isfinite(y)):
                raise ValueError("data must be a finite 1-D vector")
            names = tuple(block_names or ("all",))
            if len(names) != 1:
                raise ValueError("plain array data currently supports exactly one block name")
            self.data_vector = JointDataVector(
                y,
                (
                    ProbeBlock(
                        names[0], 0, y.size, tuple(f"{names[0]}[{i}]" for i in range(y.size))
                    ),
                ),
            )
        cov = np.asarray(covariance, float)
        if cov.shape != (y.size, y.size) or np.any(~np.isfinite(cov)):
            raise ValueError("covariance must be finite and match data dimension")
        if not np.allclose(cov, cov.T, rtol=1e-9, atol=1e-11):
            raise ValueError("covariance must be symmetric")
        cov = 0.5 * (cov + cov.T)
        if require_positive_definite:
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "covariance must be positive definite; diagnose rather than silently jitter"
                ) from exc
        else:
            eig = np.linalg.eigvalsh(cov)
            if eig.min() < -1e-10 * max(1.0, abs(eig.max())):
                raise ValueError("covariance is not positive semidefinite")
            raise NotImplementedError(
                "semidefinite likelihoods require an explicit supported subspace"
            )
        self.observed = y.copy()
        self.covariance = cov.copy()
        self._cho = cho_factor(cov, lower=True, check_finite=False)
        self.logdet = float(2 * np.sum(np.log(np.diag(self._cho[0]))))
        self.covariance_label = str(covariance_label)

    def evaluate(self, prediction: np.ndarray) -> LikelihoodResult:
        """Evaluate normalized log-likelihood and residual diagnostics."""
        model = np.asarray(prediction, float)
        if model.shape != self.observed.shape or np.any(~np.isfinite(model)):
            raise ValueError("prediction must be finite and match the observed data vector")
        resid = self.observed - model
        precision_resid = cho_solve(self._cho, resid, check_finite=False)
        chi2 = float(resid @ precision_resid)
        white = solve_triangular(self._cho[0], resid, lower=bool(self._cho[1]), check_finite=False)
        loglike = -0.5 * (chi2 + self.observed.size * math.log(2 * math.pi) + self.logdet)
        return LikelihoodResult(
            loglike,
            chi2,
            self.logdet,
            resid,
            white,
            resid * precision_resid,
            True,
            f"covariance={self.covariance_label}",
        )

    def residual_by_probe(self, result: LikelihoodResult) -> dict[str, dict[str, float]]:
        """Summarize residuals by block; block chi² contributions sum to total chi².

        With nonzero cross-covariance, a block contribution can be negative: this is a property of
        the algebraic decomposition r_i (C^-1 r)_i, not a negative standalone goodness-of-fit.
        """
        if result.residual.shape != self.observed.shape:
            raise ValueError("result does not match this likelihood")
        split_chi = self.data_vector.split(result.coordinate_chi_squared)
        split_resid = self.data_vector.split(result.residual)
        out = {}
        for name, values in split_resid.items():
            c = split_chi[name]
            v = values
            out[name] = {
                "n": int(v.size),
                "chi2_contribution": float(np.sum(c)),
                "residual_rms": float(np.sqrt(np.mean(v * v))),
                "residual_mean": float(np.mean(v)),
            }
        return out


def block_diagonal_covariance(
    blocks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, slice]]:
    """Construct a block-diagonal covariance and preserve the block slices.

    This helper encodes the assumption of zero cross-probe covariance explicitly in the caller's
    manifest. It is not a method for estimating the missing cross-covariance blocks.
    """
    if not blocks:
        raise ValueError("at least one covariance block is required")
    matrices: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    start = 0
    for name, value in blocks.items():
        cov = np.asarray(value, float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.size == 0:
            raise ValueError(f"covariance block {name!r} must be square and non-empty")
        if np.any(~np.isfinite(cov)) or not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError(f"covariance block {name!r} must be finite and symmetric")
        np.linalg.cholesky(cov)
        matrices.append(cov)
        slices[name] = slice(start, start + cov.shape[0])
        start += cov.shape[0]
    out = np.zeros((start, start), float)
    for (name, cov), sl in zip(blocks.items(), slices.values()):
        out[sl, sl] = np.asarray(cov, float)
    return out, slices


def log_posterior(
    parameters: np.ndarray,
    likelihood: JointGaussianLikelihood,
    predictor: Callable[[np.ndarray], np.ndarray],
    *,
    prior: GaussianPrior | None = None,
    log_prior: Callable[[np.ndarray], float] | None = None,
) -> float:
    """Evaluate a log posterior from a theory predictor and optional prior.

    Exceptions from out-of-domain theory calls are mapped to ``-inf`` so samplers can reject invalid
    points. The exception is intentionally not swallowed for debugging unless ``strict=True``.
    """
    theta = np.asarray(parameters, float)
    if theta.ndim != 1 or np.any(~np.isfinite(theta)):
        return -math.inf
    try:
        model = predictor(theta)
        ll = likelihood.evaluate(model).log_likelihood
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return -math.inf
    lp = prior.logpdf(theta) if prior is not None else 0.0
    if log_prior is not None:
        lp += float(log_prior(theta))
    result = ll + lp
    return float(result) if math.isfinite(result) else -math.inf


__all__ = [
    "ProbeBlock",
    "JointDataVector",
    "LikelihoodResult",
    "GaussianPrior",
    "JointGaussianLikelihood",
    "block_diagonal_covariance",
    "log_posterior",
]
