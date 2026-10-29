"""Exact Gaussian-process emulator for modest simulation budgets.

This emulator learns a smooth map from declared simulation parameters to observables. It is a useful
small-data baseline and active-learning uncertainty proxy, but the exact GP scales cubically with
training rows and cannot repair missing physics in the simulator. Its variance is conditional on the
kernel/noise specification and is not automatically calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

KernelKind = Literal["rbf", "matern52"]


def _as_matrix(value: Any, name: str) -> np.ndarray:
    x = np.asarray(value, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite [N,D]")
    return x


def pairwise_squared_distance(x: Any, y: Any) -> np.ndarray:
    """Squared Euclidean distance with roundoff clipping."""
    a, b = _as_matrix(x, "x"), _as_matrix(y, "y")
    if a.shape[1] != b.shape[1]:
        raise ValueError("x and y feature widths differ")
    d = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T
    return np.maximum(d, 0.0)


def kernel_matrix(
    x: Any,
    y: Any,
    *,
    length_scale: float = 1.0,
    amplitude: float = 1.0,
    kind: KernelKind = "matern52",
) -> np.ndarray:
    """Evaluate an isotropic RBF or Matérn-5/2 covariance kernel."""
    if length_scale <= 0 or amplitude <= 0 or not np.isfinite(length_scale + amplitude):
        raise ValueError("length_scale and amplitude must be finite and positive")
    d2 = pairwise_squared_distance(x, y)
    r = np.sqrt(d2) / length_scale
    if kind == "rbf":
        return amplitude**2 * np.exp(-0.5 * r**2)
    if kind == "matern52":
        s = np.sqrt(5.0) * r
        return amplitude**2 * (1 + s + s**2 / 3) * np.exp(-s)
    raise ValueError("kind must be rbf or matern52")


@dataclass(frozen=True)
class GPFitSummary:
    n_train: int
    input_dim: int
    output_dim: int
    length_scale: float
    amplitude: float
    noise_variance: float
    log_marginal_likelihood: float
    jitter_used: float
    kernel: str

    def to_dict(self):
        return {
            **self.__dict__,
            "warning": "Exact GP uses O(N^3) factorization and approximate kernel assumptions.",
        }


class GaussianProcessEmulator:
    """Multi-output independent exact GPs with shared input kernel and per-output scaling."""

    def __init__(
        self,
        *,
        kernel: KernelKind = "matern52",
        length_scale: float = 0.25,
        amplitude: float = 1.0,
        noise_variance: float = 1e-4,
        jitter: float = 1e-10,
        max_jitter_tries: int = 8,
    ):
        if kernel not in {"rbf", "matern52"}:
            raise ValueError("unsupported kernel")
        if min(length_scale, amplitude, noise_variance, jitter) <= 0 or max_jitter_tries < 1:
            raise ValueError("kernel scales/jitter must be positive")
        self.kernel, self.length_scale = kernel, float(length_scale)
        self.amplitude, self.noise_variance = float(amplitude), float(noise_variance)
        self.jitter, self.max_jitter_tries = float(jitter), int(max_jitter_tries)
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: np.ndarray | None = None
        self.y_scale_: np.ndarray | None = None
        self.x_train_: np.ndarray | None = None
        self.cholesky_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.fit_summary_: GPFitSummary | None = None

    def fit(self, parameters: Any, observables: Any) -> GaussianProcessEmulator:
        x, y = _as_matrix(parameters, "parameters"), _as_matrix(observables, "observables")
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("parameters and observables must align, with N>=3")
        self.x_mean_ = x.mean(axis=0)
        self.x_scale_ = x.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-10] = 1.0
        self.y_mean_ = y.mean(axis=0)
        self.y_scale_ = y.std(axis=0)
        self.y_scale_[self.y_scale_ < 1e-10] = 1.0
        xs = (x - self.x_mean_) / self.x_scale_
        ys = (y - self.y_mean_) / self.y_scale_
        k = kernel_matrix(
            xs, xs, length_scale=self.length_scale, amplitude=self.amplitude, kind=self.kernel
        )
        base = k + np.eye(len(xs)) * self.noise_variance
        jitter = self.jitter
        chol = None
        for _ in range(self.max_jitter_tries):
            try:
                chol = np.linalg.cholesky(base + np.eye(len(xs)) * jitter)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        if chol is None:
            raise np.linalg.LinAlgError(
                "GP kernel matrix remained non-positive-definite after jitter escalation"
            )
        alpha = cho_solve((chol, True), ys, check_finite=False)
        logdet = 2 * np.log(np.diag(chol)).sum()
        log_lml_each = (
            -0.5 * np.sum(ys * alpha, axis=0) - 0.5 * logdet - 0.5 * len(xs) * np.log(2 * np.pi)
        )
        self.x_train_ = xs
        self.cholesky_ = chol
        self.alpha_ = alpha
        self.fit_summary_ = GPFitSummary(
            len(x),
            x.shape[1],
            y.shape[1],
            self.length_scale,
            self.amplitude,
            self.noise_variance,
            float(log_lml_each.sum()),
            float(jitter),
            self.kernel,
        )
        return self

    def predict(
        self,
        parameters: Any,
        *,
        return_covariance: bool = False,
        include_observation_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if any(
            v is None
            for v in (
                self.x_mean_,
                self.x_scale_,
                self.y_mean_,
                self.y_scale_,
                self.x_train_,
                self.cholesky_,
                self.alpha_,
            )
        ):
            raise RuntimeError("fit must be called before predict")
        x = _as_matrix(parameters, "parameters")
        if x.shape[1] != len(self.x_mean_):
            raise ValueError("parameter width mismatch")
        xs = (x - self.x_mean_) / self.x_scale_
        cross = kernel_matrix(
            xs,
            self.x_train_,
            length_scale=self.length_scale,
            amplitude=self.amplitude,
            kind=self.kernel,
        )
        mean_scaled = cross @ self.alpha_
        v = solve_triangular(self.cholesky_, cross.T, lower=True, check_finite=False)
        prior_diag = np.full(len(xs), self.amplitude**2)
        variance_scaled = np.maximum(prior_diag - np.sum(v * v, axis=0), 0.0)
        if include_observation_noise:
            variance_scaled += self.noise_variance
        mean = mean_scaled * self.y_scale_ + self.y_mean_
        variance = variance_scaled[:, None] * self.y_scale_[None, :] ** 2
        if return_covariance:
            # Full predictive covariance for each output, all in original target units.
            cov_scaled = (
                kernel_matrix(
                    xs,
                    xs,
                    length_scale=self.length_scale,
                    amplitude=self.amplitude,
                    kind=self.kernel,
                )
                - v.T @ v
            )
            if include_observation_noise:
                cov_scaled += np.eye(len(xs)) * self.noise_variance
            covariance = np.stack([cov_scaled * s**2 for s in self.y_scale_], axis=0)
            return mean, covariance
        return mean, variance

    def fit_length_scale_grid(
        self,
        parameters: Any,
        observables: Any,
        *,
        length_scales: Any,
        noise_variances: Any | None = None,
    ) -> dict[str, Any]:
        """Select kernel hyperparameters by marginal likelihood on training simulations only.

        This simple grid uses the same data to fit and score hyperparameters, so the returned score is
        an optimization criterion, not held-out generalization performance. Report a separate grouped
        validation score and do not use the locked test set for this choice.
        """
        xs = _as_matrix(parameters, "parameters")
        ys = _as_matrix(observables, "observables")
        noises = np.asarray(
            [self.noise_variance] if noise_variances is None else noise_variances, float
        )
        scales = np.asarray(length_scales, float)
        if np.any(scales <= 0) or np.any(noises <= 0) or not len(scales) or not len(noises):
            raise ValueError("length scales and noise variances must be nonempty and positive")
        rows = []
        best = None
        for length in scales:
            for noise in noises:
                trial = GaussianProcessEmulator(
                    kernel=self.kernel,
                    length_scale=float(length),
                    amplitude=self.amplitude,
                    noise_variance=float(noise),
                    jitter=self.jitter,
                    max_jitter_tries=self.max_jitter_tries,
                ).fit(xs, ys)
                row = {
                    "length_scale": float(length),
                    "noise_variance": float(noise),
                    "log_marginal_likelihood": trial.fit_summary_.log_marginal_likelihood,
                }
                rows.append(row)
                if best is None or row["log_marginal_likelihood"] > best["log_marginal_likelihood"]:
                    best = row
        self.length_scale = best["length_scale"]
        self.noise_variance = best["noise_variance"]
        self.fit(xs, ys)
        return {
            "best": best,
            "grid": rows,
            "warning": "Use grouped validation to assess emulator generalization and calibrated predictive variance.",
        }

    def summary(self) -> dict[str, Any]:
        if self.fit_summary_ is None:
            raise RuntimeError("fit must be called first")
        return self.fit_summary_.to_dict()
