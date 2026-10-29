"""Posterior-predictive and coverage diagnostics for cosmological inference experiments.

These methods evaluate supplied posterior samples and simulator outputs. They cannot make an inference
calibrated merely by calculating a diagnostic: validation simulations must be independent of training,
parameter distributions must be declared, and all prior/likelihood transformations must be consistent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PredictiveResidualReport:
    observed: np.ndarray
    predictive_mean: np.ndarray
    predictive_std: np.ndarray
    lower_quantile: np.ndarray
    upper_quantile: np.ndarray
    standardized_residual: np.ndarray
    within_interval: np.ndarray
    coverage_fraction: float
    level: float
    n_predictive_draws: int
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed.tolist(),
            "predictive_mean": self.predictive_mean.tolist(),
            "predictive_std": self.predictive_std.tolist(),
            "lower_quantile": self.lower_quantile.tolist(),
            "upper_quantile": self.upper_quantile.tolist(),
            "standardized_residual": self.standardized_residual.tolist(),
            "within_interval": self.within_interval.tolist(),
            "coverage_fraction": self.coverage_fraction,
            "interval_level": self.level,
            "n_predictive_draws": self.n_predictive_draws,
            "caveats": list(self.caveats),
        }


def posterior_predictive_residuals(
    observed: np.ndarray,
    predictive_draws: np.ndarray,
    *,
    interval_level: float = 0.68,
) -> PredictiveResidualReport:
    """Summarize per-bin prediction intervals for draws shaped (n_draw, n_observable)."""
    y, draws = np.asarray(observed, float), np.asarray(predictive_draws, float)
    if y.ndim != 1 or draws.ndim != 2 or draws.shape[1] != y.size or draws.shape[0] < 2:
        raise ValueError("observed must be (n_data,) and predictive_draws (n_draw>=2,n_data)")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(draws)):
        raise ValueError("observed and predictive draws must be finite")
    if not 0 < interval_level < 1:
        raise ValueError("interval_level must be in (0,1)")
    tail = (1.0 - interval_level) / 2
    mean = np.mean(draws, axis=0)
    std = np.std(draws, axis=0, ddof=1)
    lo, hi = np.quantile(draws, [tail, 1 - tail], axis=0)
    z = np.divide(y - mean, std, out=np.zeros_like(y), where=std > 0)
    covered = (y >= lo) & (y <= hi)
    return PredictiveResidualReport(
        y.copy(),
        mean,
        std,
        lo,
        hi,
        z,
        covered,
        float(np.mean(covered)),
        interval_level,
        draws.shape[0],
        (
            "Bin-wise intervals ignore correlations unless predictive draws preserve them.",
            "Coverage from a single observed vector is descriptive, not a calibration estimate.",
            "Use independent simulated universes to estimate repeated-sampling coverage.",
        ),
    )


@dataclass(frozen=True, slots=True)
class PosteriorCoverageReport:
    parameter_names: tuple[str, ...]
    levels: tuple[float, ...]
    coverage: np.ndarray
    expected: np.ndarray
    absolute_error: np.ndarray
    n_simulations: int
    n_posterior_draws: int
    interval: str
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_names": list(self.parameter_names),
            "levels": list(self.levels),
            "coverage": self.coverage.tolist(),
            "expected": self.expected.tolist(),
            "absolute_error": self.absolute_error.tolist(),
            "n_simulations": self.n_simulations,
            "n_posterior_draws": self.n_posterior_draws,
            "interval": self.interval,
            "caveats": list(self.caveats),
        }


def posterior_interval_coverage(
    true_parameters: np.ndarray,
    posterior_draws: np.ndarray,
    *,
    parameter_names: Sequence[str] | None = None,
    levels: tuple[float, ...] = (0.50, 0.68, 0.90, 0.95),
) -> PosteriorCoverageReport:
    """Estimate central equal-tailed interval coverage over independent simulations.

    ``true_parameters`` has shape (n_sim,n_parameter); ``posterior_draws`` has shape
    (n_sim,n_draw,n_parameter). This uses marginal intervals and therefore diagnoses marginal, not
    joint multivariate credible-region coverage.
    """
    truth, draws = np.asarray(true_parameters, float), np.asarray(posterior_draws, float)
    if (
        truth.ndim != 2
        or draws.ndim != 3
        or draws.shape[0] != truth.shape[0]
        or draws.shape[2] != truth.shape[1]
    ):
        raise ValueError("expected truth (n_sim,n_param) and draws (n_sim,n_draw,n_param)")
    if truth.shape[0] < 1 or draws.shape[1] < 2 or truth.shape[1] < 1:
        raise ValueError(
            "at least one simulation, two posterior draws, and one parameter are required"
        )
    if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(draws)):
        raise ValueError("truth and posterior samples must be finite")
    lv = tuple(float(x) for x in levels)
    if not lv or any(not 0 < x < 1 for x in lv) or tuple(sorted(lv)) != lv:
        raise ValueError("levels must be sorted and lie strictly between zero and one")
    names = tuple(parameter_names or [f"theta_{i}" for i in range(truth.shape[1])])
    if len(names) != truth.shape[1] or len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique and match parameter dimension")
    coverage = np.empty((len(lv), truth.shape[1]), float)
    for i, level in enumerate(lv):
        alpha = (1 - level) / 2
        lo, hi = np.quantile(draws, [alpha, 1 - alpha], axis=1)
        # np.quantile output shape is (2,n_sim,n_param); move the interval axis for clarity.
        inside = (truth >= lo[0]) & (truth <= hi[1])
        coverage[i] = np.mean(inside, axis=0)
    expected = np.repeat(np.asarray(lv)[:, None], truth.shape[1], axis=1)
    return PosteriorCoverageReport(
        names,
        lv,
        coverage,
        expected,
        np.abs(coverage - expected),
        truth.shape[0],
        draws.shape[1],
        "central equal-tailed marginal intervals",
        (
            "Calibration requires simulations drawn from the same prior/model as the algorithm's target.",
            "Use independent parent simulations; repeated crops from the same box are not independent universes.",
            "Finite simulation counts create binomial uncertainty in estimated coverage.",
        ),
    )


def empirical_rank(
    true_value: float, posterior_samples: np.ndarray, *, tie_tolerance: float = 0.0
) -> float:
    """Return the normalized empirical rank of one ground-truth parameter within posterior draws."""
    samples = np.asarray(posterior_samples, float)
    if (
        samples.ndim != 1
        or samples.size < 2
        or np.any(~np.isfinite(samples))
        or not np.isfinite(true_value)
    ):
        raise ValueError("posterior_samples must be finite 1-D samples and truth finite")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    below = np.count_nonzero(samples < true_value - tie_tolerance)
    tied = np.count_nonzero(np.abs(samples - true_value) <= tie_tolerance)
    rank = below + 0.5 * tied
    return float(rank / samples.size)


def gaussian_discrepancy_p_value(
    observed: np.ndarray,
    replicated: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float | int | str]:
    """Compare a Mahalanobis discrepancy for observed data against predictive replications.

    ``replicated`` shape is (n_rep,n_data). The covariance is user supplied and may be fixed or an
    explicitly justified estimate. The empirical upper-tail p-value uses a +1 correction to avoid a
    reported exact zero; it is a diagnostic, not a Bayesian model-comparison Bayes factor.
    """
    y, reps, c = (
        np.asarray(observed, float),
        np.asarray(replicated, float),
        np.asarray(covariance, float),
    )
    if y.ndim != 1 or reps.ndim != 2 or reps.shape[1] != y.size or reps.shape[0] < 2:
        raise ValueError("observed (n_data,) and replicated (n_rep>=2,n_data) required")
    if (
        c.shape != (y.size, y.size)
        or np.any(~np.isfinite(y))
        or np.any(~np.isfinite(reps))
        or np.any(~np.isfinite(c))
    ):
        raise ValueError("invalid shape or non-finite data/covariance")
    if not np.allclose(c, c.T, atol=1e-10, rtol=0):
        raise ValueError("covariance must be symmetric")
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc

    def statistic(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, float)
        solved = np.linalg.solve(c, arr.T).T
        return np.sum(arr * solved, axis=-1)

    t_obs = float(statistic(y))
    t_rep = statistic(reps)
    exceed = int(np.count_nonzero(t_rep >= t_obs))
    p = (exceed + 1) / (len(t_rep) + 1)
    return {
        "observed_mahalanobis_chi2": t_obs,
        "replicate_median": float(np.median(t_rep)),
        "replicate_q95": float(np.quantile(t_rep, 0.95)),
        "upper_tail_p_value": float(p),
        "n_replicates": len(t_rep),
        "n_exceedances": exceed,
        "note": "Empirical predictive-tail diagnostic; interpretation depends on the replication scheme.",
    }


__all__ = [
    "PredictiveResidualReport",
    "posterior_predictive_residuals",
    "PosteriorCoverageReport",
    "posterior_interval_coverage",
    "empirical_rank",
    "gaussian_discrepancy_p_value",
]
