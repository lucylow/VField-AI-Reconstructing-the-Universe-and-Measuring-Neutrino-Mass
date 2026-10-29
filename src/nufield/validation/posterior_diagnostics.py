"""Proper-score and uncertainty summaries for posterior sample arrays."""

from __future__ import annotations

import numpy as np


def _samples_truth(samples, truth):
    s = np.asarray(samples, dtype=float)
    t = np.asarray(truth, dtype=float)
    if s.ndim != 3:
        raise ValueError("samples must have shape [N_observations,N_draws,N_parameters]")
    if t.ndim == 1:
        t = t[:, None]
    if t.shape != (s.shape[0], s.shape[2]):
        raise ValueError("truth shape must be [N,P] and match samples")
    if s.shape[1] < 2:
        raise ValueError("at least two posterior samples required")
    if not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError("samples and truth must be finite")
    return s, t


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS for empirical posterior samples, returned per observation and parameter."""
    s, t = _samples_truth(samples, truth)
    first = np.mean(np.abs(s - t[:, None, :]), axis=1)
    # E|X-X'| computed in O(S log S) after sorting, without allocating [N,S,S,P].
    sorted_s = np.sort(s, axis=1)
    n = s.shape[1]
    coeff = (2 * np.arange(1, n + 1) - n - 1).reshape(1, n, 1)
    pair_term = 2 * np.sum(coeff * sorted_s, axis=1) / (n * n)
    return first - 0.5 * pair_term


def energy_score(samples: np.ndarray, truth: np.ndarray, *, max_draws: int = 500) -> np.ndarray:
    """Multivariate energy score per observation; posterior draws may be subsampled reproducibly."""
    s, t = _samples_truth(samples, truth)
    if s.shape[1] > max_draws:
        s = s[:, :: int(np.ceil(s.shape[1] / max_draws)), :]
    n, d = s.shape[:2]
    score = np.mean(np.linalg.norm(s - t[:, None, :], axis=2), axis=1)
    # Pair term in chunks to cap intermediate memory.
    pair = np.zeros(n, dtype=float)
    for i in range(s.shape[1]):
        pair += np.linalg.norm(s - s[:, i : i + 1, :], axis=2).sum(axis=1)
    pair /= 2 * s.shape[1] ** 2
    return score - pair


def interval_coverage(
    samples: np.ndarray, truth: np.ndarray, *, levels: tuple[float, ...] = (0.68, 0.95)
) -> dict:
    s, t = _samples_truth(samples, truth)
    result = {}
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("coverage levels must lie in (0,1)")
        alpha = (1 - level) / 2
        lo = np.quantile(s, alpha, axis=1)
        hi = np.quantile(s, 1 - alpha, axis=1)
        covered = (t >= lo) & (t <= hi)
        result[str(level)] = {
            "coverage_by_parameter": covered.mean(axis=0).tolist(),
            "mean_width_by_parameter": (hi - lo).mean(axis=0).tolist(),
            "n_observations": len(t),
            "interpretation": "coverage is meaningful only for independent held-out draws from the declared prior/simulator",
        }
    return result


def posterior_diagnostic_report(
    samples: np.ndarray, truth: np.ndarray, parameter_names: list[str]
) -> dict:
    s, t = _samples_truth(samples, truth)
    if len(parameter_names) != s.shape[2]:
        raise ValueError("parameter_names length mismatch")
    crps = crps_ensemble(s, t)
    energy = energy_score(s, t)
    means = s.mean(axis=1)
    bias = (means - t).mean(axis=0)
    return {
        "n_observations": int(s.shape[0]),
        "n_samples_per_observation": int(s.shape[1]),
        "parameter_names": parameter_names,
        "mean_crps_by_parameter": crps.mean(axis=0).tolist(),
        "mean_energy_score": float(energy.mean()),
        "posterior_mean_bias_by_parameter": bias.tolist(),
        "coverage": interval_coverage(s, t),
        "warning": "These metrics do not certify posterior calibration; use SBC ranks, grouped held-out samples, and posterior predictive checks too.",
    }
