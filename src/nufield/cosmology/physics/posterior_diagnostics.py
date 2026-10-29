"""Posterior calibration and interval diagnostics for held-out cosmological simulations.

This module contains distribution diagnostics, not an inference engine. Input posterior samples must be
obtained without tuning on the held-out truths. Calibration claims require many independent simulations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class IntervalCoverage:
    nominal_level: float
    empirical_coverage: float
    n_observations: int
    standard_error_binomial: float
    covered: np.ndarray


def equal_tailed_interval(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Return an equal-tailed credible interval from finite posterior samples."""
    x = np.asarray(samples, float)
    if x.ndim != 1 or x.size < 2 or np.any(~np.isfinite(x)) or not 0 < level < 1:
        raise ValueError("samples must be a finite vector with at least two values; level in (0,1)")
    alpha = 0.5 * (1.0 - level)
    lo, hi = np.quantile(x, [alpha, 1 - alpha])
    return float(lo), float(hi)


def interval_coverage(
    truths: np.ndarray,
    posterior_samples: np.ndarray,
    nominal_levels: tuple[float, ...] = (0.68, 0.95),
) -> tuple[IntervalCoverage, ...]:
    """Evaluate equal-tailed posterior interval coverage across repeated held-out observations.

    posterior_samples must have shape (n_observation,n_draw). A single posterior's credible interval
    is not a coverage estimate; the outer dimension must represent repeated calibration cases.
    """
    t, s = np.asarray(truths, float), np.asarray(posterior_samples, float)
    if t.ndim != 1 or s.ndim != 2 or s.shape[0] != t.size or s.shape[1] < 2:
        raise ValueError(
            "truths and posterior_samples must have shapes (n_case,) and (n_case,n_draw)"
        )
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(s)):
        raise ValueError("truths and posterior samples must be finite")
    results = []
    for level in nominal_levels:
        if not 0 < level < 1:
            raise ValueError("nominal levels must be in (0,1)")
        alpha = (1 - level) / 2
        lower = np.quantile(s, alpha, axis=1)
        upper = np.quantile(s, 1 - alpha, axis=1)
        covered = (t >= lower) & (t <= upper)
        p = float(covered.mean())
        n = t.size
        se = float(np.sqrt(p * (1 - p) / n))
        results.append(IntervalCoverage(float(level), p, n, se, covered))
    return tuple(results)


def randomized_pit(truths: np.ndarray, posterior_samples: np.ndarray, *, seed: int) -> np.ndarray:
    """Compute a randomized empirical PIT value per case from discrete posterior draws.

    For continuous calibrated posteriors PIT values are uniform across independent cases. Finite sample
    ranks require tie randomization; this implementation randomizes uniformly within tied empirical CDF mass.
    """
    t, s = np.asarray(truths, float), np.asarray(posterior_samples, float)
    if t.ndim != 1 or s.ndim != 2 or s.shape[0] != t.size or s.shape[1] < 2 or seed < 0:
        raise ValueError("invalid truths/samples shapes or seed")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(s)):
        raise ValueError("truths and samples must be finite")
    rng = np.random.default_rng(seed)
    out = np.empty(t.size)
    for i, (truth, draws) in enumerate(zip(t, s)):
        n = draws.size
        less = np.count_nonzero(draws < truth)
        equal = np.count_nonzero(draws == truth)
        out[i] = (less + rng.uniform(0, equal + 1)) / (n + 1)
    return out


def pit_histogram(pit_values: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram counts and bin edges for PIT calibration plotting."""
    pit = np.asarray(pit_values, float)
    if (
        pit.ndim != 1
        or pit.size < 1
        or np.any(~np.isfinite(pit))
        or np.any((pit < 0) | (pit > 1))
        or n_bins < 2
    ):
        raise ValueError("PIT values must lie in [0,1] and n_bins>=2")
    counts, edges = np.histogram(pit, bins=n_bins, range=(0, 1))
    return counts, edges


def expected_calibration_rms(empirical: np.ndarray, nominal: np.ndarray) -> float:
    """Summarize absolute coverage deviations by unweighted RMS; no statistical correction applied."""
    e, n = np.asarray(empirical, float), np.asarray(nominal, float)
    if (
        e.shape != n.shape
        or e.ndim != 1
        or np.any(~np.isfinite(e + n))
        or np.any((e < 0) | (e > 1))
        or np.any((n < 0) | (n > 1))
    ):
        raise ValueError("empirical and nominal coverage must match and lie in [0,1]")
    return float(np.sqrt(np.mean((e - n) ** 2)))


__all__ = [
    "IntervalCoverage",
    "equal_tailed_interval",
    "interval_coverage",
    "randomized_pit",
    "pit_histogram",
    "expected_calibration_rms",
]
