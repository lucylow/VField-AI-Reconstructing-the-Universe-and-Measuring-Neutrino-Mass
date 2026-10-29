"""Calibration diagnostics. No automatic test-set recalibration is performed."""

from __future__ import annotations

import numpy as np


def empirical_coverage(indicators: np.ndarray) -> dict[str, float]:
    """Summarize Boolean coverage indicators and approximate binomial standard error."""
    x = np.asarray(indicators, dtype=bool)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("indicators must be a non-empty vector")
    p = float(x.mean())
    se = float(np.sqrt(p * (1 - p) / len(x)))
    return {
        "n": int(len(x)),
        "coverage": p,
        "binomial_standard_error": se,
        "normal_approx_95_lower": max(0.0, p - 1.96 * se),
        "normal_approx_95_upper": min(1.0, p + 1.96 * se),
    }


def rank_calibration_summary(ranks: np.ndarray, n_posterior_draws: int) -> dict[str, float]:
    """Summarize randomized rank uniformity using a descriptive chi-squared statistic."""
    r = np.asarray(ranks, dtype=int)
    if r.ndim != 1 or len(r) == 0 or n_posterior_draws < 1:
        raise ValueError("invalid ranks")
    if np.any((r < 0) | (r > n_posterior_draws)):
        raise ValueError("ranks outside valid support")
    bins = min(10, n_posterior_draws + 1)
    hist = np.histogram(r, bins=np.linspace(-0.5, n_posterior_draws + 0.5, bins + 1))[0]
    expected = len(r) / bins
    chi2 = float(np.sum((hist - expected) ** 2 / max(expected, 1e-12)))
    return {
        "n_replicates": int(len(r)),
        "n_bins": int(bins),
        "chi_square_descriptive": chi2,
        "histogram": hist.tolist(),
        "note": "Use a calibrated goodness-of-fit test and finite-sample bands for formal assessment.",
    }
