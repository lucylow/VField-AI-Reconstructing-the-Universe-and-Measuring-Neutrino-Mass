"""Finite-sample uncertainty intervals for empirical posterior coverage and SBC ranks."""

from __future__ import annotations

import numpy as np


def wilson_interval(
    successes: int, trials: int, *, z_value: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; uses no large-sample Wald approximation."""
    if trials < 1 or successes < 0 or successes > trials or z_value <= 0:
        raise ValueError("require trials>=1, 0<=successes<=trials, and z_value>0")
    p = successes / trials
    z2 = z_value * z_value
    den = 1.0 + z2 / trials
    center = (p + z2 / (2 * trials)) / den
    radius = z_value * np.sqrt(p * (1 - p) / trials + z2 / (4 * trials**2)) / den
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def coverage_interval_report(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.50, 0.68, 0.90, 0.95),
    ci_z: float = 1.959963984540054,
) -> dict:
    """Compute central-interval coverage and Wilson uncertainty intervals for every parameter."""
    s = np.asarray(samples, dtype=float)
    t = np.asarray(truth, dtype=float)
    if s.ndim != 3 or t.shape != (s.shape[0], s.shape[2]) or s.shape[1] < 2:
        raise ValueError("samples must be [N,S,P], truth [N,P], and S>=2")
    if not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError("samples/truth must be finite")
    if any(not 0 < level < 1 for level in levels):
        raise ValueError("coverage levels must be in (0,1)")
    results = []
    for j in range(s.shape[2]):
        for level in levels:
            alpha = (1 - level) / 2
            lo = np.quantile(s[:, :, j], alpha, axis=1)
            hi = np.quantile(s[:, :, j], 1 - alpha, axis=1)
            covered = (t[:, j] >= lo) & (t[:, j] <= hi)
            count = int(covered.sum())
            n = int(len(covered))
            low, high = wilson_interval(count, n, z_value=ci_z)
            results.append(
                {
                    "parameter_index": j,
                    "nominal_coverage": float(level),
                    "empirical_coverage": count / n,
                    "coverage_ci_low": low,
                    "coverage_ci_high": high,
                    "n_covered": count,
                    "n_trials": n,
                    "mean_interval_width": float((hi - lo).mean()),
                    "nominal_inside_ci": bool(low <= level <= high),
                    "warning": "few trials: coverage uncertainty is wide" if n < 100 else None,
                }
            )
    return {
        "results": results,
        "n_simulations": int(len(t)),
        "caveat": "Binomial intervals quantify simulation-to-simulation sampling uncertainty; they do not validate the assumed simulator or prior.",
    }


def rank_histogram(ranks: np.ndarray, *, n_posterior_samples: int, n_bins: int = 20) -> dict:
    """Build one-dimensional rank histogram counts for SBC diagnostic plots.

    Ranks are integers in [0,n_posterior_samples], typically returned by posterior_rank. A roughly
    uniform histogram is expected under calibrated inference and matched generative assumptions, but
    histogram uniformity alone is not proof of calibration.
    """
    r = np.asarray(ranks)
    if r.ndim != 2 or r.size == 0 or n_posterior_samples < 1 or n_bins < 2:
        raise ValueError("ranks must be [N,P], n_posterior_samples>=1, n_bins>=2")
    if not np.issubdtype(r.dtype, np.integer):
        raise ValueError("ranks must be integer-valued")
    if np.any(r < 0) or np.any(r > n_posterior_samples):
        raise ValueError("rank lies outside [0,n_posterior_samples]")
    edges = np.linspace(0, n_posterior_samples + 1, n_bins + 1)
    counts = np.histogram(r.ravel(), bins=edges)[0]
    expected = float(r.size) / n_bins
    return {
        "counts": counts.astype(int).tolist(),
        "expected_count_per_bin": expected,
        "edges": edges.tolist(),
        "n_ranks": int(r.size),
        "interpretation": "Inspect U-shaped, skewed, or central rank patterns; use more independent simulations and calibrated posterior diagnostics.",
    }
