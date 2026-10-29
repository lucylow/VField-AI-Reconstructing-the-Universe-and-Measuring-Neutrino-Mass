"""Posterior-sample metrics designed for repeated simulation-based evaluation."""

from __future__ import annotations

import numpy as np


def _validate(samples: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(samples, dtype=float)
    t = np.asarray(truth, dtype=float)
    if s.ndim != 3:
        raise ValueError("samples must have shape [N_simulations,N_posterior_samples,P]")
    if t.shape != (s.shape[0], s.shape[2]):
        raise ValueError("truth must have shape [N_simulations,P]")
    if s.shape[1] < 2:
        raise ValueError("at least two posterior samples are required")
    if not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError("inputs must be finite")
    return s, t


def posterior_sample_metrics(
    samples: np.ndarray, truth: np.ndarray, *, levels: tuple[float, ...] = (0.68, 0.95)
) -> dict:
    """Aggregate bias, RMSE, empirical central-interval coverage and average interval width."""
    s, t = _validate(samples, truth)
    med = np.median(s, axis=1)
    mean = s.mean(axis=1)
    bias = mean - t
    rmse = np.sqrt(np.mean((med - t) ** 2, axis=0))
    per_parameter = []
    for j in range(s.shape[2]):
        row = {
            "parameter_index": j,
            "truth_mean": float(t[:, j].mean()),
            "posterior_median_bias": float(np.mean(med[:, j] - t[:, j])),
            "posterior_median_rmse": float(rmse[j]),
            "mean_posterior_std": float(np.mean(s[:, :, j].std(axis=1))),
        }
        for level in levels:
            if not 0 < level < 1:
                raise ValueError("coverage levels must be between 0 and 1")
            alpha = (1 - level) / 2
            low = np.quantile(s[:, :, j], alpha, axis=1)
            high = np.quantile(s[:, :, j], 1 - alpha, axis=1)
            row[f"coverage_{int(level * 100)}"] = float(
                np.mean((t[:, j] >= low) & (t[:, j] <= high))
            )
            row[f"width_{int(level * 100)}"] = float(np.mean(high - low))
        per_parameter.append(row)
    return {
        "n_simulations": int(s.shape[0]),
        "n_samples_per_posterior": int(s.shape[1]),
        "n_parameters": int(s.shape[2]),
        "per_parameter": per_parameter,
        "warning": "Coverage estimates need many independent simulations and binomial uncertainty intervals; this function alone does not certify calibration.",
    }


def coverage_by_parameter(
    samples: np.ndarray, truth: np.ndarray, *, levels: tuple[float, ...] = (0.50, 0.68, 0.90, 0.95)
) -> list[dict]:
    s, t = _validate(samples, truth)
    results = []
    for j in range(s.shape[2]):
        for level in levels:
            if not 0 < level < 1:
                raise ValueError("levels must lie in (0,1)")
            alpha = (1 - level) / 2
            lo = np.quantile(s[:, :, j], alpha, axis=1)
            hi = np.quantile(s[:, :, j], 1 - alpha, axis=1)
            ok = (t[:, j] >= lo) & (t[:, j] <= hi)
            results.append(
                {
                    "parameter_index": j,
                    "nominal_level": level,
                    "empirical_coverage": float(ok.mean()),
                    "n_trials": int(len(ok)),
                    "n_covered": int(ok.sum()),
                    "mean_interval_width": float(np.mean(hi - lo)),
                }
            )
    return results


def posterior_rank(
    samples: np.ndarray, truth: np.ndarray, *, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Randomized posterior ranks for SBC, one rank per simulation and parameter."""
    s, t = _validate(samples, truth)
    rng = rng or np.random.default_rng(0)
    ranks = np.zeros((s.shape[0], s.shape[2]), dtype=np.int64)
    for i in range(s.shape[0]):
        for j in range(s.shape[2]):
            below = int(np.count_nonzero(s[i, :, j] < t[i, j]))
            tied = int(np.count_nonzero(s[i, :, j] == t[i, j]))
            ranks[i, j] = below + int(rng.integers(tied + 1))
    return ranks


def energy_score(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    max_samples: int = 512,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte-Carlo energy score over scaled parameter coordinates for multivariate posteriors.

    Caller should scale parameters using predeclared/prior scales before calling; mixed physical
    units make raw Euclidean distance uninterpretable.
    """
    s, t = _validate(samples, truth)
    rng = rng or np.random.default_rng(0)
    values = []
    for i in range(len(t)):
        si = s[i]
        if len(si) > max_samples:
            si = si[rng.choice(len(si), max_samples, replace=False)]
        d_truth = np.linalg.norm(si - t[i][None, :], axis=1).mean()
        pair = np.linalg.norm(si[:, None, :] - si[None, :, :], axis=-1).mean()
        values.append(d_truth - 0.5 * pair)
    return float(np.mean(values))
