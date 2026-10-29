"""Compact posterior diagnostics across independent observations."""

from __future__ import annotations

import numpy as np


def posterior_summary(samples: np.ndarray, levels=(0.68, 0.95)) -> dict:
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2 or len(s) < 2 or not np.isfinite(s).all():
        raise ValueError("samples must be finite (N_draws,N_parameters)")
    result = {
        "mean": s.mean(0).tolist(),
        "median": np.median(s, axis=0).tolist(),
        "std": s.std(0, ddof=1).tolist(),
    }
    result["quantiles"] = {}
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("levels must be in (0,1)")
        alpha = (1 - level) / 2
        result["quantiles"][str(level)] = {
            "lower": np.quantile(s, alpha, axis=0).tolist(),
            "upper": np.quantile(s, 1 - alpha, axis=0).tolist(),
        }
    return result


def standardized_residual_summary(residuals: np.ndarray) -> dict[str, float]:
    r = np.asarray(residuals, dtype=float).ravel()
    if len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("residuals must be finite and contain at least two values")
    return {
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "fraction_abs_below_1": float(np.mean(np.abs(r) < 1)),
        "fraction_abs_below_2": float(np.mean(np.abs(r) < 2)),
        "fraction_abs_below_3": float(np.mean(np.abs(r) < 3)),
    }
