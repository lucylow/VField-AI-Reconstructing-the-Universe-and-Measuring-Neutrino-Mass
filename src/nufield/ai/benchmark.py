"""Group-cluster bootstrap comparisons for AI model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate(y: Any, pred: Any, groups: Any):
    target = np.asarray(y, float)
    prediction = np.asarray(pred, float)
    ids = np.asarray(groups).reshape(-1).astype(str)
    if target.ndim == 1:
        target = target[:, None]
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if target.shape != prediction.shape or target.ndim != 2 or len(ids) != len(target):
        raise ValueError("target/prediction shape mismatch or group alignment invalid")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("target and prediction must be finite")
    return target, prediction, ids


def _score(target: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    residual = prediction - target
    if metric == "rmse":
        return float(np.sqrt(np.mean(residual**2)))
    if metric == "mae":
        return float(np.abs(residual).mean())
    if metric == "bias_abs":
        return float(np.abs(residual.mean(axis=0)).mean())
    if metric == "mse":
        return float(np.mean(residual**2))
    raise ValueError("metric must be rmse/mae/bias_abs/mse")


def cluster_bootstrap_metric(
    target: Any,
    prediction: Any,
    groups: Any,
    *,
    metric: str = "rmse",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap a metric by resampling independent groups, not correlated rows."""
    y, p, ids = _validate(target, prediction, groups)
    unique = np.unique(ids)
    if len(unique) < 5 or n_bootstrap < 100 or not 0 < alpha < 1:
        raise ValueError("need >=5 groups, >=100 bootstrap reps and alpha in (0,1)")
    row_indices = {g: np.flatnonzero(ids == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, float)
    for b in range(n_bootstrap):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([row_indices[g] for g in chosen])
        values[b] = _score(y[ix], p[ix], metric)
    return {
        "metric": metric,
        "estimate": _score(y, p, metric),
        "bootstrap_mean": float(values.mean()),
        "confidence_interval": [
            float(np.quantile(values, alpha / 2)),
            float(np.quantile(values, 1 - alpha / 2)),
        ],
        "alpha": alpha,
        "n_groups": int(len(unique)),
        "n_rows": int(len(y)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "warning": "Cluster bootstrap assumes parent groups are the independent resampling units.",
    }


def paired_cluster_bootstrap_difference(
    target: Any,
    prediction_a: Any,
    prediction_b: Any,
    groups: Any,
    *,
    metric: str = "rmse",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired group-bootstrap difference score(A)-score(B) on identical rows/groups.

    A negative difference favors model A when the chosen metric is lower-is-better. Pairing matters:
    both models are evaluated on the same held-out simulations, so bootstrap draws must be paired.
    """
    y, a, ids = _validate(target, prediction_a, groups)
    yb, b, idb = _validate(target, prediction_b, groups)
    if y.shape != yb.shape or not np.array_equal(ids, idb):
        raise ValueError("models must have aligned predictions on identical held-out rows/groups")
    unique = np.unique(ids)
    if len(unique) < 5 or n_bootstrap < 100 or not 0 < alpha < 1:
        raise ValueError("need >=5 groups, >=100 bootstrap reps and alpha in (0,1)")
    row_indices = {g: np.flatnonzero(ids == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(n_bootstrap, float)
    for k in range(n_bootstrap):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([row_indices[g] for g in chosen])
        values[k] = _score(y[ix], a[ix], metric) - _score(y[ix], b[ix], metric)
    delta = _score(y, a, metric) - _score(y, b, metric)
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric,
        "score_a": _score(y, a, metric),
        "score_b": _score(y, b, metric),
        "difference_a_minus_b": float(delta),
        "confidence_interval": [float(lower), float(upper)],
        "interval_excludes_zero": bool(lower > 0 or upper < 0),
        "alpha": alpha,
        "n_groups": int(len(unique)),
        "n_bootstrap": int(n_bootstrap),
        "seed": seed,
        "interpretation": "Negative difference favors A for lower-is-better metrics; use domain relevance and calibration too.",
    }


def repeated_seed_summary(scores: Any) -> dict[str, Any]:
    """Summarize independent-seed model scores without treating runs as independent simulations."""
    x = np.asarray(scores, float).reshape(-1)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("at least two finite seed scores are required")
    return {
        "n_seeds": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "median": float(np.median(x)),
        "max": float(x.max()),
        "warning": "Seed variability and simulation-sample uncertainty are distinct sources of variation.",
    }
