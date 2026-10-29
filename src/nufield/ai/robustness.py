"""Robustness scoring by simulation domain and predeclared systematic condition."""

from __future__ import annotations

from typing import Any

import numpy as np


def groupwise_metrics(
    target: Any, prediction: Any, groups: Any, *, metric: str = "rmse"
) -> dict[str, Any]:
    """Compute one prediction metric per named domain/group and report worst-group performance."""
    y, p = np.asarray(target, float), np.asarray(prediction, float)
    g = np.asarray(groups).reshape(-1).astype(str)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    if (
        y.shape != p.shape
        or y.ndim != 2
        or len(g) != len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("target/prediction must be finite [N,P] and align with groups")
    if metric not in {"rmse", "mae", "bias_abs"}:
        raise ValueError("metric must be rmse/mae/bias_abs")
    result = []
    residual = p - y
    for group in np.unique(g):
        r = residual[g == group]
        if metric == "rmse":
            value = float(np.sqrt(np.mean(r**2)))
        elif metric == "mae":
            value = float(np.abs(r).mean())
        else:
            value = float(np.abs(r.mean()))
        result.append(
            {
                "group": str(group),
                "n": int(len(r)),
                "score": value,
                "bias_by_target": r.mean(axis=0).tolist(),
            }
        )
    scores = np.asarray([x["score"] for x in result])
    return {
        "metric": metric,
        "groups": result,
        "macro_mean": float(scores.mean()),
        "worst_group": float(scores.max()),
        "best_group": float(scores.min()),
        "worst_to_mean_ratio": float(scores.max() / max(scores.mean(), 1e-12)),
    }


def perturbation_sweep_report(
    target: Any,
    baseline_predictions: Any,
    perturbed_predictions: dict[str, Any],
    *,
    parameter_scale: Any | None = None,
) -> dict[str, Any]:
    """Compare predeclared perturbation predictions to baseline in target-scaled units."""
    y, base = np.asarray(target, float), np.asarray(baseline_predictions, float)
    if y.ndim == 1:
        y = y[:, None]
    if base.ndim == 1:
        base = base[:, None]
    if y.shape != base.shape or not np.isfinite(y).all() or not np.isfinite(base).all():
        raise ValueError("targets and baseline predictions must be finite and same-shaped")
    scale = np.ones(y.shape[1]) if parameter_scale is None else np.asarray(parameter_scale, float)
    if scale.shape != (y.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("parameter_scale must be positive [P]")
    baseline_error = ((base - y) / scale).reshape(len(y), -1)
    records = {}
    for name, value in perturbed_predictions.items():
        pred = np.asarray(value, float)
        if pred.ndim == 1:
            pred = pred[:, None]
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError(f"perturbed predictions for {name} have invalid shape/values")
        err = ((pred - y) / scale).reshape(len(y), -1)
        records[name] = {
            "rmse_scaled": float(np.sqrt(np.mean(err**2))),
            "mae_scaled": float(np.abs(err).mean()),
            "bias_scaled": err.mean(axis=0).tolist(),
            "error_increase_vs_baseline": float(np.mean(err**2) - np.mean(baseline_error**2)),
        }
    return {
        "baseline_rmse_scaled": float(np.sqrt(np.mean(baseline_error**2))),
        "perturbations": records,
        "warning": "Results are meaningful only for declared and physically plausible perturbations.",
    }
