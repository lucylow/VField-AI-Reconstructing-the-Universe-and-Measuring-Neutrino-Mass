"""Matched posterior-archive comparison at the independent parent-simulation level."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nufield.validation.posterior_diagnostics import crps_ensemble, interval_coverage


def _load(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        required = {"samples", "truth", "simulation_id", "group_id"}
        missing = required - set(z.files)
        if missing:
            raise KeyError(f"posterior archive missing {sorted(missing)}")
        out = {k: np.asarray(z[k]) for k in required}
    out["samples"] = np.asarray(out["samples"], dtype=float)
    out["truth"] = np.asarray(out["truth"], dtype=float)
    out["simulation_id"] = out["simulation_id"].astype(str).reshape(-1)
    out["group_id"] = out["group_id"].astype(str).reshape(-1)
    n = len(out["simulation_id"])
    if (
        out["samples"].ndim != 3
        or out["samples"].shape[0] != n
        or out["truth"].shape != (n, out["samples"].shape[2])
    ):
        raise ValueError("posterior archive shapes are inconsistent")
    if len(out["group_id"]) != n or len(set(out["simulation_id"].tolist())) != n:
        raise ValueError("IDs must be unique and row-aligned")
    if not all(np.isfinite(out[k]).all() for k in ("samples", "truth")):
        raise ValueError("posterior arrays must be finite")
    return out


def _group_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    # Average row-level losses within the independence unit before bootstrap resampling.
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two independent parent groups required for comparison")
    ga = np.asarray([a[groups == g].mean() for g in unique])
    gb = np.asarray([b[groups == g].mean() for g in unique])
    diff = gb - ga
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(unique), size=(n_bootstrap, len(unique)))
    boot = diff[indexes].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "difference_mean_b_minus_a": float(diff.mean()),
        "confidence": confidence,
        "confidence_interval": [
            float(np.quantile(boot, alpha)),
            float(np.quantile(boot, 1 - alpha)),
        ],
        "n_independent_groups": int(len(unique)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "metric_direction": "lower is better; negative means model B has lower normalized CRPS than model A",
    }


def compare_posterior_archives(
    path_a: str | Path,
    path_b: str | Path,
    *,
    parameter_names: list[str],
    prior_low: list[float],
    prior_high: list[float],
    n_bootstrap: int = 3000,
    confidence: float = 0.95,
    seed: int = 41,
) -> dict[str, Any]:
    """Compare two posterior archives on exactly matched test simulations/groups.

    Scores are dimensionless CRPS (per-parameter CRPS divided by the declared prior range) so
    averaging across parameters with different units is not silently treated as meaningful.
    """
    a = _load(path_a)
    b = _load(path_b)
    names = list(parameter_names)
    low = np.asarray(prior_low, dtype=float)
    high = np.asarray(prior_high, dtype=float)
    p = a["samples"].shape[2]
    if len(names) != p or low.shape != (p,) or high.shape != (p,) or np.any(high <= low):
        raise ValueError("parameter names/prior bounds do not match posterior dimension")
    if a["samples"].shape[2] != b["samples"].shape[2]:
        raise ValueError("posterior parameter dimensions differ")
    map_b = {sid: i for i, sid in enumerate(b["simulation_id"].tolist())}
    if set(a["simulation_id"].tolist()) != set(b["simulation_id"].tolist()):
        raise ValueError("archives must contain the exact same held-out simulation IDs")
    order_b = np.asarray([map_b[sid] for sid in a["simulation_id"].tolist()], dtype=int)
    b_truth = b["truth"][order_b]
    b_group = b["group_id"][order_b]
    if not np.allclose(a["truth"], b_truth, rtol=0, atol=1e-10):
        raise ValueError("truth labels differ between matched archives")
    if not np.array_equal(a["group_id"], b_group):
        raise ValueError("parent group IDs differ for matched simulation IDs")
    sa = a["samples"]
    sb = b["samples"][order_b]
    truth = a["truth"]
    groups = a["group_id"]
    crps_a = crps_ensemble(sa, truth) / (high - low)[None, :]
    crps_b = crps_ensemble(sb, truth) / (high - low)[None, :]
    perrow_a = crps_a.mean(axis=1)
    perrow_b = crps_b.mean(axis=1)
    diff = _group_bootstrap_difference(
        perrow_a, perrow_b, groups, n_bootstrap=n_bootstrap, seed=seed, confidence=confidence
    )
    coverage_a = interval_coverage(sa, truth)
    coverage_b = interval_coverage(sb, truth)
    coverage_diff = {}
    for level in coverage_a:
        coverage_diff[level] = {
            "model_a": coverage_a[level]["coverage_by_parameter"],
            "model_b": coverage_b[level]["coverage_by_parameter"],
            "difference_b_minus_a": (
                np.asarray(coverage_b[level]["coverage_by_parameter"])
                - np.asarray(coverage_a[level]["coverage_by_parameter"])
            ).tolist(),
            "nominal": float(level),
        }
    return {
        "schema_version": "1.0",
        "model_a_archive": str(Path(path_a).resolve()),
        "model_b_archive": str(Path(path_b).resolve()),
        "n_test_rows": int(len(truth)),
        "n_parent_groups": int(len(np.unique(groups))),
        "parameter_names": names,
        "normalized_crps_mean_by_parameter_model_a": crps_a.mean(axis=0).tolist(),
        "normalized_crps_mean_by_parameter_model_b": crps_b.mean(axis=0).tolist(),
        "normalized_crps_group_bootstrap_b_minus_a": diff,
        "coverage_comparison": coverage_diff,
        "matching_policy": "exact matched simulation IDs, same truths, same parent-group assignments; bootstrap parent groups",
        "scientific_approval": False,
        "warning": "A comparative score does not establish physical validity. Archives should come from a locked held-out partition, not validation-tuned reuse.",
    }
