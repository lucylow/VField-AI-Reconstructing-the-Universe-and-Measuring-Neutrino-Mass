"""Post-split support and distribution diagnostics for grouped simulation partitions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from nufield.data_registry.splits import assert_group_disjoint


def diagnose_split_support(
    theta: np.ndarray,
    parameter_names: list[str],
    group_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    target_name: str = "sum_mnu_ev",
    require_full_partition: bool = True,
) -> dict[str, Any]:
    y = np.asarray(theta, dtype=float)
    groups = np.asarray(group_ids).astype(str)
    if y.ndim != 2 or y.shape[1] != len(parameter_names):
        raise ValueError("theta/parameter_names mismatch")
    if len(groups) != len(y):
        raise ValueError("group IDs do not align with theta")
    if target_name not in parameter_names:
        raise ValueError(f"target {target_name!r} absent")
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError(f"splits must have exactly {sorted(required)}")
    clean = {k: np.asarray(v, dtype=int).reshape(-1) for k, v in splits.items()}
    for k, idx in clean.items():
        if len(idx) == 0:
            raise ValueError(f"empty {k} partition")
        if np.any(idx < 0) or np.any(idx >= len(y)):
            raise ValueError(f"out-of-range indices in {k}")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"duplicate row indices in {k}")
    assert_group_disjoint(groups, clean)
    concat = np.concatenate(list(clean.values()))
    if len(np.unique(concat)) != len(concat):
        raise ValueError("a row occurs in more than one partition")
    omitted = sorted(set(range(len(y))) - set(concat.tolist()))
    if require_full_partition and omitted:
        raise ValueError(f"{len(omitted)} rows omitted by split")
    j = parameter_names.index(target_name)
    vals = y[:, j]
    result = {}
    warnings = []
    for name, idx in clean.items():
        v = vals[idx]
        result[name] = {
            "n_rows": len(idx),
            "n_parent_groups": int(np.unique(groups[idx]).size),
            "target_min": float(v.min()),
            "target_max": float(v.max()),
            "target_unique": int(np.unique(v).size),
            "target_quantiles": np.quantile(v, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
            "target_unique_values": np.unique(v).tolist() if np.unique(v).size <= 30 else None,
        }
    train = vals[clean["train"]]
    for name in ("validation", "test"):
        v = vals[clean[name]]
        outside = bool(v.min() < train.min() or v.max() > train.max())
        ks = ks_2samp(train, v, method="auto")
        result[name]["outside_train_range"] = outside
        result[name]["ks_statistic_vs_train"] = float(ks.statistic)
        result[name]["ks_pvalue_descriptive_only"] = float(ks.pvalue)
        if outside:
            warnings.append(
                f"{name} target range extends beyond train support; score it as extrapolation, not interpolation."
            )
        if np.unique(v).size < 2:
            warnings.append(
                f"{name} contains fewer than two target values; calibration across the target is weakly assessed."
            )
    if omitted:
        warnings.append(
            "Some rows are unassigned. This can be intentional only when documented in the split manifest."
        )
    return {
        "target_name": target_name,
        "n_rows": len(y),
        "split_summary": result,
        "omitted_rows": omitted,
        "group_disjoint": True,
        "rows_unique_across_partitions": True,
        "warnings": warnings,
        "warning": "KS p-values are descriptive and should not be used as a split-quality pass/fail; check prior support and independent-group count.",
    }
