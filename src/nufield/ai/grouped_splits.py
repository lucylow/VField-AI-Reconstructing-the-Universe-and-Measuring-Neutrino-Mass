"""Deterministic group-disjoint splits with optional target-range stratification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nufield.ai.contracts import assert_group_disjoint


@dataclass(frozen=True)
class GroupSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    seed: int
    stratifier_bins: int

    def validate(self) -> None:
        assert_group_disjoint(
            self.train_groups,
            self.validation_groups,
            np.concatenate([self.calibration_groups, self.test_groups]),
        )
        if set(self.calibration_groups).intersection(self.test_groups):
            raise ValueError("calibration and test groups overlap")
        all_i = np.concatenate(
            [
                self.train_indices,
                self.validation_indices,
                self.calibration_indices,
                self.test_indices,
            ]
        )
        if len(np.unique(all_i)) != len(all_i):
            raise ValueError("one or more rows appear in multiple partitions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_indices": self.train_indices.tolist(),
            "validation_indices": self.validation_indices.tolist(),
            "calibration_indices": self.calibration_indices.tolist(),
            "test_indices": self.test_indices.tolist(),
            "train_groups": list(self.train_groups),
            "validation_groups": list(self.validation_groups),
            "calibration_groups": list(self.calibration_groups),
            "test_groups": list(self.test_groups),
            "seed": self.seed,
            "stratifier_bins": self.stratifier_bins,
            "warning": "Group-level stratification reduces range imbalance but does not replace a held-out cosmology/domain test.",
        }


def grouped_train_validation_calibration_test_split(
    group_ids: Any,
    stratifier: Any,
    *,
    fractions: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
    n_bins: int = 5,
    seed: int = 42,
) -> GroupSplit:
    """Split rows by unique parent group while approximately preserving a scalar target distribution.

    All rows sharing a parent group stay together. If group labels include augmentations or repeated
    observations, every row in a group must have approximately the same stratification value. The
    function uses quantile bins of group-level mean values, randomized within each bin.
    """
    groups = np.asarray(group_ids).reshape(-1).astype(str)
    values = np.asarray(stratifier, float).reshape(-1)
    if len(groups) != len(values) or len(groups) < 8 or not np.isfinite(values).all():
        raise ValueError(
            "group_ids and stratifier must align and contain at least eight finite rows"
        )
    if len(fractions) != 4 or any(f <= 0 for f in fractions):
        raise ValueError("four positive fractions are required")
    if not np.isclose(sum(fractions), 1.0, atol=1e-8):
        raise ValueError("fractions must sum to one")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    unique = np.unique(groups)
    if len(unique) < 8:
        raise ValueError("at least eight independent parent groups are recommended")
    group_values = np.empty(len(unique), float)
    group_rows = {}
    for i, g in enumerate(unique):
        idx = np.flatnonzero(groups == g)
        group_rows[g] = idx
        vals = values[idx]
        if np.ptp(vals) > 1e-6 * max(1.0, abs(float(vals.mean()))):
            raise ValueError(
                f"stratifier varies within parent group {g}; pass a stable group-level target"
            )
        group_values[i] = vals.mean()
    quantile_edges = np.unique(np.quantile(group_values, np.linspace(0, 1, n_bins + 1)))
    bins = np.digitize(group_values, quantile_edges[1:-1], right=True)
    rng = np.random.default_rng(seed)
    split_groups: list[list[str]] = [[], [], [], []]
    for b in np.unique(bins):
        bucket = unique[bins == b].copy()
        rng.shuffle(bucket)
        n_bucket = len(bucket)
        raw = np.asarray(fractions) * n_bucket
        counts = np.floor(raw).astype(int)
        remainder = n_bucket - counts.sum()
        if remainder:
            order = np.argsort(-(raw - counts))
            counts[order[:remainder]] += 1
        # Keep nonzero quantile buckets represented across partitions when feasible. The final
        # global repair step below ensures that every partition has at least one independent group.
        start = 0
        for i, count in enumerate(counts):
            split_groups[i].extend(bucket[start : start + count].tolist())
            start += count
    # Rounding can leave a small partition empty for very small datasets. Repair by moving groups
    # from the currently largest partition without splitting any parent simulation.
    for i in range(4):
        if not split_groups[i]:
            donor = max(range(4), key=lambda j: len(split_groups[j]))
            if len(split_groups[donor]) <= 1:
                raise ValueError("not enough groups to populate all four partitions")
            split_groups[i].append(split_groups[donor].pop())
    # Ensure every group assigned exactly once, preserving original group order only in returned IDs.
    assigned = [g for part in split_groups for g in part]
    if len(assigned) != len(unique) or len(set(assigned)) != len(unique):
        raise RuntimeError("internal split allocation failed")
    row_indices = [
        np.sort(np.concatenate([group_rows[g] for g in part])).astype(int) for part in split_groups
    ]
    result = GroupSplit(
        *row_indices,
        *(tuple(sorted(part)) for part in split_groups),
        seed=int(seed),
        stratifier_bins=int(len(np.unique(bins))),
    )
    result.validate()
    if len(np.concatenate(row_indices)) != len(groups):
        raise RuntimeError("split did not allocate all input rows")
    return result


def split_distribution_report(split: GroupSplit, group_ids: Any, stratifier: Any) -> dict[str, Any]:
    """Summarize range balance and group counts without accessing model performance."""
    groups = np.asarray(group_ids).reshape(-1).astype(str)
    values = np.asarray(stratifier, float).reshape(-1)
    if len(groups) != len(values):
        raise ValueError("group_ids and stratifier length mismatch")
    report = {}
    for name in ("train", "validation", "calibration", "test"):
        idx = getattr(split, f"{name}_indices")
        report[name] = {
            "rows": int(len(idx)),
            "groups": int(len(np.unique(groups[idx]))),
            "stratifier_min": float(values[idx].min()),
            "stratifier_median": float(np.median(values[idx])),
            "stratifier_max": float(values[idx].max()),
            "mean": float(values[idx].mean()),
            "std": float(values[idx].std()),
        }
    return {
        "partitions": report,
        "group_ids_disjoint": True,
        "caveat": "Random group splits do not guarantee extrapolation to unseen cosmological domains.",
    }
