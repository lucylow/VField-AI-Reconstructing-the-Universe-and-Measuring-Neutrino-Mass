"""Group-aware cross-validation utilities for simulations with matched phases or multiple views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.data_registry.splits import SplitError, assert_group_disjoint


@dataclass(frozen=True)
class GroupFold:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    target_train_range: tuple[float, float] | None = None
    target_test_range: tuple[float, float] | None = None

    def to_dict(self):
        return {
            "fold": self.fold,
            "n_train": int(len(self.train_indices)),
            "n_test": int(len(self.test_indices)),
            "train_groups": list(self.train_groups),
            "test_groups": list(self.test_groups),
            "target_train_range": list(self.target_train_range)
            if self.target_train_range
            else None,
            "target_test_range": list(self.target_test_range) if self.target_test_range else None,
        }


def group_kfold_indices(
    group_ids: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    stratify_values: np.ndarray | None = None,
    n_bins: int = 5,
) -> list[GroupFold]:
    """Build reproducible disjoint folds, optionally balancing a continuous target by quantile bin.

    Fold assignment is at the unique group level. If one group contains multiple rows, all rows stay
    in one test fold. `stratify_values` only affects fold allocation; it does not change the target
    distribution and does not guarantee interpolation support. Inspect each returned target range.
    """
    groups = np.asarray(group_ids).astype(str)
    if groups.ndim != 1 or len(groups) < 3:
        raise SplitError("group_ids must be 1-D and contain at least 3 rows")
    unique = np.unique(groups)
    if not 2 <= n_splits <= len(unique):
        raise SplitError("n_splits must be in [2, number of unique groups]")
    values = None if stratify_values is None else np.asarray(stratify_values, dtype=float)
    if values is not None and (
        values.ndim != 1 or len(values) != len(groups) or not np.isfinite(values).all()
    ):
        raise SplitError("stratify_values must be finite and aligned with groups")
    rng = np.random.default_rng(seed)
    assignments: list[list[str]] = [[] for _ in range(n_splits)]
    if values is None:
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        for i, g in enumerate(shuffled):
            assignments[i % n_splits].append(str(g))
    else:
        medians = np.array([np.median(values[groups == g]) for g in unique])
        bins = min(max(1, n_bins), max(1, len(unique) // n_splits))
        cuts = (
            np.unique(np.quantile(medians, np.linspace(0, 1, bins + 1)[1:-1]))
            if bins > 1
            else np.asarray([])
        )
        labels = np.digitize(medians, cuts, right=True)
        fold_load = np.zeros(n_splits, dtype=int)
        for label in np.unique(labels):
            candidates = unique[labels == label].copy()
            rng.shuffle(candidates)
            for g in candidates:
                # Rotate with a seeded random tie order, assigning to the least populated fold first.
                min_load = int(fold_load.min())
                choices = np.flatnonzero(fold_load == min_load)
                fold = int(rng.choice(choices))
                assignments[fold].append(str(g))
                fold_load[fold] += 1
    result = []
    for i, test_groups in enumerate(assignments):
        test_mask = np.isin(groups, test_groups)
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if not len(train_idx) or not len(test_idx):
            raise SplitError("empty train/test fold; reduce n_splits")
        assert_group_disjoint(groups, {"train": train_idx, "test": test_idx})
        tr_range = (
            (float(values[train_idx].min()), float(values[train_idx].max()))
            if values is not None
            else None
        )
        te_range = (
            (float(values[test_idx].min()), float(values[test_idx].max()))
            if values is not None
            else None
        )
        result.append(
            GroupFold(
                i,
                train_idx,
                test_idx,
                tuple(sorted(set(groups[train_idx]))),
                tuple(sorted(set(groups[test_idx]))),
                tr_range,
                te_range,
            )
        )
    all_test = np.concatenate([fold.test_indices for fold in result])
    if sorted(all_test.tolist()) != list(range(len(groups))):
        raise RuntimeError("internal error: each row must occur in exactly one test fold")
    return result


def cross_validation_manifest(
    folds: list[GroupFold],
    *,
    dataset_id: str,
    target_name: str,
    seed: int,
    method: str = "group-k-fold",
) -> dict:
    return {
        "dataset_id": dataset_id,
        "target_name": target_name,
        "seed": int(seed),
        "method": method,
        "n_folds": len(folds),
        "folds": [f.to_dict() for f in folds],
        "leakage_rule": "unique group IDs are assigned to exactly one test fold",
        "caveat": "group disjointness is guaranteed; target interpolation support still needs inspection",
    }
