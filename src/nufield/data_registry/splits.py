"""Leakage-resistant group splits and mass-stratified sampling helpers."""

from __future__ import annotations

import hashlib

import numpy as np


class SplitError(ValueError):
    pass


def group_train_val_test_split(
    group_ids: np.ndarray,
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split by parent simulation/group ID, guaranteeing no group crosses split boundaries."""
    groups = np.asarray(group_ids).astype(str)
    if groups.ndim != 1 or len(groups) < 3:
        raise SplitError("group_ids must be 1-D with at least three rows")
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
        raise SplitError("fractions must be three positive values summing to 1")
    unique = np.unique(groups)
    if len(unique) < 3:
        raise SplitError("at least three independent groups are needed")
    rng = np.random.default_rng(seed)
    unique = unique.copy()
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, min(n - 2, int(round(n * fractions[0]))))
    n_val = max(1, min(n - n_train - 1, int(round(n * fractions[1]))))
    partitions = {
        "train": set(unique[:n_train]),
        "validation": set(unique[n_train : n_train + n_val]),
        "test": set(unique[n_train + n_val :]),
    }
    result = {key: np.flatnonzero(np.isin(groups, list(vals))) for key, vals in partitions.items()}
    assert_group_disjoint(groups, result)
    return result


def assert_group_disjoint(group_ids: np.ndarray, split_indices: dict[str, np.ndarray]) -> None:
    groups = np.asarray(group_ids).astype(str)
    names = list(split_indices)
    for i, a in enumerate(names):
        ga = set(groups[np.asarray(split_indices[a], dtype=int)].tolist())
        for b in names[i + 1 :]:
            gb = set(groups[np.asarray(split_indices[b], dtype=int)].tolist())
            overlap = ga & gb
            if overlap:
                raise SplitError(f"group leakage between {a} and {b}: {sorted(overlap)[:5]}")


def verify_parameter_support(
    theta: np.ndarray,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    parameter_names: list[str],
    name: str = "Mnu",
    require_test_interpolation: bool = True,
) -> dict:
    """Report support overlap; warn when test labels are out of train support or unseen discrete values."""
    values = np.asarray(theta, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(parameter_names):
        raise ValueError("theta/name shape mismatch")
    if name not in parameter_names:
        raise ValueError(f"missing parameter {name!r}")
    j = parameter_names.index(name)
    tr = values[np.asarray(train_indices, dtype=int), j]
    te = values[np.asarray(test_indices, dtype=int), j]
    report = {
        "parameter": name,
        "train_min": float(tr.min()),
        "train_max": float(tr.max()),
        "test_min": float(te.min()),
        "test_max": float(te.max()),
        "test_outside_train_range": bool(te.min() < tr.min() or te.max() > tr.max()),
        "n_train_unique": int(np.unique(tr).size),
        "n_test_unique": int(np.unique(te).size),
    }
    if require_test_interpolation and report["test_outside_train_range"]:
        report["warning"] = (
            "test includes parameter values outside training support; interpret as extrapolation"
        )
    else:
        report["warning"] = None
    return report


def split_fingerprint(simulation_ids: np.ndarray, indices: np.ndarray) -> str:
    vals = sorted(np.asarray(simulation_ids).astype(str)[np.asarray(indices, dtype=int)].tolist())
    payload = "\n".join(vals).encode()
    return hashlib.sha256(payload).hexdigest()


def stratified_group_train_val_test_split(
    group_ids: np.ndarray,
    stratify_values: np.ndarray,
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    n_bins: int = 6,
    seed: int = 42,
    require_group_constant: bool = False,
) -> dict[str, np.ndarray]:
    """Group split with approximate label-bin balance for a continuous target such as Mnu.

    `group_ids` define indivisible parent simulations/families. By default the median target within
    each group is used for approximate stratification, so paired-phase families can contain multiple
    parameter values. Set `require_group_constant=True` when every group should have one fixed target
    label. If groups span many parameter values, inspect the realized per-split support explicitly.

    Bins with >=3 groups try to place a group in each split. This is a split-design aid; inspect the
    realized parameter-support report after assignment because small/degenerate suites can make
    balanced bins impossible.
    """
    groups = np.asarray(group_ids).astype(str)
    values = np.asarray(stratify_values, dtype=float)
    if groups.ndim != 1 or values.ndim != 1 or len(groups) != len(values):
        raise SplitError("groups and stratify_values must be aligned 1-D arrays")
    if not np.isfinite(values).all():
        raise SplitError("stratify_values must be finite")
    if n_bins < 1:
        raise SplitError("n_bins must be >= 1")
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
        raise SplitError("fractions must be positive and sum to 1")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise SplitError("at least three independent groups are required")
    summaries = []
    for group in unique_groups:
        vals = values[groups == group]
        if require_group_constant and np.ptp(vals) > max(
            1e-8, 1e-6 * max(1.0, abs(float(np.mean(vals))))
        ):
            raise SplitError(
                f"stratification label is not constant within group {group!r}; choose the correct split group"
            )
        summaries.append(float(np.median(vals)))
    group_values = np.asarray(summaries)
    bins_eff = min(int(n_bins), max(1, len(unique_groups) // 3))
    quantiles = np.linspace(0.0, 1.0, bins_eff + 1)[1:-1]
    edges = np.unique(np.quantile(group_values, quantiles)) if len(quantiles) else np.asarray([])
    labels = np.digitize(group_values, edges, right=True)
    rng = np.random.default_rng(seed)
    assigned: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    names = list(assigned)
    for bin_id in np.unique(labels):
        group_bin = unique_groups[labels == bin_id].copy()
        rng.shuffle(group_bin)
        count = len(group_bin)
        # Largest-remainder allocation, with one per partition when this stratum is large enough.
        raw = np.asarray(fractions) * count
        alloc = np.floor(raw).astype(int)
        remainder = count - int(alloc.sum())
        for idx in np.argsort(-(raw - alloc))[:remainder]:
            alloc[idx] += 1
        if count >= 3:
            for idx in range(3):
                if alloc[idx] == 0:
                    donor = int(np.argmax(alloc))
                    if alloc[donor] > 1:
                        alloc[donor] -= 1
                        alloc[idx] += 1
        start = 0
        for name, num in zip(names, alloc):
            assigned[name].update(group_bin[start : start + num].tolist())
            start += num
    result = {name: np.flatnonzero(np.isin(groups, list(gset))) for name, gset in assigned.items()}
    assert_group_disjoint(groups, result)
    if any(len(result[k]) == 0 for k in result):
        raise SplitError(
            "stratified split generated empty partition; reduce bins or acquire more groups"
        )
    return result


def group_train_validation_split(
    group_ids: np.ndarray, *, validation_fraction: float = 0.12, seed: int = 42
) -> dict[str, np.ndarray]:
    """Two-way group split for inner early stopping, separate from outer model-selection validation.

    All views from a parent stay together. The function returns row indices into the supplied arrays.
    It is intended to partition the outer training split, keeping the outer validation and test sets
    independent from both model fitting and inner early stopping.
    """
    groups = np.asarray(group_ids).astype(str)
    if groups.ndim != 1 or len(groups) < 4:
        raise SplitError("inner split requires at least four rows")
    if not 0.05 <= validation_fraction <= 0.3:
        raise SplitError("validation_fraction must lie in [0.05, 0.3]")
    unique = np.unique(groups)
    if len(unique) < 4:
        raise SplitError("inner group split requires at least four independent parent groups")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_val = max(1, min(len(unique) - 2, int(round(len(unique) * validation_fraction))))
    val_groups = set(shuffled[:n_val].tolist())
    train_groups = set(shuffled[n_val:].tolist())
    result = {
        "train": np.flatnonzero(np.isin(groups, list(train_groups))),
        "validation": np.flatnonzero(np.isin(groups, list(val_groups))),
    }
    assert_group_disjoint(groups, result)
    if min(len(result["train"]), len(result["validation"])) < 2:
        raise SplitError(
            "inner split produced too few rows; acquire more independent groups or alter fraction"
        )
    return result
