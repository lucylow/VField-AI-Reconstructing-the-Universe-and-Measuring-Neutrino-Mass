"""Leakage detection utilities for immutable train/validation/test splits."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def assert_disjoint_splits(train_ids, validation_ids, test_ids) -> None:
    groups = [set(map(str, g)) for g in (train_ids, validation_ids, test_ids)]
    if any(not g for g in groups):
        raise ValueError("all split groups must be non-empty")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("split leakage detected: IDs overlap")


def split_hash(train_ids, validation_ids, test_ids) -> str:
    """Hash sorted split IDs so final reports can bind to a frozen evaluation manifest."""
    payload = {
        "train": sorted(map(str, train_ids)),
        "validation": sorted(map(str, validation_ids)),
        "test": sorted(map(str, test_ids)),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def assert_rows_not_reused(ids: np.ndarray) -> None:
    arr = np.asarray(ids).astype(str)
    if len(np.unique(arr)) != len(arr):
        duplicated = sorted({item for item in arr if np.sum(arr == item) > 1})
        raise ValueError(f"duplicate realization IDs: {duplicated[:10]}")
