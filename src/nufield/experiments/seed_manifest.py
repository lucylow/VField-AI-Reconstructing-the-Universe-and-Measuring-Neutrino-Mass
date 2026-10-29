"""Freeze and serialize simulation split and seed plans."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nufield.provenance import write_json
from nufield.validation.leakage import assert_disjoint_splits, split_hash


def create_seed_manifest(
    simulation_ids: np.ndarray, master_seed: int, train_ids, validation_ids, test_ids
) -> dict:
    ids = set(map(str, simulation_ids))
    groups = [list(map(str, g)) for g in (train_ids, validation_ids, test_ids)]
    assert_disjoint_splits(*groups)
    used = set().union(*map(set, groups))
    if used != ids:
        raise ValueError("split IDs must cover the dataset exactly once")
    return {
        "schema_version": "1.0",
        "master_seed": int(master_seed),
        "train_ids": groups[0],
        "validation_ids": groups[1],
        "test_ids": groups[2],
        "split_sha256": split_hash(*groups),
        "n_total": len(ids),
    }


def write_seed_manifest(path: str | Path, manifest: dict) -> None:
    write_json(path, manifest)
