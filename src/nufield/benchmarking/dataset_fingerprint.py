"""Fingerprint observations, labels and group assignments to make experimental comparison auditable."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def array_fingerprint(array: np.ndarray, *, chunk_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array))
    raw = memoryview(arr).cast("B")
    h = hashlib.sha256()
    for start in range(0, len(raw), chunk_bytes):
        h.update(raw[start : start + chunk_bytes])
    return {
        "sha256": h.hexdigest(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "nbytes": int(arr.nbytes),
    }


def experiment_fingerprint(
    *,
    dataset_id: str,
    release: str,
    observation: np.ndarray,
    theta: np.ndarray,
    simulation_ids: np.ndarray,
    group_ids: np.ndarray,
    parameter_names: list[str],
    preprocessing: dict,
    split_manifest: dict,
) -> dict[str, Any]:
    n = len(theta)
    if len(observation) != n or len(simulation_ids) != n or len(group_ids) != n:
        raise ValueError("all arrays must have the same row count")
    if theta.ndim != 2 or theta.shape[1] != len(parameter_names):
        raise ValueError("theta/parameter schema mismatch")
    payload = {
        "dataset_id": dataset_id,
        "release": release,
        "observation": array_fingerprint(observation),
        "theta": array_fingerprint(theta),
        "simulation_ids": array_fingerprint(np.asarray(simulation_ids).astype(str)),
        "group_ids": array_fingerprint(np.asarray(group_ids).astype(str)),
        "parameter_names": list(parameter_names),
        "preprocessing": preprocessing,
        "split_manifest": split_manifest,
    }
    payload["experiment_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    payload["warning"] = (
        "Fingerprints identify exact arrays/configuration; they do not establish physical validity or sample independence."
    )
    return payload
