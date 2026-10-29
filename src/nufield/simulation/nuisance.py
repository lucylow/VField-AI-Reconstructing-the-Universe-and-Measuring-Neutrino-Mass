"""Nuisance-parameter draws and controlled perturbations."""

from __future__ import annotations

import numpy as np


def draw_toy_nuisance(rng: np.random.Generator) -> dict[str, float]:
    """Draw generic nuisance amplitudes for controlled toy stress tests."""
    return {
        "selection_dropout": float(rng.uniform(0.0, 0.15)),
        "additive_noise": float(rng.uniform(0.0, 0.08)),
        "bias_miscalibration": float(rng.normal(0.0, 0.05)),
        "los_shift_dispersion": float(rng.uniform(0.0, 2.0)),
    }


def apply_bias_miscalibration(field: np.ndarray, fractional_shift: float) -> np.ndarray:
    x = np.asarray(field, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(fractional_shift) or abs(fractional_shift) > 1:
        raise ValueError("field must be finite and |fractional_shift| <= 1")
    return (x * (1.0 + fractional_shift)).astype(np.float32)


def nuisance_summary(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("records cannot be empty")
    keys = set(records[0])
    if any(set(r) != keys for r in records):
        raise ValueError("all nuisance records must have the same keys")
    result = {}
    for key in sorted(keys):
        arr = np.asarray([r[key] for r in records], dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite nuisance value for {key}")
        result[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return result
