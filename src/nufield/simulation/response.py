"""Paired-realization response diagnostics for controlled toy parameter perturbations."""

from __future__ import annotations

import numpy as np


def paired_field_response(
    field_a: np.ndarray, field_b: np.ndarray, epsilon: float = 1e-8
) -> dict[str, float]:
    """Compare fields generated with nominally paired phases under two model configurations."""
    a, b = np.asarray(field_a, float), np.asarray(field_b, float)
    if a.shape != b.shape or a.ndim != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("fields must be finite, same-shaped 3-D arrays")
    delta = b - a
    relative = np.sqrt(np.mean(delta**2)) / (np.sqrt(np.mean(a**2)) + epsilon)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
    return {
        "mean_difference": float(delta.mean()),
        "rms_difference": float(np.sqrt(np.mean(delta**2))),
        "relative_rms_difference": float(relative),
        "pearson_correlation": float(corr),
    }


def finite_difference_response(
    value_minus: np.ndarray, value_plus: np.ndarray, parameter_minus: float, parameter_plus: float
) -> np.ndarray:
    a, b = np.asarray(value_minus, float), np.asarray(value_plus, float)
    delta = float(parameter_plus - parameter_minus)
    if a.shape != b.shape or delta == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("arrays must match, parameter difference non-zero, and arrays finite")
    return ((b - a) / delta).astype(np.float32)
