"""Utilities for masks and modality availability."""

from __future__ import annotations

import numpy as np


def apply_boolean_mask(values: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    x, m = np.asarray(values), np.asarray(mask, dtype=bool)
    try:
        m = np.broadcast_to(m, x.shape)
    except ValueError as exc:
        raise ValueError("mask is not broadcastable to values") from exc
    return np.where(m, x, fill_value)


def masked_mean(values: np.ndarray, mask: np.ndarray, axis=None) -> np.ndarray:
    x, m = np.asarray(values, dtype=float), np.asarray(mask, dtype=bool)
    m = np.broadcast_to(m, x.shape)
    numerator = np.sum(np.where(m, x, 0.0), axis=axis)
    denominator = np.sum(m, axis=axis)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
