"""Inference metrics and interval diagnostics."""

from __future__ import annotations

import numpy as np


def mean_bias(samples_or_estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = np.asarray(samples_or_estimates, dtype=np.float64)
    true = np.asarray(truth, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("predictions and truth must have equal shapes")
    return np.mean(pred - true, axis=0)


def median_absolute_error(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred, true = np.asarray(estimates), np.asarray(truth)
    if pred.shape != true.shape:
        raise ValueError("estimates and truth must have equal shapes")
    return np.median(np.abs(pred - true), axis=0)


def central_interval(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim < 2 or not 0 < level < 1:
        raise ValueError("samples must have shape (n_samples, n_parameters) and level in (0,1)")
    alpha = (1.0 - level) / 2.0
    return np.quantile(s, alpha, axis=0), np.quantile(s, 1.0 - alpha, axis=0)


def interval_contains(samples: np.ndarray, truth: np.ndarray, level: float) -> np.ndarray:
    lo, hi = central_interval(samples, level)
    t = np.asarray(truth)
    return (t >= lo) & (t <= hi)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be a finite non-negative vector with positive sum")
    w = w / w.sum()
    return float(1.0 / np.sum(w * w))
