"""Simple support and feature-domain shift diagnostics."""

from __future__ import annotations

import numpy as np


def nearest_neighbor_distances(train: np.ndarray, evaluation: np.ndarray) -> np.ndarray:
    """Compute each evaluation row's nearest train feature distance after train-only scaling."""
    x, y = np.asarray(train, float), np.asarray(evaluation, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or len(x) == 0:
        raise ValueError("invalid feature matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features must be finite")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    xs, ys = (x - mean) / scale, (y - mean) / scale
    # Chunk to avoid allocation proportional to N_train*N_eval*D for large datasets.
    output = np.empty(len(ys), dtype=float)
    chunk = max(1, min(256, len(ys)))
    for start in range(0, len(ys), chunk):
        delta = ys[start : start + chunk, None, :] - xs[None, :, :]
        output[start : start + chunk] = np.sqrt(np.min(np.sum(delta * delta, axis=-1), axis=1))
    return output


def domain_shift_report(train: np.ndarray, evaluation: np.ndarray) -> dict[str, float]:
    d = nearest_neighbor_distances(train, evaluation)
    return {
        "median_nearest_distance": float(np.median(d)),
        "p90_nearest_distance": float(np.quantile(d, 0.9)),
        "p99_nearest_distance": float(np.quantile(d, 0.99)),
        "fraction_above_train_p95": float(
            np.mean(d > np.quantile(nearest_neighbor_distances(train, train), 0.95))
        ),
    }
