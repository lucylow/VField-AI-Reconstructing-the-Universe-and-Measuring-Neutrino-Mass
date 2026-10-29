"""Batch iterators for memory-bounded array processing."""

from __future__ import annotations

import numpy as np


def batch_indices(
    n_rows: int, batch_size: int, shuffle: bool = False, seed: int = 0
) -> list[np.ndarray]:
    if n_rows < 1 or batch_size < 1:
        raise ValueError("n_rows and batch_size must be positive")
    idx = np.arange(n_rows)
    if shuffle:
        idx = np.random.default_rng(seed).permutation(idx)
    return [idx[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def iter_batches(
    array: np.ndarray,
    batch_size: int,
    *,
    indices: np.ndarray | None = None,
    shuffle: bool = False,
    seed: int = 0,
):
    """Yield `(indices, rows)` batches without copying the whole array in advance."""
    x = np.asarray(array)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if x.ndim < 1:
        raise ValueError("array must have a row dimension")
    if indices is None:
        idx = np.arange(len(x))
    else:
        idx = np.asarray(indices, dtype=int)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= len(x)):
            raise ValueError("indices invalid")
    if shuffle:
        idx = np.random.default_rng(seed).permutation(idx)
    for start in range(0, len(idx), batch_size):
        batch = idx[start : start + batch_size]
        yield batch, x[batch]
