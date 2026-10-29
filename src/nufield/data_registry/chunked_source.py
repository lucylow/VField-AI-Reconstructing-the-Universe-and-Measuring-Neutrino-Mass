"""Explicit, chunk-oriented readers for large HDF5 and Zarr simulation products.

This is a storage abstraction only. It does not convert native Quijote, Abacus, Gadget, DESI, or
CAMELS products unless the caller supplies an exact, documented mapping of array paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

StorageKind = Literal["hdf5", "zarr"]


@dataclass(frozen=True)
class ArrayContract:
    key: str
    expected_rank: int | None = None
    expected_tail_shape: tuple[int, ...] | None = None
    dtype_kind: str = "numeric"
    row_axis: int = 0
    allow_scalar_for_single: bool = False

    def validate(self, shape: tuple[int, ...], dtype: np.dtype) -> list[str]:
        issues = []
        if self.expected_rank is not None and len(shape) != self.expected_rank:
            issues.append(f"{self.key}: expected rank {self.expected_rank}, got {len(shape)}")
        if self.expected_tail_shape is not None and tuple(
            shape[-len(self.expected_tail_shape) :]
        ) != tuple(self.expected_tail_shape):
            issues.append(
                f"{self.key}: trailing shape expected {self.expected_tail_shape}, got {shape}"
            )
        if self.dtype_kind == "numeric" and dtype.kind not in "iuf":
            issues.append(f"{self.key}: expected numeric dtype, got {dtype}")
        if self.dtype_kind == "string" and dtype.kind not in "SUO":
            issues.append(f"{self.key}: expected string-like dtype, got {dtype}")
        if self.row_axis < 0 or (shape and self.row_axis >= len(shape)):
            issues.append(f"{self.key}: invalid row_axis {self.row_axis}")
        return issues


class ChunkedArraySource:
    """Context manager exposing datasets without loading entire arrays into memory."""

    def __init__(self, path: str | Path, *, kind: StorageKind = "hdf5"):
        self.path = Path(path)
        self.kind = kind
        self._handle = None

    def __enter__(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.kind == "hdf5":
            try:
                import h5py
            except ImportError as exc:
                raise RuntimeError("Install nufield-starter[data] for HDF5") from exc
            self._handle = h5py.File(self.path, "r")
        elif self.kind == "zarr":
            try:
                import zarr
            except ImportError as exc:
                raise RuntimeError("Install nufield-starter[data] for Zarr") from exc
            self._handle = zarr.open_group(str(self.path), mode="r")
        else:
            raise ValueError(f"unsupported storage kind {self.kind}")
        return self

    def __exit__(self, exc_type, exc, tb):
        close = getattr(self._handle, "close", None)
        if callable(close):
            close()
        self._handle = None
        return False

    def metadata(self, key: str) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("use source inside `with` block")
        try:
            node = self._handle[key]
        except Exception as exc:
            raise KeyError(f"array key {key!r} not found in {self.path}") from exc
        if not hasattr(node, "shape") or not hasattr(node, "dtype"):
            raise TypeError(f"{key!r} is not an array")
        return {
            "key": key,
            "shape": tuple(int(x) for x in node.shape),
            "dtype": str(node.dtype),
            "chunks": getattr(node, "chunks", None),
            "attrs": dict(getattr(node, "attrs", {})),
        }

    def read_rows(
        self, key: str, start: int, stop: int, *, row_axis: int = 0, dtype: str | None = None
    ) -> np.ndarray:
        meta = self.metadata(key)
        shape = meta["shape"]
        if not shape:
            raise ValueError("cannot row-slice scalar data")
        if row_axis != 0:
            raise NotImplementedError("only row_axis=0 is supported in the streaming iterator")
        if start < 0 or stop < start or stop > shape[0]:
            raise IndexError(f"invalid row slice [{start}:{stop}] for {shape}")
        array = np.asarray(self._handle[key][start:stop])
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def iter_batches(
        self,
        key: str,
        *,
        batch_size: int,
        indices: np.ndarray | None = None,
        dtype: str | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(row_indices, values)`; shuffled indices are sorted only inside each read group."""
        meta = self.metadata(key)
        n = meta["shape"][0] if meta["shape"] else 1
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        idx = (
            np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        )
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= n):
            raise ValueError("indices must be a valid 1-D row array")
        # Batch indices may be shuffled. To keep results in requested order without random HDF5
        # access overhead, read a contiguous cover when compact; otherwise perform explicit gather.
        for start in range(0, len(idx), batch_size):
            request = idx[start : start + batch_size]
            order = np.argsort(request, kind="stable")
            sorted_idx = request[order]
            if len(sorted_idx) and sorted_idx[-1] - sorted_idx[0] + 1 <= max(batch_size * 3, 32):
                slab = self._handle[key][int(sorted_idx[0]) : int(sorted_idx[-1]) + 1]
                vals = np.asarray(slab)[sorted_idx - int(sorted_idx[0])]
            else:
                vals = np.stack([np.asarray(self._handle[key][int(i)]) for i in sorted_idx], axis=0)
            vals = vals[np.argsort(order, kind="stable")]
            if dtype is not None:
                vals = vals.astype(dtype, copy=False)
            yield request.copy(), vals


def validate_aligned_arrays(
    source: ChunkedArraySource, contracts: list[ArrayContract]
) -> dict[str, Any]:
    """Check exact mapped arrays and common row counts without reading their contents."""
    if not contracts:
        raise ValueError("at least one array contract is required")
    reports = []
    row_counts = []
    for contract in contracts:
        meta = source.metadata(contract.key)
        issues = contract.validate(meta["shape"], np.dtype(meta["dtype"]))
        if meta["shape"]:
            row_counts.append((contract.key, meta["shape"][contract.row_axis]))
        reports.append({**meta, "issues": issues})
    values = {n for _, n in row_counts}
    alignment_error = len(values) > 1
    return {
        "path": str(source.path.resolve()),
        "kind": source.kind,
        "arrays": reports,
        "row_counts": dict(row_counts),
        "row_aligned": not alignment_error,
        "errors": [f"row counts differ: {dict(row_counts)}"] if alignment_error else [],
        "scientific_approval": False,
    }
