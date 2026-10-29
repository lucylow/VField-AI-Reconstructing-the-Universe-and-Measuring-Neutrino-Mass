"""Metadata-only and array-level NPZ profiler for explicit dataset selection.

The profiler lists schemas and hashes arrays, but it does not guess the meaning of any column. NPZ
files are archives of NPY members; opening a member with NumPy can materialize that member in memory.
Use HDF5/Zarr chunked tooling for huge volume arrays rather than assuming this profiler is streaming.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def profile_npz(
    path: str | Path, *, inspect_values: bool = False, max_summary_values: int = 2_000_000
) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(file)
    if file.suffix.lower() != ".npz":
        raise ValueError(
            "profile_npz only accepts .npz; inspect HDF5/Zarr with format-specific tools"
        )
    archive_hash = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            archive_hash.update(chunk)
    arrays = []
    with np.load(file, allow_pickle=False) as z:
        for key in sorted(z.files):
            arr = z[key]
            entry = {
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
                "numeric": bool(np.issubdtype(arr.dtype, np.number)),
                "value_summary": None,
            }
            if (
                inspect_values
                and np.issubdtype(arr.dtype, np.number)
                and arr.size <= max_summary_values
            ):
                values = np.asarray(arr, dtype=float)
                finite = np.isfinite(values)
                entry["value_summary"] = {
                    "finite_fraction": float(finite.mean()) if values.size else 1.0,
                    "min": float(np.nanmin(values)) if finite.any() else None,
                    "max": float(np.nanmax(values)) if finite.any() else None,
                    "mean": float(np.nanmean(values)) if finite.any() else None,
                    "std": float(np.nanstd(values)) if finite.any() else None,
                    "n_unique_sampled": int(
                        np.unique(values.ravel()[: min(values.size, 100_000)]).size
                    ),
                    "summary_note": "Unique count is based on the first 100k flattened values when array exceeds that size.",
                }
            arrays.append(entry)
    return {
        "schema_version": "1.0",
        "path": str(file.resolve()),
        "archive_bytes": file.stat().st_size,
        "archive_sha256": archive_hash.hexdigest(),
        "arrays": arrays,
        "limitations": [
            "Array keys and columns are not assigned physical meanings automatically.",
            "No hidden source metadata or units are inferred.",
            "NPZ members may load fully into memory; this is not a streaming profiler.",
        ],
        "next_step": "Map exact keys and parameter units from the dataset release documentation, then create a DatasetCard.",
    }
