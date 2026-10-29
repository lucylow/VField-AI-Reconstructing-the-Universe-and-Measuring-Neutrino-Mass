"""Metadata-first HDF5/Zarr-like schema inspection helpers; never infer physical meaning.

The HDF5 profiler only inspects paths, shapes, chunking, compression, dtypes, and attributes. It does
not materialize multi-gigabyte arrays or infer that a dataset called `mnu` is in eV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def profile_hdf5(
    path: str | Path, *, max_objects: int = 5000, include_attributes: bool = True
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install nufield-starter[data] to inspect HDF5 files") from exc
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(file)
    if max_objects < 1:
        raise ValueError("max_objects must be positive")
    items: list[dict[str, Any]] = []
    truncated = False
    with h5py.File(file, "r") as h5:

        def visit(name: str, obj: Any) -> None:
            nonlocal truncated
            if len(items) >= max_objects:
                truncated = True
                return
            row: dict[str, Any] = {
                "path": "/" + name,
                "kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
            }
            if isinstance(obj, h5py.Dataset):
                row.update(
                    {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "nbytes_uncompressed": int(obj.size * obj.dtype.itemsize),
                        "chunks": list(obj.chunks) if obj.chunks else None,
                        "compression": obj.compression,
                        "compression_opts": obj.compression_opts,
                        "shuffle": bool(obj.shuffle),
                        "fletcher32": bool(obj.fletcher32),
                    }
                )
            if include_attributes:
                attrs = {}
                for key in list(obj.attrs.keys())[:100]:
                    val = obj.attrs[key]
                    if hasattr(val, "tolist"):
                        val = val.tolist()
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    if isinstance(val, (str, int, float, bool, list, tuple)) or val is None:
                        attrs[str(key)] = val
                    else:
                        attrs[str(key)] = str(val)
                row["attributes"] = attrs
            items.append(row)

        h5.visititems(visit)
        return {
            "schema_version": "1.0",
            "path": str(file.resolve()),
            "file_bytes": file.stat().st_size,
            "root_attributes": {
                str(k): (v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v))
                for k, v in h5.attrs.items()
            },
            "object_count_returned": len(items),
            "truncated": truncated,
            "objects": items,
            "interpretation_warning": "Metadata inventory only. Key names, units, cosmology, release, and neutrino-mass semantics require source-document verification.",
        }
