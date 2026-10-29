"""Small NumPy artifact store with schema and hash metadata.

For large scientific simulations, use a chunked format and institutional storage. This store is
intended for laptop-scale fixtures, not 512^3 volumes or petabyte surveys.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nufield.provenance import sha256_file, write_json


class NumpyArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, arrays: dict[str, np.ndarray], metadata: dict) -> dict:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("name must be a plain filename stem")
        clean = {}
        shapes = {}
        for key, value in arrays.items():
            arr = np.asarray(value)
            if arr.dtype == object:
                raise ValueError("object arrays are not permitted in artifact store")
            clean[key] = arr
            shapes[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        data_path = self.root / f"{name}.npz"
        np.savez_compressed(data_path, **clean)
        document = {
            "schema_version": "1.0",
            "artifact_name": name,
            "array_schema": shapes,
            "metadata": metadata,
            "sha256": sha256_file(data_path),
        }
        meta_path = self.root / f"{name}.metadata.json"
        write_json(meta_path, document)
        return {
            "data_path": str(data_path),
            "metadata_path": str(meta_path),
            "sha256": document["sha256"],
        }

    def load(self, name: str, verify_hash: bool = True) -> tuple[dict[str, np.ndarray], dict]:
        data_path = self.root / f"{name}.npz"
        meta_path = self.root / f"{name}.metadata.json"
        if not data_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"artifact `{name}` is incomplete")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if verify_hash and sha256_file(data_path) != meta["sha256"]:
            raise ValueError("artifact hash mismatch; file may have been altered")
        with np.load(data_path, allow_pickle=False) as container:
            arrays = {k: container[k] for k in container.files}
        for key, arr in arrays.items():
            expected = meta["array_schema"].get(key)
            if (
                not expected
                or expected["shape"] != list(arr.shape)
                or expected["dtype"] != str(arr.dtype)
            ):
                raise ValueError(f"schema mismatch for array {key}")
        return arrays, meta
