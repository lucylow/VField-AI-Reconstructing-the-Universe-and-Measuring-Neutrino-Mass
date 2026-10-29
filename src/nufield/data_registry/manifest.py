"""Immutable-ish source-file hashing and derived-artifact manifest generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from nufield.data_registry.cards import DatasetCard


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_files(paths: Iterable[str | Path], *, hash_files: bool = True) -> list[dict]:
    rows = []
    for item in paths:
        path = Path(item).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        row = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        row["sha256"] = sha256_file(path) if hash_files else None
        rows.append(row)
    return rows


def build_data_manifest(
    card: DatasetCard,
    source_files: Iterable[str | Path],
    *,
    derived_files: Iterable[str | Path] = (),
    transform_config: dict | None = None,
    git_commit: str | None = None,
    hash_files: bool = True,
) -> dict:
    errors = card.validate()
    if errors:
        raise ValueError("invalid dataset card: " + "; ".join(errors))
    return {
        "manifest_schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_card": card.to_dict(),
        "source_files": inventory_files(source_files, hash_files=hash_files),
        "derived_files": inventory_files(derived_files, hash_files=hash_files),
        "transform_config": transform_config or {},
        "git_commit": git_commit,
        "hash_algorithm": "sha256" if hash_files else None,
        "warning": "A manifest establishes provenance traceability, not physics validity, licensing clearance, or statistical independence.",
    }


def write_data_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
