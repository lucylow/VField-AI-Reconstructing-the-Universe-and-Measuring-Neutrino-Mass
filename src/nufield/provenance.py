"""Experiment manifests and artifact hashing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOY_WARNING = (
    "TOY PHYSICS ONLY: outputs are for software validation, not cosmological measurement. "
    "The spectrum is phenomenological and is not a validated massive-neutrino model."
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "matplotlib", "PyYAML", "nufield-starter"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed-as-distribution"
    return versions


def build_manifest(
    *,
    config: dict[str, Any],
    seed: int,
    dataset_path: str | Path | None = None,
    git_revision: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe reproducibility manifest."""
    cfg = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physics_level": "toy",
        "scientific_warning": TOY_WARNING,
        "random_seed": int(seed),
        "config_sha256": hashlib.sha256(cfg).hexdigest(),
        "git_revision": git_revision or "unknown",
        "software_versions": package_versions(),
    }
    if dataset_path is not None and Path(dataset_path).exists():
        result["dataset"] = {"path": str(dataset_path), "sha256": sha256_file(dataset_path)}
    if extra:
        result["extra"] = extra
    return result


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
