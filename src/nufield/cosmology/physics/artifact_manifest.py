"""Scientific artifact manifest for cosmology runs and model checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.validity import PhysicsScope


@dataclass(frozen=True, slots=True)
class PhysicsArtifactManifest:
    artifact_name: str
    created_utc: str
    code_version: str
    python_version: str
    backend_name: str
    backend_version: str
    parameters: dict[str, Any]
    output_units: dict[str, str]
    dataset_ids: tuple[str, ...]
    data_fingerprints: tuple[str, ...]
    scope: dict[str, Any]
    random_seeds: tuple[int, ...]
    configuration_hash: str
    warnings: tuple[str, ...]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file hash for data/product provenance without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_configuration_hash(mapping: dict[str, Any]) -> str:
    """Hash a deterministic JSON encoding of a configuration mapping."""
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    artifact_name: str,
    params: PhysicalCosmology,
    scope: PhysicsScope,
    *,
    backend_name: str,
    backend_version: str,
    output_units: dict[str, str],
    dataset_ids: tuple[str, ...] = (),
    data_fingerprints: tuple[str, ...] = (),
    random_seeds: tuple[int, ...] = (),
    code_version: str = "0.3.0-dev",
    python_version: str = "unknown",
    extra_configuration: dict[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
) -> PhysicsArtifactManifest:
    """Create a detailed manifest while preserving the provided model physics scope."""
    from nufield.cosmology.physics.theory_summary import physics_scope_record

    if not artifact_name or not backend_name or not backend_version:
        raise ValueError("artifact_name/backend_name/backend_version must be nonempty")
    if len(dataset_ids) != len(data_fingerprints) and data_fingerprints:
        raise ValueError("if fingerprints are provided, require one per dataset ID")
    cfg = {
        "parameters": params.to_dict(),
        "backend_name": backend_name,
        "backend_version": backend_version,
        "output_units": output_units,
        "datasets": dataset_ids,
        "extra": extra_configuration or {},
    }
    return PhysicsArtifactManifest(
        artifact_name,
        datetime.now(timezone.utc).isoformat(),
        code_version,
        python_version,
        backend_name,
        backend_version,
        params.to_dict(),
        dict(output_units),
        tuple(dataset_ids),
        tuple(data_fingerprints),
        physics_scope_record(scope),
        tuple(int(s) for s in random_seeds),
        canonical_configuration_hash(cfg),
        tuple(warnings),
    )


def write_manifest(path: str | Path, manifest: PhysicsArtifactManifest) -> Path:
    """Write the manifest as sorted JSON, creating its parent directory if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True, default=str) + "\n")
    return out


__all__ = [
    "PhysicsArtifactManifest",
    "sha256_file",
    "canonical_configuration_hash",
    "build_manifest",
    "write_manifest",
]
