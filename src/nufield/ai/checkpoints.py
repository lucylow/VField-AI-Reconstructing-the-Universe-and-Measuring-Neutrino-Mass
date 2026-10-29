"""Checkpoint metadata, integrity checks and safe artifact layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nufield.ai.experiment import atomic_json_write, fingerprint_json


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file by streaming bytes to keep memory use bounded."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def create_checkpoint_manifest(
    checkpoint_path: str | Path,
    *,
    model_name: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    data_fingerprint: str,
    split_fingerprint: str,
    parameter_names: list[str],
    parameter_units: dict[str, str],
    best_epoch: int,
    best_validation_loss: float,
    simulator_status: str,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Create an immutable metadata sidecar for one selected checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    if not model_name or not data_fingerprint or not split_fingerprint:
        raise ValueError("model, data fingerprint and split fingerprint are required")
    if len(set(parameter_names)) != len(parameter_names) or set(parameter_names) - set(
        parameter_units
    ):
        raise ValueError("parameter names must be unique and all have units")
    if not np.isfinite(best_validation_loss) or best_epoch < 1:
        raise ValueError("best_epoch and validation loss are invalid")
    if simulator_status not in {"toy", "prototype", "validated", "unknown"}:
        raise ValueError("invalid simulator_status")
    record = {
        "schema_version": 1,
        "model_name": model_name,
        "model_config": model_config,
        "training_config": training_config,
        "checkpoint_path": path.name,
        "checkpoint_sha256": file_sha256(path),
        "data_fingerprint": data_fingerprint,
        "split_fingerprint": split_fingerprint,
        "parameter_names": parameter_names,
        "parameter_units": parameter_units,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "simulator_status": simulator_status,
        "code_revision": code_revision,
    }
    record["manifest_fingerprint"] = fingerprint_json(record)
    record["claim_eligible"] = simulator_status == "validated"
    record["warning"] = (
        "Checkpoint integrity does not establish model calibration or physical validity."
    )
    return record


def save_checkpoint_manifest(checkpoint_path: str | Path, **kwargs) -> Path:
    """Write the checkpoint manifest next to the checkpoint."""
    path = Path(checkpoint_path)
    record = create_checkpoint_manifest(path, **kwargs)
    return atomic_json_write(path.with_suffix(path.suffix + ".manifest.json"), record)


def verify_checkpoint_manifest(
    checkpoint_path: str | Path, manifest_path: str | Path | None = None
) -> dict[str, Any]:
    """Verify file hash and manifest self-consistency before model loading."""
    path = Path(checkpoint_path)
    sidecar = (
        Path(manifest_path) if manifest_path else path.with_suffix(path.suffix + ".manifest.json")
    )
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    actual = file_sha256(path)
    if record.get("checkpoint_sha256") != actual:
        raise ValueError("checkpoint checksum does not match manifest")
    claimed = record.get("manifest_fingerprint")
    body = {
        k: v
        for k, v in record.items()
        if k not in {"manifest_fingerprint", "claim_eligible", "warning"}
    }
    if claimed != fingerprint_json(body):
        raise ValueError("manifest fingerprint does not match manifest content")
    return {
        "valid": True,
        "checkpoint_sha256": actual,
        "model_name": record.get("model_name"),
        "simulator_status": record.get("simulator_status"),
        "claim_eligible": bool(record.get("claim_eligible", False)),
    }
