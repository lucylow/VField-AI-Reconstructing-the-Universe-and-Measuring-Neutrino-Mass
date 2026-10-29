"""Provenance manifests and atomic run artifacts for AI experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    """Stable JSON encoding suitable for configuration fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def fingerprint_json(value: Any) -> str:
    """SHA-256 fingerprint of a canonical JSON-serializable object."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fingerprint_array(value: Any, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash array metadata and contiguous data bytes without serializing through pickle."""
    arr = np.asarray(value)
    if arr.dtype.hasobject:
        raise TypeError("object arrays cannot be fingerprinted safely")
    digest = hashlib.sha256()
    digest.update(canonical_json({"shape": arr.shape, "dtype": arr.dtype.str}).encode("utf-8"))
    contiguous = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
    for start in range(0, contiguous.size, chunk_bytes):
        digest.update(memoryview(contiguous[start : start + chunk_bytes]))
    return digest.hexdigest()


def environment_report() -> dict[str, Any]:
    """Capture runtime versions while keeping optional dependencies optional."""
    packages = {}
    for package in (
        "nufield-starter",
        "numpy",
        "scipy",
        "torch",
        "sbi",
        "camb",
        "classy",
        "astropy",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def atomic_json_write(path: str | Path, value: Any) -> Path:
    """Write JSON atomically; creates parent directories and refuses NaN/Infinity."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


@dataclass
class AIExperimentManifest:
    """Self-describing record for one training or inference experiment."""

    experiment_name: str
    task: str
    model_name: str
    seed: int
    config: dict[str, Any]
    data_fingerprints: dict[str, str]
    split_manifest: dict[str, Any]
    parameter_names: list[str] = field(default_factory=list)
    parameter_units: dict[str, str] = field(default_factory=dict)
    observation_convention: str = "unspecified"
    simulator_status: str = "unknown"
    code_revision: str | None = None
    run_id: str | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    environment: dict[str, Any] = field(default_factory=environment_report)

    def validate(self) -> list[str]:
        errors = []
        if not self.experiment_name.strip() or not self.model_name.strip():
            errors.append("experiment_name and model_name are required")
        if not self.data_fingerprints:
            errors.append("at least one data fingerprint is required")
        if not self.split_manifest:
            errors.append("split manifest is required")
        if self.parameter_names and len(set(self.parameter_names)) != len(self.parameter_names):
            errors.append("parameter_names must be unique")
        missing_units = set(self.parameter_names) - set(self.parameter_units)
        if missing_units:
            errors.append(f"parameter units missing for: {sorted(missing_units)}")
        if self.simulator_status not in {"toy", "prototype", "validated", "unknown"}:
            errors.append("simulator_status must be toy/prototype/validated/unknown")
        return errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["manifest_fingerprint"] = fingerprint_json(
            {
                k: v
                for k, v in result.items()
                if k not in {"created_utc", "environment", "manifest_fingerprint"}
            }
        )
        result["validation_errors"] = self.validate()
        result["scientific_claim_eligible"] = (
            self.simulator_status == "validated" and not self.validate()
        )
        result["interpretation"] = (
            "A valid manifest documents an experiment; it does not validate the physics."
        )
        return result

    def save(self, path: str | Path) -> Path:
        return atomic_json_write(path, self.to_dict())


def verify_fingerprint(value: Any, expected: str) -> bool:
    """Return whether a numerical array matches its stored SHA-256 fingerprint."""
    if len(expected) != 64:
        raise ValueError("expected fingerprint must be a SHA-256 hex digest")
    return fingerprint_array(value) == expected.lower()
