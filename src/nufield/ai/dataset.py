"""Safe numerical dataset persistence and provenance validation for AI experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nufield.ai.contracts import validate_numeric_array
from nufield.ai.experiment import atomic_json_write, fingerprint_array, fingerprint_json


@dataclass(frozen=True)
class SimulationDataset:
    observations: np.ndarray
    parameters: np.ndarray
    group_ids: np.ndarray
    observation_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    parameter_units: dict[str, str]
    metadata: dict[str, Any]

    def validate(self) -> list[str]:
        errors: list[str] = []
        try:
            validate_numeric_array("observations", self.observations, ndim=2)
            validate_numeric_array("parameters", self.parameters, ndim=2)
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
        if self.observations.ndim == 2 and self.parameters.ndim == 2:
            if self.observations.shape[0] != self.parameters.shape[0]:
                errors.append("observations and parameters row counts differ")
        if np.asarray(self.group_ids).reshape(-1).shape[0] != len(self.observations):
            errors.append("group_ids row count differs from observations")
        if (
            self.observations.ndim == 2
            and len(self.observation_names) != self.observations.shape[1]
        ):
            errors.append("observation_names width mismatch")
        if self.parameters.ndim == 2 and len(self.parameter_names) != self.parameters.shape[1]:
            errors.append("parameter_names width mismatch")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            errors.append("parameter_names must be unique")
        missing = set(self.parameter_names) - set(self.parameter_units)
        if missing:
            errors.append(f"parameter units missing: {sorted(missing)}")
        if str(self.metadata.get("simulator_status", "unknown")) not in {
            "toy",
            "prototype",
            "validated",
            "unknown",
        }:
            errors.append("metadata.simulator_status must be toy/prototype/validated/unknown")
        return errors

    def require_valid(self) -> SimulationDataset:
        errors = self.validate()
        if errors:
            raise ValueError("invalid simulation dataset: " + "; ".join(errors))
        return self

    def fingerprint(self) -> str:
        self.require_valid()
        return fingerprint_json(
            {
                "observations": fingerprint_array(self.observations),
                "parameters": fingerprint_array(self.parameters),
                "group_ids": fingerprint_array(self.group_ids.astype(str)),
                "observation_names": self.observation_names,
                "parameter_names": self.parameter_names,
                "parameter_units": self.parameter_units,
                "metadata": self.metadata,
            }
        )

    def save(self, path: str | Path) -> dict[str, Any]:
        """Save arrays in NPZ (allow_pickle=False compatible) and an adjacent JSON manifest."""
        self.require_valid()
        target = Path(path)
        if target.suffix.lower() != ".npz":
            raise ValueError("dataset array path must end with .npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            observations=np.asarray(self.observations, dtype=np.float32),
            parameters=np.asarray(self.parameters, dtype=np.float64),
            group_ids=np.asarray(self.group_ids).astype("U"),
        )
        manifest = {
            "schema_version": 1,
            "observations_shape": list(self.observations.shape),
            "parameters_shape": list(self.parameters.shape),
            "observation_names": list(self.observation_names),
            "parameter_names": list(self.parameter_names),
            "parameter_units": self.parameter_units,
            "metadata": self.metadata,
            "dataset_fingerprint": self.fingerprint(),
            "files": {"arrays": target.name},
            "warning": "NPZ contains arrays only; metadata and provenance are stored in this JSON manifest.",
        }
        manifest_path = target.with_suffix(target.suffix + ".json")
        atomic_json_write(manifest_path, manifest)
        return {
            "arrays_path": str(target),
            "manifest_path": str(manifest_path),
            "dataset_fingerprint": manifest["dataset_fingerprint"],
        }

    @classmethod
    def load(cls, path: str | Path, *, require_fingerprint: bool = True) -> SimulationDataset:
        target = Path(path)
        manifest_path = target.with_suffix(target.suffix + ".json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(target, allow_pickle=False) as data:
            obj = cls(
                data["observations"],
                data["parameters"],
                data["group_ids"],
                tuple(manifest["observation_names"]),
                tuple(manifest["parameter_names"]),
                dict(manifest["parameter_units"]),
                dict(manifest["metadata"]),
            )
        obj.require_valid()
        actual = obj.fingerprint()
        if require_fingerprint and actual != manifest.get("dataset_fingerprint"):
            raise ValueError("dataset fingerprint mismatch; arrays or manifest may have changed")
        return obj


def dataset_quality_report(dataset: SimulationDataset) -> dict[str, Any]:
    """Produce a deterministic report of shapes, groups, missing values and parameter support."""
    errors = dataset.validate()
    result: dict[str, Any] = {
        "valid": not errors,
        "errors": errors,
        "n_rows": int(len(dataset.observations)),
        "n_groups": int(len(np.unique(dataset.group_ids))),
        "observation_shape": list(dataset.observations.shape),
        "parameter_shape": list(dataset.parameters.shape),
        "simulator_status": dataset.metadata.get("simulator_status", "unknown"),
        "parameter_support": {
            name: {
                "min": float(dataset.parameters[:, j].min()),
                "max": float(dataset.parameters[:, j].max()),
                "mean": float(dataset.parameters[:, j].mean()),
                "std": float(dataset.parameters[:, j].std()),
            }
            for j, name in enumerate(dataset.parameter_names)
        }
        if dataset.parameters.ndim == 2
        else {},
        "interpretation": "Schema validity is not a statement about physical simulator validity.",
    }
    for name, arr in (("observations", dataset.observations), ("parameters", dataset.parameters)):
        arr = np.asarray(arr)
        result.setdefault("array_checks", {})[name] = {
            "finite_fraction": float(np.isfinite(arr).mean()),
            "nonfinite_count": int((~np.isfinite(arr)).sum()),
            "constant_columns": int(np.sum(np.nanstd(arr, axis=0) < 1e-12))
            if arr.ndim == 2
            else None,
            "fingerprint": fingerprint_array(arr),
        }
    return result
