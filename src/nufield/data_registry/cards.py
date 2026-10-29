"""Versioned dataset cards and lightweight serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetCard:
    """Metadata contract for a single local data product or derived dataset."""

    dataset_id: str
    source_id: str
    title: str
    release: str
    source_url: str
    license_name: str
    local_path: str
    storage_format: str
    observable_kind: str
    observable_key: str
    parameter_key: str
    parameter_names: list[str]
    neutrino_mass_parameter: str = "Mnu"
    mass_range_ev: list[float] | None = None
    neutrino_mass_varies: bool | None = None
    n_realizations: int | None = None
    redshift: float | None = None
    box_size_mpc_h: float | None = None
    field_resolution: list[int] | None = None
    simulation_code: str | None = None
    initial_condition_method: str | None = None
    density_definition: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    split_group_keys: list[str] = field(
        default_factory=lambda: ["simulation_id", "seed", "phase_id"]
    )
    checksum_algorithm: str = "sha256"
    checksum: str | None = None
    citation: str = ""
    preprocessing: list[str] = field(default_factory=list)
    known_limitations: list[str] = field(default_factory=list)
    status: str = "unverified"
    schema_version: str = "1.0"

    def validate(self) -> list[str]:
        errors = []
        if not self.dataset_id.strip():
            errors.append("dataset_id must be non-empty")
        if not self.source_id.strip():
            errors.append("source_id must be non-empty")
        if self.storage_format.lower() not in {
            "npz",
            "npy",
            "hdf5",
            "h5",
            "zarr",
            "fits",
            "parquet",
            "csv",
        }:
            errors.append("storage_format must be one of npz/npy/hdf5/h5/zarr/fits/parquet/csv")
        if not self.parameter_names:
            errors.append("parameter_names must list at least one inferred/simulated parameter")
        if self.mass_range_ev is not None:
            if (
                len(self.mass_range_ev) != 2
                or self.mass_range_ev[0] < 0
                or self.mass_range_ev[1] < self.mass_range_ev[0]
            ):
                errors.append(
                    "mass_range_ev must be [minimum, maximum] with 0 <= minimum <= maximum"
                )
        if self.neutrino_mass_varies is True and self.mass_range_ev is None:
            errors.append("a varying neutrino mass requires a declared mass_range_ev")
        if self.n_realizations is not None and self.n_realizations <= 0:
            errors.append("n_realizations must be positive when declared")
        if self.status not in {
            "unverified",
            "metadata_checked",
            "arrays_checked",
            "simulation_validated",
            "research_approved",
        }:
            errors.append("unsupported status value")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_dataset_card(card: DatasetCard, path: str | Path) -> Path:
    errors = card.validate()
    if errors:
        raise ValueError("Invalid dataset card: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = card.to_dict()
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_dataset_card(path: str | Path) -> DatasetCard:
    path = Path(path)
    raw = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.suffix.lower() in {".yaml", ".yml"}
        else json.loads(path.read_text(encoding="utf-8"))
    )
    if not isinstance(raw, dict):
        raise ValueError("dataset card root must be a mapping")
    allowed = set(DatasetCard.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown dataset-card fields: {sorted(unknown)}")
    card = DatasetCard(**raw)
    errors = card.validate()
    if errors:
        raise ValueError("Invalid dataset card: " + "; ".join(errors))
    return card
