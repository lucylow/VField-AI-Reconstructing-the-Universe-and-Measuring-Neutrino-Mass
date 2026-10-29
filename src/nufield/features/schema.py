"""Typed schema for feature vector definitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    description: str
    unit: str
    estimator: str
    preprocessing: str = "none"
    valid_range: str = "not specified"
    physics_level: str = "toy"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    version: str
    definitions: tuple[FeatureDefinition, ...]

    def __post_init__(self):
        names = [d.name for d in self.definitions]
        if not names or len(names) != len(set(names)):
            raise ValueError("feature schema must contain unique definitions")

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.definitions]

    def sha256(self) -> str:
        payload = {"version": self.version, "definitions": [asdict(d) for d in self.definitions]}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "definitions": [asdict(d) for d in self.definitions],
            "sha256": self.sha256(),
        }
