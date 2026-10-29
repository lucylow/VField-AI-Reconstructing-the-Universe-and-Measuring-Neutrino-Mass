"""Small immutable transformation-lineage records for dataset/model reproducibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TransformRecord:
    transform_id: str
    code_version: str
    input_manifest_hash: str
    output_manifest_hash: str
    parameters: dict[str, Any] = field(default_factory=dict)
    fitted_on_split: str | None = None
    seed: int | None = None
    purpose: str = ""

    def validate(self) -> list[str]:
        errors = []
        for k in ("transform_id", "code_version", "input_manifest_hash", "output_manifest_hash"):
            if not getattr(self, k).strip():
                errors.append(f"{k} must be non-empty")
        if self.fitted_on_split and self.fitted_on_split.lower() in {
            "all",
            "full",
            "train+validation+test",
            "test",
        }:
            errors.append(
                "data-dependent transforms may not be fit on test data; declare training split only"
            )
        return errors

    def to_dict(self):
        return asdict(self)


def canonical_json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_lineage_manifest(
    *,
    dataset_card: dict[str, Any],
    transforms: list[TransformRecord],
    split_manifest: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    errors = [f"transform {t.transform_id}: {e}" for t in transforms for e in t.validate()]
    if errors:
        raise ValueError("invalid lineage: " + "; ".join(errors))
    if model_config.get("uses_test_during_training") is True:
        raise ValueError("model config declares test-set use during training")
    payload = {
        "schema_version": "1.0",
        "dataset_card": dataset_card,
        "transforms": [t.to_dict() for t in transforms],
        "split_manifest": split_manifest,
        "model_config": model_config,
        "rules": [
            "split before data-dependent preprocessing",
            "fit scalers on train partition only",
            "the final test set is accessed only after model/config selection",
            "source data is not redistributed by this manifest",
        ],
    }
    payload["manifest_hash"] = canonical_json_hash(payload)
    return payload
