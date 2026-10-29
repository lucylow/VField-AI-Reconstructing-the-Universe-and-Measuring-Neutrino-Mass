"""Predeclared model benchmark policy that prevents unfair architecture comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkPolicy:
    dataset_id: str
    release: str
    split_hash: str
    test_group_hash: str
    target_names: tuple[str, ...]
    prior_low: tuple[float, ...]
    prior_high: tuple[float, ...]
    physics_level: str
    observation_model_id: str
    normalizer_fit_partition: str = "train_only"
    allowed_selection_partition: str = "validation_only"
    final_test_accesses: int = 1

    def validate(self) -> list[str]:
        errors = []
        for k in ("dataset_id", "release", "split_hash", "test_group_hash", "observation_model_id"):
            if not getattr(self, k).strip():
                errors.append(f"{k} is required")
        if not self.target_names or len(set(self.target_names)) != len(self.target_names):
            errors.append("target_names must be unique and non-empty")
        if len(self.prior_low) != len(self.target_names) or len(self.prior_high) != len(
            self.target_names
        ):
            errors.append("prior dimension mismatch")
        if any(lo >= hi for lo, hi in zip(self.prior_low, self.prior_high)):
            errors.append("prior bounds must satisfy low < high")
        if self.normalizer_fit_partition != "train_only":
            errors.append("normalizer must be fitted on training only")
        if self.allowed_selection_partition != "validation_only":
            errors.append("model selection must use validation only")
        if self.final_test_accesses != 1:
            errors.append(
                "the policy reserves one final test evaluation; do not repeatedly tune on test"
            )
        return errors

    def to_dict(self):
        return {
            **asdict(self),
            "scientific_approval": False,
            "interpretation": "This policy standardizes a comparison; it does not validate the simulator or inference model.",
        }


def compare_policies(
    a: BenchmarkPolicy, b: BenchmarkPolicy, *, strict_physics: bool = True
) -> dict[str, Any]:
    """Return policy comparability without looking at model performance metrics."""
    fields = (
        "dataset_id",
        "release",
        "split_hash",
        "test_group_hash",
        "target_names",
        "prior_low",
        "prior_high",
        "observation_model_id",
    )
    mismatches = {
        f: {"a": getattr(a, f), "b": getattr(b, f)}
        for f in fields
        if getattr(a, f) != getattr(b, f)
    }
    if strict_physics and a.physics_level != b.physics_level:
        mismatches["physics_level"] = {"a": a.physics_level, "b": b.physics_level}
    return {
        "comparable": not mismatches,
        "mismatches": mismatches,
        "note": "Only compare neural candidates under matched data, prior, observation model, split, and locked test groups.",
    }
