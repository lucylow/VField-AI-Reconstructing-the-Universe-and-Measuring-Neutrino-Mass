"""Guardrails for comparing model benchmark records under matched experimental conditions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

REQUIRED_MATCH_FIELDS = (
    "dataset_id",
    "dataset_release",
    "split_id",
    "test_fingerprint",
    "parameter_names",
    "prior_low",
    "prior_high",
    "physics_level",
)


@dataclass(frozen=True)
class ComparisonFinding:
    severity: str
    code: str
    message: str
    field: str | None = None
    values: dict[str, Any] | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class FairComparisonReport:
    n_models: int
    fair_comparison_possible: bool
    findings: list[ComparisonFinding]
    matched_fields: list[str]

    def to_dict(self):
        return {
            "n_models": self.n_models,
            "fair_comparison_possible": self.fair_comparison_possible,
            "findings": [f.to_dict() for f in self.findings],
            "matched_fields": self.matched_fields,
            "warning": "Metadata matching is necessary but not sufficient: also match simulator assumptions, preprocessing, compute budgets, and validation policy.",
        }


def audit_fair_comparison(
    records: list[dict[str, Any]],
    *,
    require_same_dataset: bool = True,
    require_same_split: bool = True,
    allow_physics_level_mix: bool = False,
) -> FairComparisonReport:
    """Verify declared experiment metadata before ranking model results.

    Result reports should explicitly include the dataset release, held-out simulation fingerprint,
    inference prior, target order, and physics level. Missing fields are treated as blockers because
    similarity cannot be inferred from model names or output metrics.
    """
    if len(records) < 2:
        raise ValueError("at least two model result records are needed for comparison")
    findings: list[ComparisonFinding] = []
    matched: list[str] = []
    for i, record in enumerate(records):
        for field in REQUIRED_MATCH_FIELDS:
            if field not in record or record[field] is None:
                findings.append(
                    ComparisonFinding(
                        "error",
                        "MISSING_MATCH_METADATA",
                        f"record {i} is missing required comparison field {field!r}",
                        field,
                        {"record_index": i},
                    )
                )
    available = [
        f for f in REQUIRED_MATCH_FIELDS if all(f in r and r[f] is not None for r in records)
    ]
    for field in available:
        if field in {"dataset_id", "dataset_release"} and not require_same_dataset:
            continue
        if field in {"split_id", "test_fingerprint"} and not require_same_split:
            continue
        if field == "physics_level" and allow_physics_level_mix:
            continue
        values = [record[field] for record in records]
        # Treat list-valued metadata as ordered scientific configuration, not as unordered sets.
        if any(value != values[0] for value in values[1:]):
            findings.append(
                ComparisonFinding(
                    "error",
                    "MISMATCHED_COMPARISON_CONDITION",
                    f"records have different values for {field!r}",
                    field,
                    {str(i): values[i] for i in range(len(values))},
                )
            )
        else:
            matched.append(field)
    ok = not any(f.severity == "error" for f in findings)
    return FairComparisonReport(len(records), ok, findings, matched)
