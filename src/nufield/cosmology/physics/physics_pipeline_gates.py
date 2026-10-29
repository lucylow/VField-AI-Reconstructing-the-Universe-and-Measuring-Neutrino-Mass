"""Auditable go/no-go gates for a cosmological inference campaign.

The gate engine does not manufacture evidence. It accepts structured results from physical validation,
data contracts, solver comparisons, unit audits, covariance checks and posterior calibration. Missing
checks are recorded as blockers, and a research result is not labeled publication-ready merely because
individual modules execute successfully.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GateEvidence:
    name: str
    passed: bool
    severity: str
    evidence_source: str
    summary: str
    metric: float | int | str | None = None
    threshold: float | int | str | None = None
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.summary.strip():
            raise ValueError("gate name and summary are required")
        if self.severity not in {"info", "warning", "error", "critical"}:
            raise ValueError("severity must be info, warning, error or critical")


@dataclass(frozen=True, slots=True)
class PhysicsGateReport:
    schema_version: str
    created_utc: str
    campaign: str
    status: str
    evidence: tuple[GateEvidence, ...]
    missing_required_checks: tuple[str, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    scope: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "campaign": self.campaign,
            "status": self.status,
            "evidence": [asdict(x) for x in self.evidence],
            "missing_required_checks": list(self.missing_required_checks),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "scope": list(self.scope),
        }


def assess_physics_campaign(
    campaign: str,
    evidence: Sequence[GateEvidence],
    *,
    required_checks: Sequence[str] = (
        "physics_equations",
        "solver_reference",
        "dataset_contract",
        "unit_conventions",
        "covariance_validation",
        "posterior_calibration",
        "heldout_systematics",
        "provenance",
    ),
    scope: Sequence[str] = (),
) -> PhysicsGateReport:
    """Build status from exact gate evidence; all required names need one passing evidence record.

    Severity governs negative findings: a failed warning is reported but does not automatically block;
    failed error/critical gates do. A missing required check always blocks. Duplicate names are rejected
    to prevent ambiguous contradictory status records.
    """
    if not campaign.strip():
        raise ValueError("campaign name is required")
    names = [e.name for e in evidence]
    if len(names) != len(set(names)):
        raise ValueError("gate evidence names must be unique")
    required = tuple(str(x) for x in required_checks)
    if len(set(required)) != len(required):
        raise ValueError("required check names must be unique")
    missing = tuple(
        name
        for name in required
        if name not in names or not next(e for e in evidence if e.name == name).passed
    )
    blockers: list[str] = []
    warnings: list[str] = []
    for item in evidence:
        if item.blockers:
            blockers.extend(f"{item.name}: {msg}" for msg in item.blockers)
        if not item.passed and item.severity in {"error", "critical"}:
            blockers.append(f"{item.name}: {item.summary}")
        elif not item.passed:
            warnings.append(f"{item.name}: {item.summary}")
    for name in required:
        if name not in names:
            blockers.append(f"missing required check: {name}")
        elif not next(e for e in evidence if e.name == name).passed:
            blockers.append(f"required check did not pass: {name}")
    # Preserve insertion order while deduplicating repeated blockers.
    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    if blockers:
        status = "blocked"
    elif warnings:
        status = "conditional"
    else:
        status = "ready_for_scientific_review"
    return PhysicsGateReport(
        "1.0",
        datetime.now(timezone.utc).isoformat(),
        campaign,
        status,
        tuple(evidence),
        missing,
        tuple(blockers),
        tuple(warnings),
        tuple(scope),
    )


def standard_evidence_from_mapping(
    results: Mapping[str, Mapping[str, object]],
) -> tuple[GateEvidence, ...]:
    """Convert simple report mappings to typed gates; each item must explicitly contain `passed`.

    This helper intentionally does not infer pass status from a numeric metric or a truthy string.
    The caller must review the upstream report and provide the threshold interpretation.
    """
    out = []
    for name, value in results.items():
        if "passed" not in value or not isinstance(value["passed"], bool):
            raise ValueError(f"result {name!r} must contain explicit boolean 'passed'")
        out.append(
            GateEvidence(
                str(name),
                bool(value["passed"]),
                str(value.get("severity", "error")),
                str(value.get("evidence_source", "unspecified")),
                str(value.get("summary", "no summary supplied")),
                value.get("metric"),
                value.get("threshold"),
                tuple(str(x) for x in value.get("blockers", ())),
            )
        )
    return tuple(out)


def write_physics_gate_report(path: str | Path, report: PhysicsGateReport) -> Path:
    """Write a sorted JSON release-gate report."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return out


__all__ = [
    "GateEvidence",
    "PhysicsGateReport",
    "assess_physics_campaign",
    "standard_evidence_from_mapping",
    "write_physics_gate_report",
]
