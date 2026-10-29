"""Machine-readable acceptance gates for software and physics maturity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    passed: bool
    observed: str
    required: str
    severity: str = "blocking"

    def __post_init__(self):
        if self.severity not in {"blocking", "warning", "informational"}:
            raise ValueError("invalid gate severity")


def release_ready(gates: list[GateResult]) -> bool:
    """Return false when any blocking gate fails."""
    return all(g.passed or g.severity != "blocking" for g in gates)


def starter_gates() -> list[GateResult]:
    """Return explicit release requirements; physical gates are intentionally unmet in v0.1."""
    return [
        GateResult("tests", False, "not run", "all tests pass"),
        GateResult("data_leakage", False, "not evaluated", "frozen disjoint simulation IDs"),
        GateResult(
            "physical_forward_model", False, "toy spectrum", "validated neutrino-aware simulator"
        ),
        GateResult(
            "posterior_calibration", False, "not established", "independent SBC and PPC pass"
        ),
        GateResult(
            "external_validation", False, "not done", "reference simulation/likelihood comparison"
        ),
    ]
