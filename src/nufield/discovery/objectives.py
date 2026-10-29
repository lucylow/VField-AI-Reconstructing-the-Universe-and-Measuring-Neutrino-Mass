"""Multi-objective score functions for research algorithm candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateScore:
    reconstruction_error: float
    parameter_bias_sigma: float
    coverage_error: float
    runtime_seconds: float
    memory_mb: float
    tests_passed: bool

    def admissible(self, *, max_coverage_error: float = 0.08, max_bias_sigma: float = 0.5) -> bool:
        vals = [
            self.reconstruction_error,
            self.parameter_bias_sigma,
            self.coverage_error,
            self.runtime_seconds,
            self.memory_mb,
        ]
        if any(v < 0 or not (v < float("inf")) for v in vals):
            return False
        return (
            self.tests_passed
            and self.coverage_error <= max_coverage_error
            and self.parameter_bias_sigma <= max_bias_sigma
        )


def scalar_score(
    score: CandidateScore,
    *,
    error_weight: float = 1.0,
    bias_weight: float = 2.0,
    coverage_weight: float = 4.0,
    runtime_weight: float = 0.01,
    memory_weight: float = 0.001,
) -> float:
    """A transparent selection score; gate admissibility separately before comparing scores."""
    return (
        error_weight * score.reconstruction_error
        + bias_weight * score.parameter_bias_sigma
        + coverage_weight * score.coverage_error
        + runtime_weight * score.runtime_seconds
        + memory_weight * score.memory_mb
    )
