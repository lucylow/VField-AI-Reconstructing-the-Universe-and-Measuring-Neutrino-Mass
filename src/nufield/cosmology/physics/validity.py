"""Validity labels and hard gates separating toy, linear and survey-level cosmology."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PhysicsLevel(str, Enum):
    TOY = "toy"
    HOMOGENEOUS_BACKGROUND = "homogeneous_background"
    LINEAR_BOLTZMANN = "linear_boltzmann"
    LINEAR_GAUSSIAN_FIELD = "linear_gaussian_field"
    N_BODY_SIMULATION = "n_body_simulation"
    SURVEY_ANALYSIS = "survey_analysis"


@dataclass(frozen=True, slots=True)
class PhysicsScope:
    level: PhysicsLevel
    neutrino_perturbations: bool
    nonlinear_gravity: bool
    survey_window: bool
    observational_noise: bool
    validated_against_reference: bool = False
    notes: tuple[str, ...] = ()

    def assert_allowed_for_claim(self, claim: str) -> None:
        """Reject claims requiring physics absent from the registered model scope."""
        normalized = claim.lower().replace("-", "_").replace(" ", "_")
        problems: list[str] = []
        if any(
            token in normalized
            for token in ("neutrino_mass_constraint", "mnu_constraint", "mass_measurement")
        ):
            if not self.neutrino_perturbations:
                problems.append(
                    "model lacks validated massive-neutrino perturbation/free-streaming treatment"
                )
            if self.level in (
                PhysicsLevel.TOY,
                PhysicsLevel.HOMOGENEOUS_BACKGROUND,
                PhysicsLevel.LINEAR_GAUSSIAN_FIELD,
            ):
                problems.append(
                    "model scope is insufficient for a cosmological neutrino-mass constraint"
                )
            if not self.validated_against_reference:
                problems.append("model has not been validated against an independent reference")
        if any(
            token in normalized
            for token in ("desi_result", "survey_measurement", "observational_constraint")
        ):
            if not self.survey_window:
                problems.append("model does not include a validated survey window/selection model")
            if not self.observational_noise:
                problems.append(
                    "model does not include relevant observational noise and systematics"
                )
        if problems:
            raise ValueError("claim is blocked by physics-scope gates: " + "; ".join(problems))


def default_scope_for_level(level: PhysicsLevel) -> PhysicsScope:
    """Create conservative default capabilities for each broad model category."""
    if level == PhysicsLevel.TOY:
        return PhysicsScope(
            level,
            False,
            False,
            False,
            False,
            False,
            ("Software testing only; parameter response may be arbitrary.",),
        )
    if level == PhysicsLevel.HOMOGENEOUS_BACKGROUND:
        return PhysicsScope(
            level,
            False,
            False,
            False,
            False,
            False,
            ("Background evolution only; no perturbation predictions.",),
        )
    if level == PhysicsLevel.LINEAR_BOLTZMANN:
        return PhysicsScope(
            level,
            True,
            False,
            False,
            False,
            False,
            ("Linear perturbations only; verify solver configuration and observable units.",),
        )
    if level == PhysicsLevel.LINEAR_GAUSSIAN_FIELD:
        return PhysicsScope(
            level,
            False,
            False,
            False,
            False,
            False,
            ("Gaussian-field generation does not model nonlinear evolution.",),
        )
    if level == PhysicsLevel.N_BODY_SIMULATION:
        return PhysicsScope(
            level,
            False,
            True,
            False,
            False,
            False,
            (
                "N-body status alone does not prove massive-neutrino dynamics are present; inspect the simulator.",
            ),
        )
    return PhysicsScope(
        level,
        False,
        False,
        False,
        False,
        False,
        (
            "Survey-level label is not proof of validated windows/noise; explicitly register each capability.",
        ),
    )


__all__ = ["PhysicsLevel", "PhysicsScope", "default_scope_for_level"]
