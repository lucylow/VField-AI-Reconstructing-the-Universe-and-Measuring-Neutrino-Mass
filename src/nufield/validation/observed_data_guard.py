"""Guard against accidentally applying toy-mode inference to claimed real-survey analysis."""

from __future__ import annotations

from nufield.exceptions import ScientificValidityError


def require_research_grade_physics(physics_level: str, intended_use: str) -> None:
    """Raise when a workflow attempts a real-data or scientific-claim use with toy physics."""
    use = intended_use.lower().strip()
    research_uses = {
        "real_data",
        "cosmological_constraint",
        "scientific_measurement",
        "publication_result",
    }
    if physics_level == "toy" and use in research_uses:
        raise ScientificValidityError(
            "Toy physics cannot be used for real-data inference or a physical cosmological constraint. "
            "Replace the forward model and pass independent validation gates first."
        )
    if physics_level not in {"toy", "approximate", "validated"}:
        raise ValueError("unknown physics_level")


def promote_physics_level(current: str, required_checks: dict[str, bool]) -> str:
    """Fail-closed promotion policy: all required checks must be present and true."""
    names = {
        "backend_validated",
        "reference_comparison",
        "split_frozen",
        "sbc_passed",
        "ppc_reviewed",
        "expert_review",
    }
    missing = names - set(required_checks)
    if missing:
        raise ScientificValidityError(f"missing required checks: {sorted(missing)}")
    failed = sorted(k for k in names if not required_checks[k])
    if failed:
        raise ScientificValidityError(f"cannot promote physics level; failed checks: {failed}")
    if current != "toy":
        raise ScientificValidityError(
            "this helper only documents the toy-to-validated gate; review intermediate levels"
        )
    return "validated"
