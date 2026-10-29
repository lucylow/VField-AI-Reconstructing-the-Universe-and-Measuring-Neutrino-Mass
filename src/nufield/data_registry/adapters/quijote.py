"""Quijote source metadata contracts and safety checks for research configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QuijoteProductSpec:
    suite_name: str
    product_name: str
    redshift: float
    field_kind: str
    resolution: int
    box_size_mpc_h: float
    parameter_names: tuple[str, ...]
    neutrino_mass_ev_min: float | None
    neutrino_mass_ev_max: float | None
    initialization_method: str
    phase_policy: str
    official_doc_url: str = "https://quijote-simulations.readthedocs.io/en/latest/"

    def to_dict(self) -> dict:
        return asdict(self)


def validate_quijote_spec(spec: QuijoteProductSpec, *, target_parameter: str = "Mnu") -> list[str]:
    """Return blockers/warnings before using a declared product for training."""
    issues = []
    if not spec.suite_name.strip() or not spec.product_name.strip():
        issues.append("error: suite_name/product_name must identify an exact official product")
    if spec.redshift < 0:
        issues.append("error: redshift must be nonnegative")
    if spec.resolution < 4:
        issues.append("error: resolution is implausibly small for a research field product")
    if spec.box_size_mpc_h <= 0:
        issues.append("error: box_size_mpc_h must be positive")
    if target_parameter not in spec.parameter_names:
        issues.append(
            f"error: target parameter {target_parameter!r} is not declared in this product"
        )
    if target_parameter in spec.parameter_names:
        if spec.neutrino_mass_ev_min is None or spec.neutrino_mass_ev_max is None:
            issues.append(
                "error: neutrino mass range must be verified from the official parameter table"
            )
        elif (
            spec.neutrino_mass_ev_min < 0 or spec.neutrino_mass_ev_max <= spec.neutrino_mass_ev_min
        ):
            issues.append("error: invalid or non-varying neutrino mass range")
    if not spec.phase_policy.strip():
        issues.append("error: phase policy must be explicit for leakage-resistant split design")
    if spec.initialization_method.lower() in {"unknown", "", "tbd"}:
        issues.append(
            "warning: initial-condition method unknown; do not compare this product to another suite until reconciled"
        )
    issues.append(
        "warning: this metadata gate is not a validation of simulation fidelity or mass sensitivity"
    )
    return issues
