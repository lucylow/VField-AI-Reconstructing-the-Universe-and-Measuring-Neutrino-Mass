"""Fail-closed data eligibility gates before selecting a posterior model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from nufield.data_registry.cards import DatasetCard
from nufield.data_registry.parameter_grid import assert_target_support


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    passed: bool
    severity: str
    evidence: str
    remediation: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DatasetGateReport:
    dataset_id: str
    target_name: str
    gates: tuple[GateResult, ...]
    eligible_for_software_smoke_test: bool
    eligible_for_preliminary_mass_training: bool
    eligible_for_observed_data_application: bool
    scientific_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "gates": [g.to_dict() for g in self.gates],
            "scientific_approval": False,
        }


def evaluate_dataset_gates(
    card: DatasetCard,
    *,
    theta: np.ndarray | None,
    observed_parameter_names: list[str] | None,
    simulation_ids: np.ndarray | None,
    group_ids: np.ndarray | None,
    expected_source_release: str | None = None,
    prior_bounds: tuple[float, float] | None = None,
    split_manifest: dict | None = None,
    source_schema_verified: bool = False,
    physics_validation_evidence: dict[str, bool] | None = None,
) -> DatasetGateReport:
    """Make data suitability explicit; the function deliberately cannot approve a science claim."""
    gates = []

    def add(key, ok, severity, evidence, remediation):
        gates.append(GateResult(key, bool(ok), severity, evidence, remediation))

    card_errors = card.validate()
    add(
        "dataset_card_valid",
        not card_errors,
        "blocker",
        "; ".join(card_errors) if card_errors else "DatasetCard fields pass structural validation.",
        "Correct required metadata and rerun card validation.",
    )
    release_ok = expected_source_release is None or card.release == expected_source_release
    add(
        "release_matches",
        release_ok,
        "blocker",
        f"card release={card.release!r}; expected={expected_source_release!r}",
        "Pin a specific source release and checksum.",
    )
    add(
        "schema_verified",
        source_schema_verified,
        "blocker",
        "Caller-supplied source-schema review flag.",
        "Inspect official product schema and explicitly map every key, unit, selection, and identifier.",
    )
    names = list(observed_parameter_names or card.parameter_names)
    if theta is None:
        target_ok = False
        evidence = "No parameter matrix supplied."
    else:
        t = np.asarray(theta)
        target = card.neutrino_mass_parameter
        target_ok = (
            t.ndim == 2 and target in names and t.shape[1] == len(names) and np.isfinite(t).all()
        )
        evidence = f"theta_shape={t.shape}; target={target}; names={names}"
    add(
        "target_column_mapped",
        target_ok,
        "blocker",
        evidence,
        "Map the actual target column and canonical units from verified source metadata.",
    )
    mass_varies = False
    support_errors = []
    if target_ok and prior_bounds is not None:
        support = assert_target_support(
            np.asarray(theta),
            names,
            target_name=card.neutrino_mass_parameter,
            prior_bounds=prior_bounds,
        )
        mass_varies = support["eligible_for_preliminary_target_training"]
        support_errors = support["errors"]
        support_evidence = f"data_range={support['data_range']}; n_unique={support['n_unique']}; errors={support_errors}"
    else:
        support_evidence = "Actual target/prior support not evaluated; prior_bounds required."
    mass_gate = card.neutrino_mass_varies is True and card.mass_range_ev is not None and mass_varies
    add(
        "mnu_varies_and_supports_prior",
        mass_gate,
        "blocker",
        support_evidence,
        "Select a genuinely mass-varying product; confirm units and prior range; inspect the actual parameter array.",
    )
    ids_ok = (
        theta is not None
        and simulation_ids is not None
        and len(np.asarray(simulation_ids)) == len(theta)
    )
    unique_ids = ids_ok and np.unique(np.asarray(simulation_ids).astype(str)).size == len(theta)
    add(
        "simulation_ids_align",
        ids_ok,
        "blocker",
        f"aligned={ids_ok}; unique={unique_ids}",
        "Attach verified source realization identifiers; duplicates may be views but need parent-group IDs.",
    )
    groups_ok = (
        theta is not None
        and group_ids is not None
        and len(np.asarray(group_ids)) == len(theta)
        and np.unique(np.asarray(group_ids).astype(str)).size >= 3
    )
    add(
        "parent_groups_available",
        groups_ok,
        "blocker",
        f"aligned={groups_ok}; n_groups={np.unique(np.asarray(group_ids).astype(str)).size if groups_ok else 0}",
        "Record initial-condition/phase/parent identifiers and split at parent level.",
    )
    split_ok = False
    if split_manifest:
        ids = set(np.asarray(group_ids).astype(str)) if groups_ok else set()
        split_groups = {
            name: set(map(str, split_manifest.get(name, [])))
            for name in ("train", "validation", "test")
        }
        split_ok = (
            bool(ids)
            and all(split_groups[n] for n in split_groups)
            and not (
                split_groups["train"] & split_groups["validation"]
                or split_groups["train"] & split_groups["test"]
                or split_groups["validation"] & split_groups["test"]
            )
        )
    add(
        "split_manifest_disjoint",
        split_ok,
        "blocker",
        "Caller-supplied group split manifest; this check verifies set disjointness only.",
        "Write a frozen split manifest derived from parent IDs, hash it, and verify every row is assigned exactly once.",
    )
    evidence = physics_validation_evidence or {}
    core_physics = all(
        evidence.get(k) is True
        for k in (
            "simulator_validated",
            "baseline_reproduced",
            "sbc_passed",
            "posterior_predictive_passed",
        )
    )
    add(
        "physics_and_inference_validated",
        core_physics,
        "warning",
        str(
            {
                k: evidence.get(k, False)
                for k in (
                    "simulator_validated",
                    "baseline_reproduced",
                    "sbc_passed",
                    "posterior_predictive_passed",
                )
            }
        ),
        "Independent domain-expert review and linked experiment artifacts are required.",
    )
    blockers = any(g.severity == "blocker" and not g.passed for g in gates)
    smoke = not any(
        g.gate_id in {"dataset_card_valid", "target_column_mapped"} and not g.passed for g in gates
    )
    mass = not blockers and mass_gate
    observed = (
        mass
        and core_physics
        and card.observable_kind in {"galaxy_catalog", "density_field", "lss_catalog", "survey_map"}
    )
    return DatasetGateReport(
        card.dataset_id, card.neutrino_mass_parameter, tuple(gates), smoke, mass, observed, False
    )
