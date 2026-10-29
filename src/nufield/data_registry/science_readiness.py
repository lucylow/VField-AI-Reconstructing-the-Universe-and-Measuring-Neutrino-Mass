"""Checklist-based scientific readiness assessment for a candidate cosmology dataset.

This is a conservative audit report, not automated peer review. It has no "approved" outcome: the
highest possible result is a set of documented preconditions still needing expert validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from nufield.data_registry.cards import DatasetCard


@dataclass(frozen=True)
class ReadinessItem:
    key: str
    satisfied: bool
    severity: str
    explanation: str
    evidence: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class ScienceReadinessReport:
    dataset_id: str
    checklist_stage: str
    items: list[ReadinessItem]
    blockers: list[str]
    warnings: list[str]
    scientific_approval: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scientific_approval"] = False
        result["satisfied_count"] = sum(item.satisfied for item in self.items)
        result["total_count"] = len(self.items)
        return result


def assess_science_readiness(
    card: DatasetCard,
    *,
    theta: np.ndarray | None = None,
    observed_parameter_names: list[str] | None = None,
    simulation_ids: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    independent_validation: dict[str, bool] | None = None,
) -> ScienceReadinessReport:
    """Audit prerequisites for Mν training while explicitly withholding scientific approval.

    `independent_validation` must be a record of actual completed validation checks, not checkboxes
    selected just to obtain a higher status. Evidence should accompany those booleans in experiment
    artifacts; this helper does not verify external evidence itself.
    """
    card_errors = card.validate()
    items: list[ReadinessItem] = []
    blockers: list[str] = []
    warnings: list[str] = []

    def add(key, ok, severity, explanation, evidence=""):
        item = ReadinessItem(key, bool(ok), severity, explanation, evidence)
        items.append(item)
        if not ok and severity == "blocker":
            blockers.append(f"{key}: {explanation}")
        elif not ok:
            warnings.append(f"{key}: {explanation}")

    add(
        "dataset_card_valid",
        not card_errors,
        "blocker",
        "; ".join(card_errors) if card_errors else "Dataset card schema is valid.",
    )
    add(
        "source_release",
        bool(card.release.strip() and card.source_url.strip()),
        "blocker",
        "A stable source release and source URL are required; do not use an unversioned web directory as an immutable identifier.",
    )
    add(
        "license_and_access",
        bool(card.license_name.strip()),
        "warning",
        "Record license/access terms and redistribution conditions.",
    )
    add(
        "physical_units",
        bool(card.units) and bool(card.density_definition),
        "blocker",
        "Document field quantity, units, density definition, and any h-scaled convention.",
    )
    add(
        "simulation_context",
        card.redshift is not None
        and card.box_size_mpc_h is not None
        and bool(card.simulation_code),
        "warning",
        "For simulation fields, record code, redshift/snapshot, and box size; catalogs need their own selection metadata.",
    )
    add(
        "mass_varies_in_card",
        card.neutrino_mass_varies is True
        and card.mass_range_ev is not None
        and card.mass_range_ev[1] > card.mass_range_ev[0],
        "blocker",
        "The specific product must explicitly vary Mnu across a documented range.",
    )
    if theta is None:
        add(
            "mass_varies_in_arrays",
            False,
            "blocker",
            "Load the parameter array and verify that the selected mass column truly varies.",
        )
    else:
        arr = np.asarray(theta, dtype=float)
        names = list(observed_parameter_names or card.parameter_names)
        mass_name = card.neutrino_mass_parameter
        ok_shape = (
            arr.ndim == 2
            and arr.shape[1] == len(names)
            and arr.shape[0] >= 3
            and mass_name in names
            and np.isfinite(arr).all()
        )
        if ok_shape:
            col = arr[:, names.index(mass_name)]
            varies = np.unique(col).size >= 3 and np.ptp(col) > 0
            add(
                "mass_varies_in_arrays",
                varies,
                "blocker",
                f"Array has {np.unique(col).size} distinct values for {mass_name}; require at least three distinct values for the initial benchmark.",
                f"shape={arr.shape}, min={float(col.min())}, max={float(col.max())}",
            )
        else:
            add(
                "mass_varies_in_arrays",
                False,
                "blocker",
                "Parameter array shape/name/finite-value checks failed.",
                f"shape={arr.shape}",
            )
    ids_ok = (
        simulation_ids is not None
        and theta is not None
        and len(np.asarray(simulation_ids)) == len(theta)
    )
    add(
        "simulation_ids",
        ids_ok,
        "blocker",
        "Use verified source simulation IDs; never fabricate IDs for research data.",
    )
    groups_ok = (
        group_ids is not None
        and theta is not None
        and len(np.asarray(group_ids)) == len(theta)
        and np.unique(np.asarray(group_ids)).size >= 3
    )
    add(
        "parent_group_ids",
        groups_ok,
        "blocker",
        "Record parent realization/phase IDs and split by group to prevent crop/phase leakage.",
    )
    add(
        "source_citation",
        bool(card.citation.strip()),
        "warning",
        "Add the canonical scientific citation and data-release citation to reports.",
    )
    checks = independent_validation or {}
    for key, text in [
        ("baseline_reproduced", "Reproduce a classical baseline against a trusted reference."),
        ("split_disjoint", "Verify no parent simulation crosses train/validation/test."),
        ("sbc_passed", "Run simulation-based calibration on independent prior draws."),
        (
            "posterior_predictive_passed",
            "Run posterior predictive checks with prespecified criteria.",
        ),
        ("cross_suite_passed", "Test cross-resolution or cross-suite/generalization as relevant."),
    ]:
        add(
            key,
            checks.get(key, False) is True,
            "warning",
            text,
            "Caller-supplied result; require linked log/results to verify.",
        )
    if blockers:
        stage = "data_contract_incomplete"
    elif not all(checks.get(k, False) is True for k in ("baseline_reproduced", "split_disjoint")):
        stage = "array_contract_met_physics_unverified"
    elif not all(
        checks.get(k, False) is True
        for k in ("sbc_passed", "posterior_predictive_passed", "cross_suite_passed")
    ):
        stage = "baseline_reproduced_calibration_incomplete"
    else:
        stage = "checklist_complete_requires_expert_review"
    return ScienceReadinessReport(
        card.dataset_id, stage, items, blockers, warnings, scientific_approval=False
    )
