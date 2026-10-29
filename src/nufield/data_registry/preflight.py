"""Combined preflight gate for a dataset card and its derived array bundle."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from nufield.data_registry.assessment import assess_arrays
from nufield.data_registry.cards import DatasetCard
from nufield.data_registry.support import parameter_support_report


@dataclass
class PreflightReport:
    ready_for_pipeline_smoke_test: bool
    ready_for_neural_training: bool
    ready_for_physical_mnu_claim: bool
    blocking_issues: list[dict]
    warnings: list[dict]
    array_assessment: dict
    support_report: dict
    card_status: str

    def to_dict(self) -> dict:
        return asdict(self)


def preflight(
    card: DatasetCard,
    observations: np.ndarray,
    theta: np.ndarray,
    *,
    simulation_ids: np.ndarray,
    group_ids: np.ndarray,
    prior_low: list[float] | None = None,
    prior_high: list[float] | None = None,
    mass_parameter_name: str | None = None,
) -> PreflightReport:
    """Evaluate metadata and numeric gates. Final science approval is always a human process."""
    blockers = []
    warnings = []
    for err in card.validate():
        blockers.append({"code": "CARD_INVALID", "message": err})
    if card.status == "unverified":
        warnings.append(
            {
                "code": "CARD_UNVERIFIED",
                "message": "Verify exact source product, parameter design, units, and access terms.",
            }
        )
    if card.neutrino_mass_varies is False:
        blockers.append(
            {
                "code": "CARD_MNU_FIXED",
                "message": "Card states neutrino mass does not vary; not suitable for supervised Mnu posterior training.",
            }
        )
    if card.neutrino_mass_varies is None:
        blockers.append(
            {
                "code": "CARD_MNU_UNKNOWN",
                "message": "Card does not verify that neutrino mass varies.",
            }
        )
    names = card.parameter_names
    mass_name = mass_parameter_name or card.neutrino_mass_parameter
    assessment = assess_arrays(
        observations,
        theta,
        parameter_names=names,
        simulation_ids=simulation_ids,
        mass_parameter_name=mass_name,
    )
    for f in assessment.findings:
        item = {"code": f.code, "message": f.message, "recommendation": f.recommendation}
        if f.severity == "error":
            blockers.append(item)
        elif f.severity == "warning":
            warnings.append(item)
    ids = np.asarray(simulation_ids).astype(str)
    groups = np.asarray(group_ids).astype(str)
    if len(ids) != len(theta) or len(groups) != len(theta):
        blockers.append(
            {"code": "IDS_SHAPE", "message": "simulation/group IDs must align with theta"}
        )
    else:
        # Repeated IDs can be legit derived rows, but row-level independence must not be assumed.
        if len(set(ids.tolist())) != len(ids):
            warnings.append(
                {
                    "code": "NON_UNIQUE_SIMULATION_IDS",
                    "message": "Rows share simulation IDs; splits and confidence intervals must use parent groups.",
                }
            )
        if len(set(groups.tolist())) < 3:
            blockers.append(
                {
                    "code": "TOO_FEW_GROUPS",
                    "message": "Need at least three independent groups for a train/validation/test smoke split.",
                }
            )
    support = parameter_support_report(
        theta, names, prior_low=prior_low, prior_high=prior_high, mass_parameter_name=mass_name
    )
    for issue in support["issues"]:
        if issue["severity"] == "error":
            blockers.append(issue)
        else:
            warnings.append(issue)
    if card.source_id == "desi_dr1_data":
        blockers.append(
            {
                "code": "OBSERVED_DATA_NOT_SUPERVISED",
                "message": "Observed DESI catalogs do not have known per-observation true theta; use for late application, not supervised labels.",
            }
        )
    if card.source_id == "camels" and card.neutrino_mass_varies is not True:
        warnings.append(
            {
                "code": "CAMELS_ROLE",
                "message": "Use as domain-shift/feedback robustness source unless selected suite explicitly varies Mnu.",
            }
        )
    if card.status not in {"simulation_validated", "research_approved"}:
        warnings.append(
            {
                "code": "SIMULATOR_UNVALIDATED",
                "message": "Dataset card does not establish validated physics; high-level claims are blocked regardless of array checks.",
            }
        )
    smoke_ok = not any(
        x["code"] in {"CARD_INVALID", "IDS_SHAPE", "TOO_FEW_GROUPS"} for x in blockers
    ) and not any(f.severity == "error" for f in assessment.findings)
    train_ok = (
        smoke_ok
        and assessment.can_train_mnu_posterior
        and card.neutrino_mass_varies is True
        and card.status
        in {"metadata_checked", "arrays_checked", "simulation_validated", "research_approved"}
    )
    # This code never automatically approves a physical claim.
    return PreflightReport(
        smoke_ok, train_ok, False, blockers, warnings, assessment.to_dict(), support, card.status
    )
