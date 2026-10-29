"""Role-based dataset suitability plan with hard scientific exclusions.

The source ranking is intentionally separated from its role in the research pipeline. A dataset may
be excellent for survey realism but be ineligible for supervised neutrino-mass training because it
has no known per-realization truth or does not span M_nu.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from nufield.data_registry.catalog import DATASETS, Objective
from nufield.data_registry.selector import recommend_datasets

Role = Literal[
    "primary_target_training",
    "classical_reference",
    "survey_robustness",
    "baryonic_domain_shift",
    "external_validation",
    "observed_data_application",
    "software_smoke_test",
]


@dataclass(frozen=True)
class SuitabilityDecision:
    source_id: str
    name: str
    role: Role
    eligible: bool
    confidence: str
    rationale: tuple[str, ...]
    hard_conditions: tuple[str, ...]
    avoid_claims: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def _decision(
    source_id: str, role: Role, eligible: bool, rationale: list[str], conditions: list[str]
) -> SuitabilityDecision:
    src = DATASETS[source_id]
    confidence = "curated_planning_guidance_only"
    return SuitabilityDecision(
        source_id,
        src.name,
        role,
        eligible,
        confidence,
        tuple(rationale),
        tuple(conditions),
        src.avoid_claiming,
    )


def build_suitability_plan(
    objective: Objective = "neutrino_mass_inference",
    *,
    budget: str = "workstation",
    exact_product_mass_grid_verified: bool = False,
    physics_baseline_validated: bool = False,
    survey_observation_model_validated: bool = False,
) -> dict[str, Any]:
    """Assign datasets to research roles; scores are not conflated with fitness for an exact task."""
    if budget not in {"laptop", "workstation", "hpc"}:
        raise ValueError("invalid budget")
    ranked = recommend_datasets(objective, max_results=len(DATASETS), compute_budget=budget)
    rank = {r.source_id: r.rank for r in ranked}
    decisions = []
    # The exact product must be verified. A source family-level label cannot silently certify a file.
    primary_ok = exact_product_mass_grid_verified and budget in {"workstation", "hpc"}
    decisions.append(
        _decision(
            "quijote_mnu",
            "primary_target_training",
            primary_ok,
            [
                "Best first source family for controlled massive-neutrino response studies.",
                "The exact sub-suite, parameter column, units, redshift, resolution and independent groups must be inspected.",
            ],
            [
                "verify nwLH or another chosen product actually provides the intended M_nu support",
                "verify units/cosmology and paired phases",
                "validate the neutrino-aware simulator product and field conventions",
                "make group-disjoint train/validation/test partitions",
            ],
        )
    )
    decisions.append(
        _decision(
            "quijote_mnu",
            "classical_reference",
            True,
            [
                "Use the same physical simulations to make a matched P(k)/correlation baseline.",
                "Derivative-focused Mnu_p/Mnu_pp/Mnu_ppp subsets may answer local-response questions but should not be assumed to provide a dense continuous posterior grid.",
            ],
            [
                "fit covariance/summary transformations on training simulations only",
                "lock test realizations before hyperparameter selection",
            ],
        )
    )
    decisions.append(
        _decision(
            "desi_dr1_mocks",
            "survey_robustness",
            True,
            [
                "Use mocks for survey mask/window, weighting and catalog pipeline checks after simulation-space training.",
                "Mock count is not equivalent to independent cosmology coverage.",
            ],
            [
                "pin tracer and redshift bin",
                "record mock family and covariance purpose",
                "do not assume a dense Mnu target grid",
            ],
        )
    )
    decisions.append(
        _decision(
            "abacussummit",
            "external_validation",
            physics_baseline_validated,
            [
                "Potential cross-suite or cross-code stress test where cosmology, phase, redshift and resolution align."
            ],
            [
                "inspect exact product-level Mnu labels",
                "harmonize simulation conventions",
                "do not treat sparse discrete masses as continuous validation",
            ],
        )
    )
    decisions.append(
        _decision(
            "camels",
            "baryonic_domain_shift",
            True,
            [
                "Use as a complementary astrophysical-domain-shift check where the chosen CAMELS product is relevant.",
                "Do not assume CAMELS is an Mnu-supervised target suite.",
            ],
            [
                "verify model family and astrophysical parameter variations",
                "define what constitutes a domain shift before looking at results",
            ],
        )
    )
    decisions.append(
        _decision(
            "desi_dr1_data",
            "observed_data_application",
            survey_observation_model_validated,
            [
                "Only for late-stage application after simulation validation and established survey analysis comparisons."
            ],
            [
                "survey selection/window/weights",
                "no supervised ground-truth Mnu labels per observed sky",
                "posterior predictive and nuisance sensitivity checks",
            ],
        )
    )
    decisions.append(
        _decision(
            "jax_fli",
            "software_smoke_test",
            True,
            [
                "Use its documented forward-modeling and lensing capabilities as an integration reference where compatible."
            ],
            [
                "do not infer compatibility merely from a shared parameter name",
                "pin software version and conventions",
            ],
        )
    )
    ordering = [
        "quijote_mnu",
        "abacussummit",
        "desi_dr1_mocks",
        "camels",
        "desi_dr1_data",
        "jax_fli",
    ]
    by_source = {d.source_id: d for d in decisions}
    return {
        "schema_version": "1.0",
        "objective": objective,
        "compute_budget": budget,
        "source_rankings_heuristic_only": [r.to_dict() for r in ranked],
        "role_decisions": [by_source[k].to_dict() for k in ordering if k in by_source],
        "primary_mass_training_source": "quijote_mnu" if primary_ok else None,
        "blocking_conditions": (
            []
            if primary_ok
            else [
                "Exact Quijote product-level neutrino-mass grid has not been verified, or compute budget is not yet suitable for the selected product."
            ]
        ),
        "recommended_order": [
            "A: verify source product/schema and data card",
            "B: inspect actual target-column values and units",
            "C: run paired mass-response and parameter-grid diagnostics",
            "D: reproduce classical baseline",
            "E: train summary MLP + NPE",
            "F: consider CNN3D; FNO only for periodic uniform grids",
            "G: test survey mocks / baryonic shift",
            "H: observed-data application only after validation",
        ],
        "rank_mapping": rank,
        "scientific_approval": False,
        "warning": "The exact local data files were not inspected by this planner. Suitability decisions are conditional; no source is approved based on family name alone.",
    }
