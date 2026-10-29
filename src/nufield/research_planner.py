"""Join dataset and model recommendations into a gate-driven, explainable research plan.

The output is a planning artifact, not an auto-approval and not a prediction of scientific accuracy.
It puts the physical data contract ahead of model complexity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from nufield.data_registry.selector import recommend_datasets
from nufield.models.selector import recommend_models


@dataclass(frozen=True)
class ResearchStage:
    stage_id: str
    title: str
    purpose: str
    tasks: tuple[str, ...]
    exit_criteria: tuple[str, ...]
    blocking: bool = True

    def to_dict(self):
        return asdict(self)


STAGES = (
    ResearchStage(
        "D0",
        "Data product verification",
        "Prevent training on a dataset that does not identify the target parameter.",
        (
            "Resolve exact product/release and schema",
            "Record mass column, units, redshift, box size, density convention, phase/parent IDs",
            "Compute checksum manifest",
        ),
        (
            "Dataset card matches inspected product",
            "At least three distinct target values for a preliminary mass-response check",
            "Independent groups and provenance identified",
        ),
    ),
    ResearchStage(
        "D1",
        "Signal response and identifiability",
        "Demonstrate that the chosen representation contains measurable target-dependent information.",
        (
            "Run matched-phase low/high-mass response",
            "Compute P(k) response per k-bin",
            "Inspect parameter degeneracies and train/test support",
        ),
        (
            "Response stable over matched phases",
            "No unresolved target leakage",
            "Sensitivity survives basic numerical and sample-size checks",
        ),
    ),
    ResearchStage(
        "M0",
        "Classical reference",
        "Establish a trustworthy benchmark before any deep architecture.",
        (
            "Power-spectrum/correlation-function baseline",
            "Document covariance, prior, nuisance assumptions",
            "Evaluate on held-out groups",
        ),
        (
            "Parameters and units reproduce reference behavior",
            "Held-out results reported with uncertainty",
        ),
    ),
    ResearchStage(
        "M1",
        "Low-capacity neural baseline",
        "Test whether learned summaries add value without a raw 3-D model.",
        (
            "Train summary MLP + NPE",
            "Compare density estimators and simulation budgets",
            "Run SBC and posterior predictive checks",
        ),
        (
            "Calibration and bias within predeclared thresholds",
            "Paired comparison uses same test groups",
            "Compute cost recorded",
        ),
    ),
    ResearchStage(
        "M2",
        "Field-level model comparison",
        "Test whether field morphology adds useful information.",
        (
            "Compare residual CNN3D to MLP/P(k)",
            "Optionally compare FNO only on periodic regular grids",
            "Run field-shuffle and resolution ablations",
        ),
        (
            "Any gain exceeds finite-test uncertainty",
            "Gain survives systematics and cross-suite tests",
            "No shortcut-learning evidence",
        ),
    ),
    ResearchStage(
        "M3",
        "Multimodal extension",
        "Test complementary channels rather than simply concatenating maps.",
        (
            "Use physically matched galaxy/lensing simulations",
            "Run single-modality and shuffled-pairing ablations",
            "Check joint posterior calibration",
        ),
        ("Joint model improves predeclared proper score or interval width with calibration intact"),
    ),
    ResearchStage(
        "R0",
        "Observed-data application",
        "Apply a validated pipeline to real survey observables only after simulation validation.",
        (
            "Implement survey selection/window/weights and observation model",
            "Compare established summary-likelihood results",
            "Report nuisance assumptions and posterior predictive checks",
        ),
        ("All physical and inference gates reviewed by a domain expert"),
    ),
)


def plan_research_project(
    *,
    objective: str = "neutrino_mass_inference",
    n_simulations: int = 1500,
    field_shape: tuple[int, ...] | None = (32, 32, 32),
    budget: str = "workstation",
    periodic_box: bool = False,
    has_lensing: bool = False,
    mass_grid_verified: bool = False,
    target_uncertainty: bool = True,
) -> dict[str, Any]:
    """Return source ranking, model candidates, and a gate-driven implementation sequence."""
    datasets = recommend_datasets(objective, max_results=6, compute_budget=budget)
    task_map = {
        "neutrino_mass_inference": "parameter_posterior",
        "field_reconstruction": "field_reconstruction",
        "survey_validation": "survey_summary",
        "baryonic_robustness": "parameter_posterior",
        "multimodal_extension": "multimodal_posterior",
        "software_smoke_test": "parameter_posterior",
    }
    model_task = task_map.get(objective)
    if model_task is None:
        raise ValueError(f"unsupported objective {objective!r}")
    candidates = recommend_models(
        task=model_task,
        n_simulations=n_simulations,
        field_shape=field_shape,
        has_power_spectrum=True,
        has_lensing=has_lensing,
        has_mock_domain_shift=(objective in {"survey_validation", "baryonic_robustness"}),
        budget=budget,
        target_uncertainty=target_uncertainty,
        periodic_box=periodic_box,
    )
    blockers = []
    if objective == "neutrino_mass_inference" and not mass_grid_verified:
        blockers.append(
            "Verify the exact data product has genuine, correctly labelled variation in sum(m_nu) before fitting a mass posterior."
        )
    if n_simulations < 50:
        blockers.append(
            "Fewer than 50 independent simulations: do only code smoke tests, not a meaningful learned posterior."
        )
    if objective in {"neutrino_mass_inference", "multimodal_extension"} and not field_shape:
        blockers.append("No field shape supplied; a full-field architecture cannot be assessed.")
    if "fno3d" in {x.name for x in candidates} and not periodic_box:
        blockers.append(
            "FNO is excluded unless a regular periodic simulation box is explicitly declared."
        )
    return {
        "schema_version": "1.0",
        "project": "nuField",
        "objective": objective,
        "budget": budget,
        "n_simulations_planned": int(n_simulations),
        "field_shape": list(field_shape) if field_shape else None,
        "periodic_box_declared": periodic_box,
        "mass_grid_verified": mass_grid_verified,
        "scientific_approval": False,
        "readiness": "BLOCKED_BY_DATA_CHECK" if blockers else "PLANNING_ONLY",
        "blockers": blockers,
        "dataset_candidates": [d.to_dict() for d in datasets],
        "model_candidates": [c.to_dict() for c in candidates],
        "recommended_sequence": [stage.to_dict() for stage in STAGES],
        "model_policy": [
            "Power-spectrum/classical baseline first",
            "Summary MLP + NPE second",
            "CNN/FNO only after benchmark evidence",
            "FNO only for periodic boxes",
            "Multimodal model only when modalities share a physically matched latent cosmology",
        ],
        "selection_warning": "Dataset and model scores are engineering triage heuristics, not accuracy predictions or scientific approvals.",
    }
