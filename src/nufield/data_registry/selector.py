"""Explainable data-source ranking, with suitability gates before weighted scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from nufield.data_registry.catalog import DATASETS, DatasetSource, Objective

WEIGHTS = {
    "neutrino_mass_inference": {
        "mass": 0.36,
        "field": 0.24,
        "coverage": 0.22,
        "access": 0.08,
        "compute": 0.10,
    },
    "field_reconstruction": {
        "field": 0.36,
        "coverage": 0.20,
        "access": 0.10,
        "compute": 0.16,
        "mass": 0.18,
    },
    "survey_validation": {
        "realism": 0.48,
        "access": 0.18,
        "field": 0.12,
        "compute": 0.10,
        "coverage": 0.12,
    },
    "baryonic_robustness": {
        "realism": 0.10,
        "field": 0.20,
        "coverage": 0.20,
        "access": 0.10,
        "compute": 0.10,
        "baryons": 0.30,
    },
    "multimodal_extension": {
        "realism": 0.28,
        "field": 0.23,
        "coverage": 0.17,
        "access": 0.14,
        "compute": 0.08,
        "multimodal": 0.10,
    },
    "software_smoke_test": {
        "access": 0.30,
        "compute": 0.30,
        "field": 0.15,
        "coverage": 0.10,
        "realism": 0.05,
        "mass": 0.10,
    },
}


@dataclass(frozen=True)
class DatasetRecommendation:
    source_id: str
    name: str
    score_0_to_100: float
    rank: int
    suitable_for_direct_mnu_training: bool
    strengths: tuple[str, ...]
    caveats: tuple[str, ...]
    suggested_first_action: str

    def to_dict(self) -> dict:
        return asdict(self)


def _feature_score(src: DatasetSource, feature: str) -> float:
    if feature == "mass":
        return {
            "explicit": 5.0,
            "limited": 2.5,
            "not_default": 0.8,
            "unknown": 0.5,
            "not_applicable": 0.0,
        }[src.mass_variation]
    if feature == "baryons":
        return 5.0 if src.source_id == "camels" else 1.5
    if feature == "multimodal":
        return (
            5.0
            if src.source_id == "jax_fli"
            else (4.0 if src.source_id == "desi_dr1_data" else 2.0)
        )
    field = {
        "field": src.field_level,
        "realism": src.observational_realism,
        "coverage": src.parameter_coverage,
        "access": src.accessibility,
        "compute": src.compute_friendliness,
    }
    return float(field[feature])


def recommend_datasets(
    objective: Objective = "neutrino_mass_inference",
    *,
    max_results: int = 8,
    compute_budget: str = "laptop",
) -> list[DatasetRecommendation]:
    """Rank candidate resources; for mass inference, fail closed on non-varying labels.

    Scores are planning heuristics. A `score_0_to_100` is not a data-quality probability and must
    not be used as an automated scientific approval. The exact subset/card still needs inspection.
    """
    if objective not in WEIGHTS:
        raise ValueError(f"Unsupported objective {objective!r}; choose from {sorted(WEIGHTS)}")
    if max_results < 1:
        raise ValueError("max_results must be >= 1")
    weights = WEIGHTS[objective]
    rows: list[tuple[float, DatasetSource, list[str]]] = []
    for src in DATASETS.values():
        raw = sum(weights[f] * _feature_score(src, f) for f in weights)
        score = raw / (5.0 * sum(weights.values())) * 100.0
        reasons = []
        if objective == "neutrino_mass_inference" and src.mass_variation != "explicit":
            score *= 0.43 if src.mass_variation == "limited" else 0.12
            reasons.append(
                "Mnu training gate: exact chosen product must contain a genuine neutrino-mass parameter grid."
            )
        if compute_budget == "laptop" and src.compute_friendliness <= 2:
            score *= 0.78
            reasons.append(
                "Large-scale fields/data will likely require downsampled products, remote compute, or HPC."
            )
        if objective == "survey_validation" and src.source_id == "desi_dr1_data":
            reasons.append(
                "Use as observed-data application only after mock-based validation; no true label exists per observation."
            )
        if src.source_id == "camels":
            reasons.append(
                "Recommended mainly for astrophysical/domain-shift tests, not assumed Mnu supervision."
            )
        rows.append((score, src, reasons))
    rows.sort(key=lambda item: item[0], reverse=True)
    output = []
    for i, (score, src, extra) in enumerate(rows[:max_results], 1):
        direct = src.mass_variation == "explicit"
        strengths = [src.primary_role]
        if objective in src.objectives:
            strengths.append(f"Listed use case includes {objective.replace('_', ' ')}.")
        output.append(
            DatasetRecommendation(
                source_id=src.source_id,
                name=src.name,
                score_0_to_100=round(score, 1),
                rank=i,
                suitable_for_direct_mnu_training=direct,
                strengths=tuple(strengths),
                caveats=tuple(list(src.limitations[:2]) + extra),
                suggested_first_action=src.recommended_use,
            )
        )
    return output
