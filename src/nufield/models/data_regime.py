"""Rules and diagnostics for matching model capacity to data regime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataRegime:
    independent_groups: int
    parameter_dimension: int
    mass_unique_values: int
    observation_dimensions: int
    total_observation_elements: int
    field_voxels: int
    physics_level: str
    has_group_leakage_risk: bool
    periodic_regular_grid: bool
    modalities_matched: bool

    def to_dict(self):
        return asdict(self)


def characterize_regime(
    *,
    theta: np.ndarray,
    parameter_names: list[str],
    group_ids: np.ndarray,
    observation_shape: tuple[int, ...],
    physics_level: str,
    mass_parameter_name: str = "sum_mnu_ev",
    periodic_regular_grid: bool = False,
    modalities_matched: bool = False,
    simulation_ids: np.ndarray | None = None,
) -> DataRegime:
    y = np.asarray(theta, dtype=float)
    groups = np.asarray(group_ids).astype(str)
    if y.ndim != 2 or y.shape[1] != len(parameter_names) or len(groups) != len(y):
        raise ValueError("theta/group/name schema mismatch")
    if not np.isfinite(y).all():
        raise ValueError("theta contains non-finite values")
    if not observation_shape or min(observation_shape) < 1:
        raise ValueError("observation_shape must be positive")
    mass_unique = (
        int(np.unique(y[:, parameter_names.index(mass_parameter_name)]).size)
        if mass_parameter_name in parameter_names
        else 0
    )
    sim_dup = (
        False
        if simulation_ids is None
        else np.unique(np.asarray(simulation_ids).astype(str)).size < len(y)
    )
    return DataRegime(
        int(np.unique(groups).size),
        y.shape[1],
        mass_unique,
        len(observation_shape),
        int(np.prod(observation_shape)),
        int(np.prod(observation_shape[-3:])) if len(observation_shape) >= 3 else 0,
        physics_level,
        bool(sim_dup or len(np.unique(groups)) < len(groups)),
        periodic_regular_grid,
        modalities_matched,
    )


def model_capacity_advice(regime: DataRegime, *, budget: str = "workstation") -> dict[str, Any]:
    """Return capacity guardrails and a conservative shortlist, not a predicted leaderboard."""
    if budget not in {"laptop", "workstation", "hpc"}:
        raise ValueError("invalid budget")
    models = []
    warnings = []
    if regime.physics_level in {"toy", "prototype", "unknown"}:
        warnings.append(
            "Non-validated physics: all model outputs are engineering smoke tests only."
        )
    if regime.has_group_leakage_risk:
        warnings.append(
            "Repeated simulation IDs or multiple views may share parent information; group-safe splitting is mandatory."
        )
    if regime.mass_unique_values < 3:
        warnings.append(
            "Target mass grid has fewer than three levels; do not fit a continuous mass posterior."
        )
    models.append(
        {
            "model_id": "pk_classical",
            "eligible_for_benchmark": True,
            "priority": "first",
            "reason": "data-efficient and necessary scientific reference",
            "acceptance": "reproduce trusted summary/covariance behavior",
        }
    )
    if regime.independent_groups >= 100:
        models.append(
            {
                "model_id": "mlp_npe",
                "eligible_for_benchmark": True,
                "priority": "second",
                "reason": "learns from engineered summaries with lower compute than raw 3-D fields",
                "acceptance": "independent SBC and coverage versus classical baseline",
            }
        )
    else:
        models.append(
            {
                "model_id": "mlp_npe",
                "eligible_for_benchmark": False,
                "priority": "defer",
                "reason": "fewer than 100 independent groups; prioritize pipeline smoke tests/data acquisition",
                "acceptance": "re-evaluate after independent simulation count grows",
            }
        )
    if regime.field_voxels > 0:
        cnn_ok = regime.independent_groups >= 500 and (
            budget != "laptop" or regime.field_voxels <= 48**3
        )
        models.append(
            {
                "model_id": "cnn3d_npe",
                "eligible_for_benchmark": cnn_ok,
                "priority": "after_mlp",
                "reason": "requires enough independent simulation groups; full fields can exploit simulator artifacts",
                "acceptance": "matched compute, resolution holdout, field-shuffle negative control, calibrated posterior",
            }
        )
        fno_ok = (
            regime.independent_groups >= 1000
            and regime.periodic_regular_grid
            and budget in {"workstation", "hpc"}
        )
        models.append(
            {
                "model_id": "fno3d_npe",
                "eligible_for_benchmark": fno_ok,
                "priority": "specialized_ablation",
                "reason": "requires regular periodic field grid, enough independent cases, and non-laptop compute",
                "acceptance": "mode sweep, cross-resolution evaluation and same-budget CNN comparison",
            }
        )
    if regime.modalities_matched and regime.independent_groups >= 1000:
        models.append(
            {
                "model_id": "hybrid_npe",
                "eligible_for_benchmark": True,
                "priority": "late_extension",
                "reason": "candidate joint model only after all modalities use a matched latent cosmology/observation model",
                "acceptance": "single-modality ablation, shuffled pairing and joint posterior calibration",
            }
        )
    else:
        models.append(
            {
                "model_id": "hybrid_npe",
                "eligible_for_benchmark": False,
                "priority": "blocked",
                "reason": "multimodal alignment or independent groups are not adequate/verified",
                "acceptance": "demonstrate physical alignment and modality-specific noise/mask model first",
            }
        )
    return {
        "regime": regime.to_dict(),
        "compute_budget": budget,
        "models": models,
        "warnings": warnings,
        "selector_kind": "hard-rule eligibility; no empirical accuracy scores used",
        "scientific_approval": False,
    }
