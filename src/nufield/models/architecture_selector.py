"""Hard-gated architecture selection based on data contracts and compute assumptions.

Scores are deliberately absent. This utility answers whether a model family is eligible to be
benchmarked; it does not predict which model will have the best cosmological accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Budget = Literal["laptop", "workstation", "hpc"]


@dataclass(frozen=True)
class DataModelContract:
    n_independent_groups: int
    sample_shape: tuple[int, ...]
    observation_kind: Literal["spectrum", "field3d", "multimodal", "catalog", "image2d"]
    target_dimension: int
    has_mass_variation: bool
    mass_grid_verified: bool
    train_validation_test_disjoint: bool
    physics_level: Literal["toy", "prototype", "validated", "unknown"]
    mnu_target: bool = True
    regular_grid: bool = False
    periodic_box: bool = False
    has_power_spectrum: bool = True
    has_lensing: bool = False
    modalities_physically_matched: bool = False
    prior_declared: bool = False
    source_schema_verified: bool = False

    def validate(self) -> list[str]:
        errors = []
        if self.n_independent_groups < 3:
            errors.append(
                "At least three independent groups are required even for a preliminary split."
            )
        if not self.sample_shape or min(self.sample_shape) < 1:
            errors.append("sample_shape must be positive and non-empty.")
        if self.target_dimension < 1:
            errors.append("target_dimension must be positive.")
        if self.observation_kind == "field3d" and len(self.sample_shape) not in {3, 4}:
            errors.append("field3d shape must be [D,H,W] or [C,D,H,W].")
        if self.periodic_box and not self.regular_grid:
            errors.append("periodic_box requires regular_grid=True.")
        if self.has_lensing and not self.modalities_physically_matched:
            errors.append(
                "lensing modality declared without proof of a shared latent cosmology/forward model."
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureDecision:
    model_id: str
    eligible: bool
    reason: str
    must_pass_before_training: tuple[str, ...]
    resource_class: str

    def to_dict(self):
        return asdict(self)


def eligible_architectures(
    contract: DataModelContract, *, budget: Budget = "workstation"
) -> list[ArchitectureDecision]:
    """Return model gates and required checks, including baselines even when neural models are blocked."""
    if budget not in {"laptop", "workstation", "hpc"}:
        raise ValueError("unknown budget")
    errors = contract.validate()
    if errors:
        return [ArchitectureDecision("invalid_contract", False, "; ".join(errors), (), "none")]
    baseline_checks = (
        "documented observable/units",
        "held-out independent simulation groups",
        "posterior predictive checks",
        "covariance/summary assumptions",
    )
    common_neural = (
        "declared prior bounds",
        "split manifest frozen",
        "train-only normalization",
        "independent SBC simulations",
        "locked test groups",
    )
    candidates = [
        ArchitectureDecision(
            "pk_classical",
            True,
            "eligible when a defensible summary-statistic/likelihood path exists",
            baseline_checks,
            "low",
        )
    ]
    mass_gate = (not contract.mnu_target) or (
        contract.has_mass_variation
        and contract.mass_grid_verified
        and contract.source_schema_verified
    )
    if contract.has_power_spectrum:
        ok = contract.n_independent_groups >= 100 and mass_gate
        why = (
            "eligible for a preliminary summary-NPE trial"
            if ok
            else (
                "blocked for Mnu: verify actual mass variation and source schema first"
                if not mass_gate
                else "needs at least 100 independent groups for a preliminary neural trial; fewer is a smoke test only"
            )
        )
        candidates.append(
            ArchitectureDecision(
                "mlp_npe", ok, why, common_neural + ("summary-bin stability ablation",), "low"
            )
        )
    is_field = contract.observation_kind == "field3d"
    cnn_budget_ok = budget != "laptop" or (
        len(contract.sample_shape) >= 3 and max(contract.sample_shape[-3:]) <= 48
    )
    if is_field:
        cnn_ok = contract.n_independent_groups >= 500 and cnn_budget_ok and mass_gate
        cnn_why = (
            "eligible for a preliminary field-level benchmark"
            if cnn_ok
            else (
                "blocked for Mnu: verify actual mass variation and source schema first"
                if not mass_gate
                else (
                    "needs at least 500 independent groups"
                    if contract.n_independent_groups < 500
                    else "field is too large for the default laptop budget; downsample only with a resolution-convergence study"
                )
            )
        )
        candidates.append(
            ArchitectureDecision(
                "cnn3d_npe",
                cnn_ok,
                cnn_why,
                common_neural
                + ("cross-phase holdout", "resolution stress test", "field-shuffle ablation"),
                "medium" if cnn_budget_ok else "high",
            )
        )
        fno_ok = (
            contract.regular_grid
            and contract.periodic_box
            and budget in {"workstation", "hpc"}
            and contract.n_independent_groups >= 1000
            and mass_gate
        )
        why = "requires a regular periodic grid and a compute budget suitable for spectral layers"
        if not mass_gate:
            why = "blocked for Mnu: verify actual mass variation and source schema first"
        if not contract.periodic_box:
            why = "blocked: periodic boundary assumption has not been declared"
        elif not contract.regular_grid:
            why = "blocked: observations are not a regular grid"
        elif budget == "laptop":
            why = "blocked on laptop budget; test memory and smaller modes before enabling"
        elif contract.n_independent_groups < 1000:
            why = "blocked: fewer than 1000 independent groups; consider only a controlled architecture diagnostic"
        candidates.append(
            ArchitectureDecision(
                "fno3d_npe",
                fno_ok,
                why,
                common_neural
                + (
                    "mode-count ablation",
                    "cross-resolution held-out test",
                    "matched compute versus CNN",
                ),
                "high",
            )
        )
    if contract.observation_kind == "multimodal" or contract.has_lensing:
        multimodal_ok = (
            contract.modalities_physically_matched
            and contract.n_independent_groups >= 1000
            and contract.prior_declared
            and mass_gate
        )
        reason = (
            "eligible for a pilot only when modalities share a validated latent simulation and prior"
            if multimodal_ok
            else (
                "blocked for Mnu: verify actual mass variation and source schema first"
                if not mass_gate
                else "blocked until modalities, prior, units and latent cosmology are physically matched"
            )
        )
        candidates.append(
            ArchitectureDecision(
                "hybrid_npe",
                multimodal_ok,
                reason,
                common_neural
                + (
                    "single-modality ablations",
                    "shuffled pairing negative control",
                    "joint posterior calibration",
                ),
                "high",
            )
        )
    # A declared toy/prototype source may exercise the software but never supports real physical claims.
    if contract.physics_level in {"toy", "prototype", "unknown"}:
        candidates = [
            ArchitectureDecision(
                c.model_id,
                c.eligible,
                c.reason + "; source physics is not validated, so outputs are software tests only",
                c.must_pass_before_training,
                c.resource_class,
            )
            for c in candidates
        ]
    return candidates
