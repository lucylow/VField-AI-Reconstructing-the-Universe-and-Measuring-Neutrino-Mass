"""Hard contracts for AI experiments in scientific simulation workflows.

These checks are deliberately conservative. Passing a contract means that the inputs are
structurally eligible for an experiment; it does not establish that the simulator is physically
correct, the posterior is calibrated, or the resulting cosmology is publishable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ObservationKind = Literal["summary", "field3d", "map2d", "catalog", "multimodal"]
TaskKind = Literal["regression", "posterior", "reconstruction", "classification"]


@dataclass(frozen=True)
class ArrayContract:
    """Describe a numerical array without retaining its data."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    units: str = "dimensionless"
    finite: bool = True
    semantic: str = "unspecified"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name.strip():
            errors.append("array name must not be empty")
        if not self.shape or any(int(d) < 1 for d in self.shape):
            errors.append(f"{self.name}: shape dimensions must all be positive")
        if not self.dtype:
            errors.append(f"{self.name}: dtype must be declared")
        if not self.units.strip():
            errors.append(
                f"{self.name}: units must be explicit (use 'dimensionless' when appropriate)"
            )
        return errors


@dataclass(frozen=True)
class AIExperimentContract:
    """Eligibility checklist for training a model on simulation-based data."""

    task: TaskKind
    observation_kind: ObservationKind
    n_rows: int
    n_independent_groups: int
    n_targets: int
    train_validation_test_disjoint: bool
    preprocessing_fit_on_train_only: bool
    prior_declared: bool
    target_units_declared: bool
    simulator_status: Literal["toy", "prototype", "validated", "unknown"]
    parameter_variation_verified: bool = False
    modalities_share_latent_cosmology: bool = False
    target_observation_mapping_verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, for_scientific_claim: bool = False) -> list[str]:
        errors: list[str] = []
        if self.n_rows < 3:
            errors.append("at least three rows are required for a structural preflight")
        if self.n_independent_groups < 3:
            errors.append("at least three independent parent groups are required")
        if self.n_independent_groups > self.n_rows:
            errors.append("independent group count cannot exceed row count")
        if self.n_targets < 1:
            errors.append("n_targets must be positive")
        if not self.train_validation_test_disjoint:
            errors.append("train/validation/test groups must be disjoint")
        if not self.preprocessing_fit_on_train_only:
            errors.append("preprocessing must be fitted on the training partition only")
        if not self.prior_declared:
            errors.append("parameter prior must be declared")
        if not self.target_units_declared:
            errors.append("target units and convention must be declared")
        if self.task == "posterior" and not self.parameter_variation_verified:
            errors.append(
                "posterior task requires verified target variation in the training simulations"
            )
        if self.observation_kind == "multimodal" and not self.modalities_share_latent_cosmology:
            errors.append(
                "multimodal data must be generated from a documented shared latent cosmology"
            )
        if not self.target_observation_mapping_verified:
            errors.append("target-to-observation mapping must be documented and checked")
        if for_scientific_claim and self.simulator_status != "validated":
            errors.append("scientific claims require an externally validated simulator status")
        return errors

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["preflight_errors"] = self.validate()
        value["claim_eligible"] = not self.validate(for_scientific_claim=True)
        return value


def assert_group_disjoint(
    train_groups: Any,
    validation_groups: Any,
    test_groups: Any,
    *,
    allow_empty_test: bool = False,
) -> None:
    """Raise when parent simulation identifiers overlap across partitions."""
    import numpy as np

    train = set(np.asarray(train_groups).astype(str).tolist())
    validation = set(np.asarray(validation_groups).astype(str).tolist())
    test = set(np.asarray(test_groups).astype(str).tolist())
    if not train or not validation:
        raise ValueError("training and validation groups must be non-empty")
    if not test and not allow_empty_test:
        raise ValueError("test groups must be non-empty for final evaluation")
    pairs = (
        ("train", train, "validation", validation),
        ("train", train, "test", test),
        ("validation", validation, "test", test),
    )
    for a_name, a, b_name, b in pairs:
        overlap = a.intersection(b)
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(f"parent groups overlap between {a_name} and {b_name}: {examples}")


def validate_numeric_array(
    name: str, values: Any, *, ndim: int | None = None, allow_nan: bool = False
) -> None:
    """Validate finite real-valued model input arrays with actionable error messages."""
    import numpy as np

    arr = np.asarray(values)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"{name} must have a real numeric dtype; got {arr.dtype}")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not allow_nan and not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def assert_same_rows(**arrays: Any) -> int:
    """Require all named arrays to have the same leading sample dimension."""
    lengths = {name: len(value) for name, value in arrays.items()}
    if not lengths:
        raise ValueError("at least one array is required")
    unique = set(lengths.values())
    if len(unique) != 1:
        raise ValueError(f"sample counts differ: {lengths}")
    n = next(iter(unique))
    if n < 1:
        raise ValueError("arrays must contain at least one sample")
    return n
