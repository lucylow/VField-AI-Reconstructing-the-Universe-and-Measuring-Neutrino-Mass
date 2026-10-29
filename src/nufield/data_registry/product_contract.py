"""Product-level science contract: source-family reputation never substitutes for exact-product checks.

Use this contract after inspecting the official release documentation and the actual local file. It
records human-verified facts and then checks whether a data product is role-compatible. It cannot
independently validate the truth of those declared facts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

ProductRole = Literal[
    "primary_target_training",
    "classical_reference",
    "survey_robustness",
    "baryonic_domain_shift",
    "external_validation",
    "observed_data_application",
    "software_smoke_test",
]
DataKind = Literal["simulation", "mock", "observed", "software"]


@dataclass(frozen=True)
class DatasetProductContract:
    dataset_id: str
    source_family: str
    exact_product_id: str
    release: str
    source_url: str
    data_kind: DataKind
    intended_role: ProductRole
    license_or_terms: str
    product_schema_verified: bool
    source_cosmology_documented: bool
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    target_name: str
    target_unit: str
    target_values_verified: bool
    target_grid_kind: Literal["continuous", "discrete", "none", "unknown"]
    independent_group_definition: str
    source_has_known_truth: bool
    observed_product_has_target_truth: bool = False
    field_geometry: Literal[
        "periodic_box", "survey_masked", "projected_map", "summary_vector", "catalog", "unknown"
    ] = "unknown"
    coordinate_and_length_units_documented: bool = False
    redshift_or_snapshot_documented: bool = False
    product_notes: tuple[str, ...] = ()

    def validate_schema(self) -> list[str]:
        errors = []
        for name in (
            "dataset_id",
            "source_family",
            "exact_product_id",
            "release",
            "source_url",
            "license_or_terms",
            "target_name",
            "target_unit",
            "independent_group_definition",
        ):
            if not str(getattr(self, name)).strip():
                errors.append(f"{name} must be nonempty")
        if len(self.parameter_names) != len(self.parameter_units):
            errors.append("parameter_names/parameter_units length mismatch")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            errors.append("parameter_names must be unique")
        if self.target_name not in self.parameter_names:
            errors.append("target_name must appear in parameter_names")
        target_unit = (
            self.parameter_units[self.parameter_names.index(self.target_name)]
            if self.target_name in self.parameter_names
            and len(self.parameter_units) == len(self.parameter_names)
            else None
        )
        if target_unit is not None and target_unit != self.target_unit:
            errors.append(
                "target_unit must match the target column's explicit parameter_units entry"
            )
        if self.data_kind == "observed" and self.intended_role == "primary_target_training":
            errors.append("observed data cannot provide per-realization supervised cosmology truth")
        if self.data_kind == "software" and self.intended_role in {
            "primary_target_training",
            "classical_reference",
        }:
            errors.append("software-only sources cannot be simulation training arrays")
        if self.data_kind == "observed" and self.source_has_known_truth:
            errors.append(
                "observed sky should not be declared to have known per-realization theta truth"
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def review_dataset_product(
    contract: DatasetProductContract,
    theta: np.ndarray,
    *,
    parameter_names: list[str],
    group_ids: np.ndarray,
    prior_bounds: tuple[float, float],
    min_unique_target_values: int = 3,
    min_groups_per_target_value: int = 2,
) -> dict[str, Any]:
    """Check a declared product contract against loaded row/label arrays and role requirements."""
    schema_errors = contract.validate_schema()
    y = np.asarray(theta, dtype=float)
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    names = list(parameter_names)
    if y.ndim != 2 or y.shape[1] != len(names) or len(groups) != len(y):
        raise ValueError("theta/group/parameter dimensions are inconsistent")
    if not np.isfinite(y).all():
        raise ValueError("theta contains non-finite values")
    if names != list(contract.parameter_names):
        schema_errors.append(
            "loaded parameter_names differ from the product contract in order or spelling"
        )
    if contract.target_name not in names:
        raise ValueError("contract target name is not present in loaded theta")
    j = names.index(contract.target_name)
    target = y[:, j]
    unique = np.unique(target)
    lo, hi = map(float, prior_bounds)
    if not lo < hi:
        raise ValueError("prior bounds must be ordered")
    level_counts = []
    for val in unique:
        selected = np.isclose(target, val, atol=1e-10, rtol=0)
        level_counts.append(
            {
                "value": float(val),
                "rows": int(selected.sum()),
                "independent_groups": int(np.unique(groups[selected]).size),
            }
        )
    errors = list(schema_errors)
    warnings = []
    if np.any(target < lo) or np.any(target > hi):
        errors.append("target values outside configured prior bounds")
    if len(unique) < min_unique_target_values:
        errors.append(
            f"only {len(unique)} unique target values; need >= {min_unique_target_values}"
        )
    if contract.intended_role == "primary_target_training":
        if contract.data_kind != "simulation":
            errors.append(
                "primary target training requires controlled simulations with known labels"
            )
        if not contract.product_schema_verified:
            errors.append("exact product schema has not been verified")
        if not contract.source_cosmology_documented:
            errors.append("source cosmology/conventions are not documented")
        if not contract.target_values_verified:
            errors.append(
                "target values/units have not been explicitly verified against source documentation"
            )
        if not contract.source_has_known_truth:
            errors.append("simulation ground-truth theta is not declared available")
        if contract.target_grid_kind not in {"continuous", "discrete"}:
            errors.append("target grid kind is unknown or absent")
        if not contract.coordinate_and_length_units_documented:
            errors.append("field coordinate and length units are undocumented")
        if not contract.redshift_or_snapshot_documented:
            errors.append("redshift/snapshot is undocumented")
        if contract.target_unit != "eV" and contract.target_name in {"sum_mnu_ev", "Mnu"}:
            errors.append(
                "neutrino mass target must be explicitly converted to eV before canonical training"
            )
        if len(unique) >= min_unique_target_values and any(
            row["independent_groups"] < min_groups_per_target_value for row in level_counts
        ):
            warnings.append(
                "one or more target levels have fewer independent groups than the requested minimum"
            )
        if len(unique) < 8:
            warnings.append(
                "sparse target grid: focus on response/finite-difference studies; do not imply a well-supported continuous posterior"
            )
    elif contract.intended_role == "observed_data_application":
        if contract.data_kind != "observed":
            warnings.append("intended observed-data role is assigned to a non-observed product")
        if contract.observed_product_has_target_truth:
            warnings.append(
                "observed data declared to have known theta truth; verify this is not a mistaken supervised label"
            )
    elif contract.intended_role == "survey_robustness" and contract.data_kind not in {
        "mock",
        "observed",
    }:
        warnings.append(
            "survey-robustness role typically requires realistic catalogs, masks, weights, and windows"
        )
    result = {
        "schema_version": "1.0",
        "contract": contract.to_dict(),
        "n_rows": len(y),
        "n_groups": int(np.unique(groups).size),
        "target_unique_values": unique.tolist(),
        "target_range": [float(target.min()), float(target.max())],
        "prior_bounds": [lo, hi],
        "target_level_group_counts": level_counts,
        "errors": list(dict.fromkeys(errors)),
        "warnings": warnings,
        "eligible_for_declared_role": not errors,
        "scientific_approval": False,
        "interpretation": "This report checks consistency of declared metadata and observed arrays; it cannot verify source claims, physical validity, independence semantics, or licenses.",
    }
    return result
