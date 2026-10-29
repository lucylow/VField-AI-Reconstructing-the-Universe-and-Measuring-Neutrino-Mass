"""One-stop, fail-closed preflight for a candidate NPZ before AI model training.

It checks array and target contracts, parent-group identifiers, input numerical quality, dataset
role, and architecture eligibility. It writes a report but cannot approve the physical simulator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nufield.data_registry.assessment import assess_arrays
from nufield.data_registry.parameter_grid import inspect_parameter_grid
from nufield.data_registry.quality import audit_numeric_dataset
from nufield.data_registry.splits import group_train_val_test_split, split_fingerprint
from nufield.data_registry.suitability import build_suitability_plan
from nufield.models.architecture_selector import DataModelContract, eligible_architectures
from nufield.models.observation_quality import audit_observations


def run_experiment_preflight(
    dataset_path: str,
    *,
    observation_key: str,
    theta_key: str,
    parameter_names: list[str],
    simulation_id_key: str,
    group_id_key: str,
    observation_kind: str,
    dataset_id: str,
    release: str,
    physics_level: str,
    field_shape: tuple[int, ...] | None,
    prior_low: list[float],
    prior_high: list[float],
    budget: str = "workstation",
    periodic_box: bool = False,
    source_schema_verified: bool = False,
    has_lensing: bool = False,
    lensing_key: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        required = {observation_key, theta_key, simulation_id_key, group_id_key}
        missing = required - set(keys)
        if missing:
            raise KeyError(f"missing NPZ keys {sorted(missing)}; available={keys}")
        x = np.asarray(z[observation_key])
        theta = np.asarray(z[theta_key])
        ids = np.asarray(z[simulation_id_key]).astype(str)
        groups = np.asarray(z[group_id_key]).astype(str)
        lensing = np.asarray(z[lensing_key]) if lensing_key and lensing_key in z else None
    if len(prior_low) != theta.shape[1] or len(prior_high) != theta.shape[1]:
        raise ValueError("prior dimension differs from theta columns")
    if len(parameter_names) != theta.shape[1]:
        raise ValueError("parameter_names length differs from theta columns")
    if any(not (float(lo) < float(hi)) for lo, hi in zip(prior_low, prior_high)):
        raise ValueError("every prior bound pair must satisfy prior_low < prior_high")
    if (
        not np.isfinite(np.asarray(prior_low, dtype=float)).all()
        or not np.isfinite(np.asarray(prior_high, dtype=float)).all()
    ):
        raise ValueError("prior bounds must be finite")
    if len(ids) != len(theta) or len(groups) != len(theta):
        raise ValueError("ID/parent group arrays do not align")
    mass_name = (
        "sum_mnu_ev"
        if "sum_mnu_ev" in parameter_names
        else ("Mnu" if "Mnu" in parameter_names else "")
    )
    array_assessment = assess_arrays(
        x,
        theta,
        parameter_names=parameter_names,
        simulation_ids=ids,
        mass_parameter_name=mass_name or "sum_mnu_ev",
    )
    numerical_audit = audit_numeric_dataset(
        x,
        theta,
        parameter_names=parameter_names,
        dataset_id=dataset_id,
        simulation_ids=ids,
        group_ids=groups,
        metadata={
            "release": release,
            "physics_level": physics_level,
            "source_schema_verified": source_schema_verified,
        },
    )
    expected_kind = {
        "spectrum": "spectrum",
        "field3d": ("field3d_channel" if x.ndim == 5 else "field3d"),
        "catalog": "catalog",
        "image2d": "image2d",
        "multimodal": "field3d",
    }.get(observation_kind)
    observation_audit = audit_observations(
        x,
        expected_kind=expected_kind or observation_kind,
        expected_shape=field_shape if field_shape and observation_kind == "field3d" else None,
    )
    grid = inspect_parameter_grid(
        theta, parameter_names, target_names=(mass_name,) if mass_name else ()
    )
    group_count = int(np.unique(groups).size)
    ntest = max(1, int(round(group_count * 0.15)))
    nval = max(1, int(round(group_count * 0.15)))
    if group_count < 5:
        split_preview = {
            "created": False,
            "reason": "fewer than five parent groups; increase the number of independent simulations",
        }
    else:
        split = group_train_val_test_split(
            groups,
            fractions=(
                1 - ntest / group_count - nval / group_count,
                nval / group_count,
                ntest / group_count,
            ),
            seed=31415,
        )
        group_sets = {name: sorted(set(groups[idx].tolist())) for name, idx in split.items()}
        split_preview = {
            "created": True,
            "method": "group-disjoint unstratified preview; do not substitute for final frozen split",
            "seed": 31415,
            "n_parent_groups": group_count,
            "n_test_target": ntest,
            "n_validation_target": nval,
            "groups": group_sets,
            "group_hashes": {name: split_fingerprint(groups, idx) for name, idx in split.items()},
            "validation_note": "Generate a final stratified train/validation/test manifest and hash before training.",
        }
    shape = tuple(int(d) for d in (x.shape[1:] if field_shape is None else field_shape))
    if observation_kind == "field3d" and len(shape) == 4:
        model_shape = shape
    elif observation_kind == "field3d" and len(shape) == 3:
        model_shape = (1, *shape)
    else:
        model_shape = shape
    contract = DataModelContract(
        n_independent_groups=group_count,
        sample_shape=model_shape,
        observation_kind=observation_kind
        if observation_kind in {"spectrum", "field3d", "multimodal", "catalog", "image2d"}
        else "spectrum",
        target_dimension=theta.shape[1],
        has_mass_variation=bool(
            mass_name and np.unique(theta[:, parameter_names.index(mass_name)]).size >= 3
        ),
        mass_grid_verified=(
            source_schema_verified
            and bool(mass_name)
            and np.unique(theta[:, parameter_names.index(mass_name)]).size >= 3
        ),
        train_validation_test_disjoint=False,
        physics_level=physics_level,
        mnu_target=bool(mass_name),
        regular_grid=observation_kind == "field3d",
        periodic_box=periodic_box,
        has_power_spectrum=(observation_kind == "spectrum" or "spectrum" in keys),
        has_lensing=(lensing is not None or has_lensing),
        modalities_physically_matched=(lensing is not None and has_lensing),
        prior_declared=True,
        source_schema_verified=source_schema_verified,
    )
    architectures = [x.to_dict() for x in eligible_architectures(contract, budget=budget)]
    plan = build_suitability_plan(
        "neutrino_mass_inference",
        budget=budget,
        exact_product_mass_grid_verified=source_schema_verified,
        physics_baseline_validated=False,
        survey_observation_model_validated=False,
    )
    errors = (
        [f.code for f in array_assessment.errors]
        + [f.code for f in numerical_audit.findings if f.severity == "error"]
        + [f.code for f in observation_audit.findings if f.severity == "error"]
    )
    report = {
        "schema_version": "1.0",
        "dataset_path": str(path.resolve()),
        "dataset_id": dataset_id,
        "release": release,
        "physics_level": physics_level,
        "n_rows": int(len(theta)),
        "n_parent_groups": group_count,
        "array_keys": keys,
        "parameter_names": parameter_names,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "source_schema_verified": source_schema_verified,
        "array_assessment": array_assessment.to_dict(),
        "numerical_audit": numerical_audit.to_dict(),
        "observation_audit": observation_audit.to_dict(),
        "parameter_grid": grid.to_dict(),
        "split_preview": split_preview,
        "architecture_eligibility": architectures,
        "dataset_suitability_plan": {
            "primary_mass_training_source": plan["primary_mass_training_source"],
            "blocking_conditions": plan["blocking_conditions"],
        },
        "blocking_error_codes": errors,
        "can_run_software_smoke_test": not bool(errors),
        "can_claim_physical_mnu_inference": False,
        "next_actions": [
            "verify exact source product and parameter units",
            "inspect target-grid response and identifiability",
            "freeze a parent-group train/validation/test split",
            "train classical P(k) reference",
            "only then run calibrated NPE experiments",
        ],
        "warning": "This preflight never grants scientific approval. The source schema flag is user-supplied and must be supported by inspected official documentation.",
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["output_path"] = str(out.resolve())
    return report
