"""Grouped-split end-to-end conditional MDN posterior experiment for summary features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nufield.data_registry.preprocessing import Standardizer
from nufield.data_registry.splits import (
    group_train_validation_split,
    split_fingerprint,
    stratified_group_train_val_test_split,
)
from nufield.models.bounded_transform import BoundedParameterTransform
from nufield.models.conditional_mdn import MDNConfig, draw_physical_posterior, fit_conditional_mdn
from nufield.validation.posterior_diagnostics import posterior_diagnostic_report


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train_mdn_from_npz(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    observation_key: str = "features",
    theta_key: str = "theta",
    group_id_key: str = "group_id",
    simulation_id_key: str = "simulation_id",
    parameter_names: list[str],
    prior_low: list[float],
    prior_high: list[float],
    target_name: str = "sum_mnu_ev",
    seed: int = 17,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    config: MDNConfig | None = None,
    n_validation_observations: int = 32,
    posterior_samples: int = 512,
    evaluate_test_once: bool = False,
    allow_toy: bool = False,
) -> dict[str, Any]:
    """Train on grouped training simulations, stop on validation, preserve final test untouched by default."""
    path = Path(dataset_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    physics_level = str(metadata.get("physics_level", "unknown"))
    if physics_level == "toy" and not allow_toy:
        raise ValueError(
            "dataset is marked toy; pass allow_toy only for software smoke tests, never physical claims"
        )
    with np.load(path, allow_pickle=False) as z:
        for key in (observation_key, theta_key, group_id_key, simulation_id_key):
            if key not in z:
                raise KeyError(f"required key {key!r} missing; available={sorted(z.files)}")
        x = np.asarray(z[observation_key], dtype=np.float32)
        theta = np.asarray(z[theta_key], dtype=np.float64)
        groups = np.asarray(z[group_id_key]).astype(str)
        ids = np.asarray(z[simulation_id_key]).astype(str)
    if x.ndim != 2:
        raise ValueError(
            "MDN currently expects a fixed-width summary feature matrix [N,F]; use CNN/NPE workflow for spatial observations"
        )
    if theta.ndim != 2 or theta.shape[1] != len(parameter_names):
        raise ValueError("theta/name dimensions do not align")
    if len(x) != len(theta) or len(groups) != len(theta) or len(ids) != len(theta):
        raise ValueError("features, theta, groups and simulation IDs must align")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(
            "simulation IDs must be unique per row; represent repeated views with unique simulation IDs and a shared parent group ID"
        )
    if not np.isfinite(x).all() or not np.isfinite(theta).all():
        raise ValueError("features/theta must be finite")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter_names must be unique")
    if target_name not in parameter_names:
        raise ValueError(
            f"target {target_name!r} not in parameter_names {parameter_names}; use the exact documented column name and units"
        )
    if len(prior_low) != len(parameter_names) or len(prior_high) != len(parameter_names):
        raise ValueError("prior bounds must match parameter columns")
    bounds = BoundedParameterTransform(np.asarray(prior_low), np.asarray(prior_high))
    transformed = bounds.forward(theta)
    target_idx = parameter_names.index(target_name)
    split = stratified_group_train_val_test_split(
        groups, theta[:, target_idx], fractions=fractions, seed=seed, n_bins=6
    )
    # Fail closed if stratification leaves too little mass or parameter support in training.
    y_target = theta[:, target_idx]
    if np.unique(y_target[split["train"]]).size < 3:
        raise ValueError(
            "training partition has fewer than three distinct target values; acquire more groups or change the split"
        )
    if min(len(split[k]) for k in split) < 2:
        raise ValueError("each split needs at least two rows")
    # The outer validation partition is for model/architecture selection only. Inner early stopping
    # uses groups carved from the outer training partition, otherwise selecting HPO candidates on the
    # same simulations that chose each epoch would bias the reported selection score.
    outer_train_idx = split["train"]
    inner_local = group_train_validation_split(
        groups[outer_train_idx], validation_fraction=0.12, seed=seed + 1009
    )
    fit_idx = outer_train_idx[inner_local["train"]]
    early_idx = outer_train_idx[inner_local["validation"]]
    # Fit every preprocessing statistic on the actual gradient-fit groups only. The inner
    # early-stopping groups must not affect means/scales, and the outer validation/test
    # groups remain fully held out.
    feature_scaler = Standardizer().fit(x[fit_idx])
    x_scaled = feature_scaler.transform(x)
    target_scaler = Standardizer().fit(transformed[fit_idx])
    y_scaled = target_scaler.transform(transformed)
    model, train_report = fit_conditional_mdn(
        x_scaled[fit_idx],
        y_scaled[fit_idx],
        x_scaled[early_idx],
        y_scaled[early_idx],
        config=config,
    )
    n_val = min(int(n_validation_observations), len(split["validation"]))
    if n_val < 1 or posterior_samples < 2:
        raise ValueError("need validation observations and at least 2 posterior draws")
    val_idx = split["validation"][:n_val]
    model_device = (config or MDNConfig()).device
    bounds_state = bounds.state_dict()
    samples = draw_physical_posterior(
        model,
        x[val_idx],
        n_samples=posterior_samples,
        seed=seed,
        device=model_device,
        feature_state=feature_scaler.state_dict(),
        target_state=target_scaler.state_dict(),
        bounds_state=bounds_state,
    )
    val_metrics = posterior_diagnostic_report(samples, theta[val_idx], parameter_names)
    # This check is performed after all validation-based choices. The final test is not otherwise evaluated.
    test_metrics = None
    test_samples = None
    test_idx = np.asarray([], dtype=int)
    if evaluate_test_once:
        n_test = min(int(n_validation_observations), len(split["test"]))
        if n_test < 1:
            raise ValueError("test split empty")
        test_idx = split["test"][:n_test]
        test_samples = draw_physical_posterior(
            model,
            x[test_idx],
            n_samples=posterior_samples,
            seed=seed + 100000,
            device=model_device,
            feature_state=feature_scaler.state_dict(),
            target_state=target_scaler.state_dict(),
            bounds_state=bounds_state,
        )
        test_metrics = posterior_diagnostic_report(test_samples, theta[test_idx], parameter_names)
    report = {
        "schema_version": "1.0",
        "experiment": "conditional_mdn_summary_posterior",
        "dataset_path": str(path.resolve()),
        "dataset_sha256": _sha256(path),
        "dataset_metadata": metadata,
        "physics_level": physics_level,
        "n_rows": len(theta),
        "n_features": int(x.shape[1]),
        "parameter_names": parameter_names,
        "target_name": target_name,
        "prior_low": list(map(float, prior_low)),
        "prior_high": list(map(float, prior_high)),
        "split_rows": {k: int(len(v)) for k, v in split.items()},
        "split_group_counts": {k: int(len(np.unique(groups[v]))) for k, v in split.items()},
        "split_group_hashes": {k: split_fingerprint(groups, v) for k, v in split.items()},
        "inner_training_rows": {"fit": int(len(fit_idx)), "early_stopping": int(len(early_idx))},
        "inner_training_group_hashes": {
            "fit": split_fingerprint(groups, fit_idx),
            "early_stopping": split_fingerprint(groups, early_idx),
        },
        "model_selection_validation_is_not_used_for_early_stopping": True,
        "preprocessing": {
            "features": feature_scaler.state_dict(),
            "parameters_after_logit": target_scaler.state_dict(),
            "bounded_transform": bounds_state,
            "fit_on": "inner gradient-fit groups only",
        },
        "training": train_report,
        "validation_posterior_metrics": val_metrics,
        "test_evaluated_once": bool(evaluate_test_once),
        "final_test_posterior_metrics": test_metrics,
        "scientific_approval": False,
        "claim_guard": "A calibrated synthetic/controlled benchmark is a prerequisite. This report does not establish a neutrino-mass constraint.",
    }
    import torch

    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": (config or MDNConfig()).to_dict(),
        "input_dim": int(x.shape[1]),
        "target_dim": int(theta.shape[1]),
        "parameter_names": parameter_names,
        "feature_scaler": feature_scaler.state_dict(),
        "target_scaler": target_scaler.state_dict(),
        "bounded_transform": bounds_state,
        "dataset_sha256": report["dataset_sha256"],
        "physics_level": physics_level,
    }
    # Only load checkpoints produced by trusted local runs and matching the recorded software versions.
    torch.save(ckpt, out / "mdn_checkpoint.pt")
    np.savez_compressed(
        out / "validation_posterior_samples.npz",
        samples=samples,
        truth=theta[val_idx],
        simulation_id=ids[val_idx],
        group_id=groups[val_idx],
        row_indices=val_idx,
    )
    if evaluate_test_once and test_samples is not None:
        np.savez_compressed(
            out / "heldout_posterior_samples.npz",
            samples=test_samples,
            truth=theta[test_idx],
            simulation_id=ids[test_idx],
            group_id=groups[test_idx],
            row_indices=test_idx,
        )
    (out / "mdn_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
