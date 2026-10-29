"""Leakage-resistant training/evaluation of a discrete classical mass-grid posterior baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nufield.benchmarking.classical_mass_grid import ClassicalMassGridModel
from nufield.data_registry.preprocessing import Standardizer
from nufield.data_registry.splits import split_fingerprint, stratified_group_train_val_test_split
from nufield.validation.posterior_diagnostics import posterior_diagnostic_report


def run_classical_mass_grid(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    observation_key: str = "features",
    theta_key: str = "theta",
    parameter_names: list[str],
    target_name: str = "sum_mnu_ev",
    group_id_key: str = "group_id",
    simulation_id_key: str = "simulation_id",
    prior_bounds: tuple[float, float],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    shrinkage: float = 0.15,
    diagonal_floor: float = 1e-5,
    min_groups_per_mass: int = 2,
    n_eval: int = 64,
    posterior_samples: int = 2000,
    evaluate_test_once: bool = False,
    allow_toy: bool = False,
) -> dict[str, Any]:
    """Fit a transparent discrete mass posterior with train groups only; validation is for selection."""
    path = Path(dataset_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    physics = str(metadata.get("physics_level", "unknown"))
    if physics == "toy" and not allow_toy:
        raise ValueError("toy dataset requires allow_toy for software smoke test only")
    with np.load(path, allow_pickle=False) as z:
        for key in (observation_key, theta_key, group_id_key, simulation_id_key):
            if key not in z:
                raise KeyError(f"required key {key!r} missing; keys={z.files}")
        x = np.asarray(z[observation_key], dtype=float)
        theta = np.asarray(z[theta_key], dtype=float)
        groups = np.asarray(z[group_id_key]).astype(str)
        ids = np.asarray(z[simulation_id_key]).astype(str)
    if (
        x.ndim != 2
        or theta.ndim != 2
        or len(x) != len(theta)
        or len(groups) != len(theta)
        or len(ids) != len(theta)
    ):
        raise ValueError("arrays must align; observations/features and theta must be rank 2")
    if len(parameter_names) != theta.shape[1] or len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter names do not match theta columns")
    if target_name not in parameter_names:
        raise ValueError(f"target {target_name!r} not in exact parameter names")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("simulation IDs must be unique row IDs; related views share group_id")
    if not np.isfinite(x).all() or not np.isfinite(theta).all():
        raise ValueError("features and theta must be finite")
    j = parameter_names.index(target_name)
    mass = theta[:, j]
    split = stratified_group_train_val_test_split(
        groups, mass, fractions=fractions, seed=seed, n_bins=6
    )
    scaler = Standardizer().fit(x[split["train"]])
    x_scaled = scaler.transform(x)
    nuisance = np.delete(theta, j, axis=1) if theta.shape[1] > 1 else None
    model = ClassicalMassGridModel.fit(
        x_scaled[split["train"]],
        mass[split["train"]],
        group_ids=groups[split["train"]],
        nuisance_parameters=nuisance[split["train"]] if nuisance is not None else None,
        prior_bounds=prior_bounds,
        shrinkage=shrinkage,
        diagonal_floor=diagonal_floor,
        min_replicates_per_mass=min_groups_per_mass,
        require_fixed_nuisance=True,
    )

    def evaluate(indices: np.ndarray, offset: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        take = indices[: min(len(indices), n_eval)]
        arr = np.stack(
            [
                model.sample_posterior(
                    x_scaled[i], n_samples=posterior_samples, seed=seed + offset + k
                )
                for k, i in enumerate(take)
            ]
        )
        truth = mass[take, None]
        metrics = posterior_diagnostic_report(arr, truth, [target_name])
        return metrics, arr, take

    val_metrics, val_samples, val_idx = evaluate(split["validation"], 1000)
    test_metrics = None
    test_samples = None
    test_idx = np.array([], dtype=int)
    if evaluate_test_once:
        test_metrics, test_samples, test_idx = evaluate(split["test"], 100000)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "schema_version": "1.0",
        "experiment": "classical_discrete_mass_grid_likelihood",
        "dataset_path": str(path.resolve()),
        "dataset_sha256": h,
        "dataset_metadata": metadata,
        "physics_level": physics,
        "target_name": target_name,
        "parameter_names": parameter_names,
        "prior_bounds_ev": list(prior_bounds),
        "split_rows": {k: int(len(v)) for k, v in split.items()},
        "split_group_counts": {k: int(len(np.unique(groups[v]))) for k, v in split.items()},
        "split_group_hashes": {k: split_fingerprint(groups, v) for k, v in split.items()},
        "feature_preprocessing": scaler.state_dict(),
        "model": model.to_dict(),
        "validation_metrics": val_metrics,
        "test_evaluated_once": bool(evaluate_test_once),
        "final_test_metrics": test_metrics,
        "scientific_approval": False,
        "limitations": [
            "posterior is discrete on observed mass levels; no interpolation is performed",
            "assumes nuisance parameters are fixed; if they vary, this baseline rejects the dataset",
            "shared Gaussian covariance and Gaussian summary likelihood are approximations",
            "covariance regularization and feature transforms require sensitivity analysis",
        ],
    }
    np.savez_compressed(
        out / "validation_posterior_samples.npz",
        samples=val_samples,
        truth=mass[val_idx, None],
        simulation_id=ids[val_idx],
        group_id=groups[val_idx],
        row_indices=val_idx,
    )
    if evaluate_test_once and test_samples is not None:
        np.savez_compressed(
            out / "heldout_posterior_samples.npz",
            samples=test_samples,
            truth=mass[test_idx, None],
            simulation_id=ids[test_idx],
            group_id=groups[test_idx],
            row_indices=test_idx,
        )
    np.savez_compressed(
        out / "feature_scaler.npz", mean=np.asarray(scaler.mean_), scale=np.asarray(scaler.scale_)
    )
    (out / "classical_mass_grid_model.json").write_text(
        json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (out / "classical_mass_grid_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
