"""End-to-end grouped-split summary-statistic Gaussian baseline workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nufield.data_registry.splits import stratified_group_train_val_test_split
from nufield.training.torch_regression import RegressionTrainConfig, fit_gaussian_regressor


def train_regression_from_npz(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    observation_key: str,
    theta_key: str,
    parameter_names: list[str],
    simulation_id_key: str = "simulation_id",
    group_id_key: str | None = None,
    allow_synthetic_ids: bool = False,
    allow_toy: bool = False,
    seed: int = 42,
    config: RegressionTrainConfig | None = None,
) -> dict:
    path = Path(dataset_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    physics_level = str(metadata.get("physics_level", "unknown"))
    if physics_level == "toy" and not allow_toy:
        raise ValueError("dataset is marked toy; pass allow_toy only for a software smoke test")
    with np.load(path, allow_pickle=False) as z:
        for key in (observation_key, theta_key):
            if key not in z:
                raise KeyError(f"{key!r} absent from dataset; keys={z.files}")
        x = np.asarray(z[observation_key], dtype=np.float32)
        y = np.asarray(z[theta_key], dtype=np.float32)
        if simulation_id_key in z:
            ids = np.asarray(z[simulation_id_key]).astype(str)
        elif allow_synthetic_ids:
            ids = np.asarray([f"row-{i}" for i in range(len(y))])
        else:
            raise KeyError(
                "simulation IDs missing; synthetic IDs allowed only in a documented smoke test"
            )
        if group_id_key and group_id_key not in z:
            raise KeyError(f"group key {group_id_key!r} absent")
        groups = np.asarray(z[group_id_key]).astype(str) if group_id_key else ids.copy()
    if x.ndim != 2:
        raise ValueError(
            "regression baseline expects a 2-D summary-feature matrix; use features/P(k), not raw 3-D fields"
        )
    if y.ndim != 2 or y.shape[1] != len(parameter_names):
        raise ValueError("theta/parameter_names shape mismatch")
    if any(len(z) != len(y) for z in (x, ids, groups)):
        raise ValueError("row counts do not align")
    mass_idx = (
        parameter_names.index("sum_mnu_ev")
        if "sum_mnu_ev" in parameter_names
        else (parameter_names.index("Mnu") if "Mnu" in parameter_names else 0)
    )
    split = stratified_group_train_val_test_split(
        groups,
        y[:, mass_idx],
        fractions=(0.65, 0.10, 0.25),
        seed=seed,
        n_bins=6,
        require_group_constant=False,
    )
    fit = fit_gaussian_regressor(
        x[split["train"]],
        y[split["train"]],
        x[split["validation"]],
        y[split["validation"]],
        parameter_names=parameter_names,
        config=config or RegressionTrainConfig(seed=seed),
        output_dir=out / "model",
    )
    pred = fit.predict(x[split["test"]])
    mean = pred["mean"]
    var = np.maximum(pred["variance"], 1e-12)
    truth = y[split["test"]]
    rows = []
    for j, name in enumerate(parameter_names):
        residual = truth[:, j] - mean[:, j]
        sd = np.sqrt(var[:, j])
        rows.append(
            {
                "parameter": name,
                "bias": float(np.mean(mean[:, j] - truth[:, j])),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "coverage_68_approx": float(np.mean(np.abs(residual) <= 1.0 * sd)),
                "coverage_95_approx": float(np.mean(np.abs(residual) <= 1.96 * sd)),
                "mean_predicted_sd": float(sd.mean()),
                "truth_std": float(truth[:, j].std()),
            }
        )
    report = {
        "model": "heteroscedastic Gaussian MLP summary baseline",
        "dataset_path": str(path.resolve()),
        "physics_level": physics_level,
        "scientific_status": "TOY_SOFTWARE_SMOKE_TEST_ONLY"
        if physics_level == "toy"
        else "UNVALIDATED_NEURAL_BASELINE",
        "observation_key": observation_key,
        "theta_key": theta_key,
        "parameter_names": parameter_names,
        "split": {
            k: {
                "n_rows": int(len(v)),
                "n_groups": int(len(set(groups[v].tolist()))),
                "mnu_min": float(y[v, mass_idx].min()),
                "mnu_max": float(y[v, mass_idx].max()),
            }
            for k, v in split.items()
        },
        "test_metrics": rows,
        "train_report": fit.report,
        "warnings": [
            "The diagonal Gaussian head is not a general posterior density estimator.",
            "Empirical test coverage uses finite held-out groups; report binomial uncertainty.",
            "Toy physics outputs have no scientific interpretation.",
        ],
    }
    (out / "summary_regression_report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        out / "test_predictions.npz",
        truth=truth,
        mean=mean,
        variance=var,
        simulation_id=ids[split["test"]],
        group_id=groups[split["test"]],
    )
    return report
