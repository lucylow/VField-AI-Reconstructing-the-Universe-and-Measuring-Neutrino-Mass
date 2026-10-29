"""Command-line workflows for AI model evaluation and active simulation design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from nufield.ai.active_learning import candidate_pool, select_candidates
from nufield.ai.dataset import SimulationDataset, dataset_quality_report
from nufield.ai.experiment import (
    AIExperimentManifest,
    atomic_json_write,
    fingerprint_array,
    fingerprint_json,
)
from nufield.ai.group_conformal import GroupConformalRegressor
from nufield.ai.grouped_splits import (
    grouped_train_validation_calibration_test_split,
    split_distribution_report,
)
from nufield.ai.metrics import regression_metrics
from nufield.ai.normalization import Standardizer
from nufield.ai.synthetic import make_toy_inverse_problem
from nufield.ai.variance_calibration import VarianceTemperature
from nufield.exceptions import OptionalDependencyError


def _predict_gaussian(model, features, *, batch_size: int = 128):
    import torch

    model.eval()
    means, variances = [], []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            out = model(torch.as_tensor(features[i : i + batch_size], dtype=torch.float32))
            means.append(out["mean"].cpu().numpy())
            variances.append(out["log_variance"].exp().cpu().numpy())
    return np.concatenate(means), np.concatenate(variances)


def run_toy_demo(args) -> dict[str, Any]:
    """Train/evaluate the full AI pipeline on non-cosmological toy data."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from nufield.ai.networks import HeteroscedasticRegressor, ResidualSummaryEncoder
        from nufield.ai.training import TrainConfig, fit_model, set_global_seed
    except ImportError as exc:
        raise OptionalDependencyError(
            "Install nufield-starter[ml] to run `nufield-ai demo`"
        ) from exc
    if args.epochs < 1 or args.n_samples < 120 or args.batch_size < 4:
        raise ValueError("demo requires epochs>=1, n_samples>=120 and batch_size>=4")
    dataset = make_toy_inverse_problem(
        n_samples=args.n_samples,
        seed=args.seed,
        observation_noise=args.noise,
        domain_shift=args.domain_shift,
    )
    split = grouped_train_validation_calibration_test_split(
        dataset.group_ids,
        dataset.parameters[:, 0],
        seed=args.seed,
        n_bins=min(5, max(1, args.n_samples // 40)),
    )
    x_scaler = Standardizer().fit(dataset.observations[split.train_indices])
    y_scaler = Standardizer().fit(dataset.parameters[split.train_indices])
    x = x_scaler.transform(dataset.observations)
    y = y_scaler.transform(dataset.parameters)
    set_global_seed(args.seed)

    def loader(indices, shuffle):
        tx = torch.tensor(x[indices], dtype=torch.float32)
        ty = torch.tensor(y[indices], dtype=torch.float32)
        generator = torch.Generator().manual_seed(args.seed + (1 if shuffle else 2))
        return DataLoader(
            TensorDataset(tx, ty),
            batch_size=min(args.batch_size, len(indices)),
            shuffle=shuffle,
            generator=generator,
            drop_last=False,
        )

    model = HeteroscedasticRegressor(
        ResidualSummaryEncoder(x.shape[1], embedding_dim=64, width=96, depth=2, dropout=0.05),
        embedding_dim=64,
        target_dim=y.shape[1],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        max_epochs=args.epochs,
        patience=max(5, min(18, args.epochs // 3 or 1)),
        scheduler_patience=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        seed=args.seed,
        checkpoint_path=str(output_dir / "best_model.pt"),
        history_path=str(output_dir / "training_history.json"),
    )
    train_result = fit_model(
        model,
        loader(split.train_indices, True),
        loader(split.validation_indices, False),
        config=config,
        train_group_ids=dataset.group_ids[split.train_indices],
        validation_group_ids=dataset.group_ids[split.validation_indices],
        test_group_ids=np.concatenate(
            [dataset.group_ids[split.calibration_indices], dataset.group_ids[split.test_indices]]
        ),
    )

    # Use disjoint subgroups for variance-temperature fitting and conformal radius calibration.
    # Reusing one calibration sample to fit the scale and the conformal quantile can invalidate the
    # ordinary split-conformal exchangeability argument. Keep the locked test completely untouched.
    cal_ix_all = split.calibration_indices
    unique_cal_groups = np.unique(dataset.group_ids[cal_ix_all])
    if len(unique_cal_groups) < 4:
        raise ValueError("demo needs at least four independent calibration groups")
    cal_rng = np.random.default_rng(args.seed + 9181)
    shuffled_groups = unique_cal_groups.copy()
    cal_rng.shuffle(shuffled_groups)
    midpoint = max(2, len(shuffled_groups) // 2)
    temp_groups = set(shuffled_groups[:midpoint].tolist())
    conformal_groups = set(shuffled_groups[midpoint:].tolist())
    temp_ix = np.asarray([i for i in cal_ix_all if dataset.group_ids[i] in temp_groups], dtype=int)
    conf_ix = np.asarray(
        [i for i in cal_ix_all if dataset.group_ids[i] in conformal_groups], dtype=int
    )
    test_ix = split.test_indices
    temp_mean_scaled, temp_var_scaled = _predict_gaussian(train_result.model, x[temp_ix])
    conf_mean_scaled, conf_var_scaled = _predict_gaussian(train_result.model, x[conf_ix])
    test_mean_scaled, test_var_scaled = _predict_gaussian(train_result.model, x[test_ix])
    temp_mean = y_scaler.inverse_transform(temp_mean_scaled)
    conf_mean = y_scaler.inverse_transform(conf_mean_scaled)
    test_mean = y_scaler.inverse_transform(test_mean_scaled)
    target_scale = y_scaler.scale_.reshape(1, -1)
    temp_var = temp_var_scaled * target_scale**2
    conf_var = conf_var_scaled * target_scale**2
    test_var = test_var_scaled * target_scale**2
    temp_targets = dataset.parameters[temp_ix]
    conf_targets = dataset.parameters[conf_ix]
    test_targets = dataset.parameters[test_ix]
    variance_calibration = VarianceTemperature.fit(
        temp_mean, temp_var, temp_targets, minimum=0.25, maximum=8.0, grid_size=121
    )
    conf_var = variance_calibration.apply(conf_var)
    test_var = variance_calibration.apply(test_var)
    conformal = GroupConformalRegressor.fit(
        conf_targets, conf_mean, dataset.group_ids[conf_ix], alpha=0.1, scale=np.sqrt(conf_var)
    )
    lower, upper = conformal.predict(test_mean, scale=np.sqrt(test_var))
    covered = (test_targets >= lower) & (test_targets <= upper)
    report = {
        "project": "nuField AI features demo",
        "dataset_status": "toy",
        "not_a_cosmology_result": True,
        "target_names": list(dataset.parameter_names),
        "split": split_distribution_report(split, dataset.group_ids, dataset.parameters[:, 0]),
        "training": train_result.to_dict(),
        "calibration": {
            "variance_temperature": variance_calibration.to_dict(),
            "group_conformal": conformal.to_dict(),
            "disjoint_calibration_groups": {
                "variance_temperature_groups": sorted(temp_groups),
                "conformal_groups": sorted(conformal_groups),
                "overlap": sorted(temp_groups & conformal_groups),
            },
            "test_interval_coverage_marginal": covered.mean(axis=0).tolist(),
            "test_interval_coverage_simultaneous": float(covered.all(axis=1).mean()),
            "test_interval_mean_width": float((upper - lower).mean()),
        },
        "test_metrics": regression_metrics(test_targets, test_mean, variance=test_var),
        "artifacts": {"checkpoint": "best_model.pt", "training_history": "training_history.json"},
        "fingerprints": {
            "dataset": dataset.fingerprint(),
            "split": fingerprint_json(split.to_dict()),
            "train_features": fingerprint_array(x[split.train_indices]),
        },
        "warning": "Synthetic nonlinear inverse problem only. No cosmological observations, physical parameters, or neutrino-mass constraints are represented.",
    }
    manifest_split = split.to_dict()
    manifest_split["calibration_roles"] = {
        "variance_temperature_groups": sorted(temp_groups),
        "conformal_groups": sorted(conformal_groups),
    }
    manifest = AIExperimentManifest(
        "toy_ai_feature_demo",
        "regression",
        "ResidualSummaryEncoder+HeteroscedasticRegressor",
        args.seed,
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "noise": args.noise,
            "domain_shift": args.domain_shift,
        },
        {"dataset": dataset.fingerprint()},
        manifest_split,
        list(dataset.parameter_names),
        dict(dataset.parameter_units),
        "toy synthetic feature vector",
        "toy",
    )
    report["manifest"] = manifest.to_dict()
    atomic_json_write(output_dir / "ai_demo_report.json", report)
    return report


def _load_dataset(args) -> int:
    dataset = SimulationDataset.load(args.dataset)
    report = dataset_quality_report(dataset)
    report["dataset_fingerprint"] = dataset.fingerprint()
    report["manifest_valid"] = True
    atomic_json_write(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 2


def _acquire(args) -> int:
    low = np.asarray(args.low, float)
    high = np.asarray(args.high, float)
    candidates = candidate_pool(low, high, n_candidates=args.n_candidates, seed=args.seed)
    existing = None if not args.existing else np.load(args.existing, allow_pickle=False)
    batch = select_candidates(
        candidates,
        low,
        high,
        n_select=args.n_select,
        existing_parameters=existing,
        method="maximin",
        seed=args.seed,
    )
    result = batch.to_dict()
    atomic_json_write(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nufield-ai", description="AI evaluation and active simulation design tools"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="train/evaluate a non-cosmological toy AI model")
    demo.add_argument("--output-dir", default="outputs/ai_demo")
    demo.add_argument("--n-samples", type=int, default=600)
    demo.add_argument("--epochs", type=int, default=50)
    demo.add_argument("--batch-size", type=int, default=64)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--noise", type=float, default=0.08)
    demo.add_argument("--domain-shift", type=float, default=0.0)
    audit = sub.add_parser("audit-dataset", help="validate a manifest-backed numerical dataset")
    audit.add_argument("dataset")
    audit.add_argument("--output", default="outputs/ai_dataset_audit.json")
    acquire = sub.add_parser("acquire", help="propose diverse prior-bounded simulation points")
    acquire.add_argument("--low", nargs="+", type=float, required=True)
    acquire.add_argument("--high", nargs="+", type=float, required=True)
    acquire.add_argument("--n-candidates", type=int, default=2048)
    acquire.add_argument("--n-select", type=int, default=10)
    acquire.add_argument("--seed", type=int, default=42)
    acquire.add_argument(
        "--existing", help="optional .npy matrix [N,P] of existing simulation parameters"
    )
    acquire.add_argument("--output", default="outputs/ai_acquisition.json")
    args = parser.parse_args(argv)
    if args.command == "demo":
        report = run_toy_demo(args)
        print(
            json.dumps(
                {
                    "report": str(Path(args.output_dir) / "ai_demo_report.json"),
                    "best_epoch": report["training"]["best_epoch"],
                    "test_mae_mean": report["test_metrics"]["mae_mean"],
                    "dataset_status": "toy",
                    "not_a_cosmology_result": True,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "audit-dataset":
        return _load_dataset(args)
    if args.command == "acquire":
        if len(args.low) != len(args.high):
            parser.error("--low and --high must have the same number of entries")
        if args.n_candidates < args.n_select:
            parser.error("--n-candidates must be >= --n-select")
        return _acquire(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
