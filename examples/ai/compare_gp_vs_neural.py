"""Compare a small-data GP inverse emulator with a neural probabilistic regressor.

Run from the repository root after `pip install -e '.[ml,dev]'`::

    python examples/ai/compare_gp_vs_neural.py --n-samples 320 --epochs 20 \
        --output outputs/ai/gp_vs_neural.json

The dataset is synthetic and deliberately unrelated to real cosmological observables. The point of
this example is testing group-aware splits, fair preprocessing, calibration, paired uncertainty on
performance differences, and artifact provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nufield.ai.benchmark import paired_cluster_bootstrap_difference
from nufield.ai.experiment import atomic_json_write
from nufield.ai.gaussian_process_emulator import GaussianProcessEmulator
from nufield.ai.grouped_splits import grouped_train_validation_calibration_test_split
from nufield.ai.metrics import regression_metrics
from nufield.ai.networks import HeteroscedasticRegressor, ResidualSummaryEncoder
from nufield.ai.normalization import Standardizer
from nufield.ai.synthetic import make_toy_inverse_problem
from nufield.ai.training import TrainConfig, fit_model, set_global_seed
from nufield.ai.variance_calibration import VarianceTemperature


def _torch_predict(model, x: np.ndarray, *, batch_size: int = 128):
    model.eval()
    means, variances = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            output = model(torch.as_tensor(x[start : start + batch_size], dtype=torch.float32))
            means.append(output["mean"].cpu().numpy())
            variances.append(output["log_variance"].exp().cpu().numpy())
    return np.concatenate(means), np.concatenate(variances)


def run(args):
    if args.n_samples < 160:
        raise ValueError(
            "use at least 160 samples for useful train/validation/calibration/test partitions"
        )
    set_global_seed(args.seed)
    ds = make_toy_inverse_problem(
        n_samples=args.n_samples, seed=args.seed, observation_noise=args.noise
    )
    split = grouped_train_validation_calibration_test_split(
        ds.group_ids, ds.parameters[:, 0], seed=args.seed, n_bins=5
    )
    tr, va, ca, te = (
        split.train_indices,
        split.validation_indices,
        split.calibration_indices,
        split.test_indices,
    )
    x_scaler = Standardizer().fit(ds.observations[tr])
    y_scaler = Standardizer().fit(ds.parameters[tr])
    x = x_scaler.transform(ds.observations)
    y = y_scaler.transform(ds.parameters)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x[tr]), torch.tensor(y[tr])),
        batch_size=min(args.batch_size, len(tr)),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x[va]), torch.tensor(y[va])),
        batch_size=min(args.batch_size, len(va)),
        shuffle=False,
    )
    neural = HeteroscedasticRegressor(
        ResidualSummaryEncoder(x.shape[1], embedding_dim=64, width=96, depth=2, dropout=0.05),
        64,
        y.shape[1],
    )
    fit = fit_model(
        neural,
        train_loader,
        val_loader,
        config=TrainConfig(
            max_epochs=args.epochs,
            patience=max(5, args.epochs // 3),
            seed=args.seed,
            checkpoint_path=str(Path(args.output).with_suffix(".pt")),
        ),
        train_group_ids=ds.group_ids[tr],
        validation_group_ids=ds.group_ids[va],
        test_group_ids=np.concatenate([ds.group_ids[ca], ds.group_ids[te]]),
    )

    # Exact GP is a small-data reference. This fixed kernel is a declared baseline rather than
    # a hyperparameter winner selected on the test partition.
    gp = GaussianProcessEmulator(
        kernel="matern52", length_scale=0.7, amplitude=1.0, noise_variance=1e-4
    ).fit(x[tr], y[tr])
    gp_cal_mean_scaled, gp_cal_var_scaled = gp.predict(x[ca])
    gp_test_mean_scaled, gp_test_var_scaled = gp.predict(x[te])
    nn_cal_mean_scaled, nn_cal_var_scaled = _torch_predict(fit.model, x[ca])
    nn_test_mean_scaled, nn_test_var_scaled = _torch_predict(fit.model, x[te])

    scale2 = y_scaler.scale_.reshape(1, -1) ** 2
    targets_cal, targets_test = ds.parameters[ca], ds.parameters[te]
    gp_cal_mean = y_scaler.inverse_transform(gp_cal_mean_scaled)
    gp_test_mean = y_scaler.inverse_transform(gp_test_mean_scaled)
    nn_cal_mean = y_scaler.inverse_transform(nn_cal_mean_scaled)
    nn_test_mean = y_scaler.inverse_transform(nn_test_mean_scaled)
    gp_cal_var, gp_test_var = gp_cal_var_scaled * scale2, gp_test_var_scaled * scale2
    nn_cal_var, nn_test_var = nn_cal_var_scaled * scale2, nn_test_var_scaled * scale2

    # Variance temperature is fit only to the held-out calibration partition, never test.
    gp_temp = VarianceTemperature.fit(gp_cal_mean, gp_cal_var, targets_cal, grid_size=101)
    nn_temp = VarianceTemperature.fit(nn_cal_mean, nn_cal_var, targets_cal, grid_size=101)
    gp_test_var = gp_temp.apply(gp_test_var)
    nn_test_var = nn_temp.apply(nn_test_var)
    gp_metrics = regression_metrics(targets_test, gp_test_mean, variance=gp_test_var)
    nn_metrics = regression_metrics(targets_test, nn_test_mean, variance=nn_test_var)
    comparison = paired_cluster_bootstrap_difference(
        targets_test,
        nn_test_mean,
        gp_test_mean,
        ds.group_ids[te],
        metric="rmse",
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    report = {
        "name": "toy_gp_vs_neural_inverse_benchmark",
        "not_a_cosmology_result": True,
        "dataset": {
            "rows": args.n_samples,
            "features": ds.observations.shape[1],
            "targets": list(ds.parameter_names),
            "simulator_status": "toy",
            "fingerprint": ds.fingerprint(),
        },
        "split": {
            "train_rows": len(tr),
            "validation_rows": len(va),
            "calibration_rows": len(ca),
            "test_rows": len(te),
            "disjoint_groups": True,
            "test_used_for_tuning": False,
        },
        "neural": {
            "best_epoch": fit.best_epoch,
            "best_validation_loss": fit.best_val_loss,
            "metrics": nn_metrics,
            "variance_temperature": nn_temp.to_dict(),
        },
        "gaussian_process": {
            "summary": gp.summary(),
            "metrics": gp_metrics,
            "variance_temperature": gp_temp.to_dict(),
        },
        "paired_test_rmse_difference_neural_minus_gp": comparison,
        "interpretation": "A single synthetic benchmark does not establish which model is best for cosmology.",
    }
    out = Path(args.output)
    atomic_json_write(out, report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=320)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--noise", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--output", default="outputs/ai/gp_vs_neural.json")
    args = p.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "output": args.output,
                "neural_rmse": result["neural"]["metrics"]["rmse_mean"],
                "gp_rmse": result["gaussian_process"]["metrics"]["rmse_mean"],
                "not_a_cosmology_result": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
