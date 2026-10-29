"""CLI for a cheap P(k)/summary-feature regression benchmark."""

from __future__ import annotations

import argparse
import json

from nufield.training.regression_workflow import train_regression_from_npz
from nufield.training.torch_regression import RegressionTrainConfig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-train-regression", description="Train the summary-feature Gaussian baseline"
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/regression")
    p.add_argument("--observation-key", default="features")
    p.add_argument("--theta-key", default="theta")
    p.add_argument("--simulation-id-key", default="simulation_id")
    p.add_argument("--group-id-key")
    p.add_argument("--parameter-names", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--hidden-dims", default="256,128")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--allow-toy", action="store_true")
    p.add_argument("--allow-synthetic-ids", action="store_true")
    args = p.parse_args(argv)
    params = [s.strip() for s in args.parameter_names.split(",") if s.strip()]
    hidden = tuple(int(s) for s in args.hidden_dims.split(",") if s.strip())
    config = RegressionTrainConfig(
        hidden_dims=hidden,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    report = train_regression_from_npz(
        args.dataset,
        args.output_dir,
        observation_key=args.observation_key,
        theta_key=args.theta_key,
        parameter_names=params,
        simulation_id_key=args.simulation_id_key,
        group_id_key=args.group_id_key,
        allow_synthetic_ids=args.allow_synthetic_ids,
        allow_toy=args.allow_toy,
        seed=args.seed,
        config=config,
    )
    print(
        json.dumps(
            {
                "report": f"{args.output_dir}/summary_regression_report.json",
                "status": report["scientific_status"],
                "n_test": report["split"]["test"]["n_rows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
