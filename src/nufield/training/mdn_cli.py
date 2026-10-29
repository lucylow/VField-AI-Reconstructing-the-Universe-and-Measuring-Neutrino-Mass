"""CLI for a grouped-split conditional mixture-density posterior baseline."""

from __future__ import annotations

import argparse
import json

from nufield.models.conditional_mdn import MDNConfig
from nufield.training.mdn_workflow import train_mdn_from_npz


def _csv(value, cast=float):
    try:
        return [cast(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated values") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-train-mdn", description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/mdn")
    p.add_argument("--observation-key", default="features")
    p.add_argument("--theta-key", default="theta")
    p.add_argument("--group-id-key", default="group_id")
    p.add_argument("--simulation-id-key", default="simulation_id")
    p.add_argument("--parameter-names", required=True)
    p.add_argument("--prior-low", required=True)
    p.add_argument("--prior-high", required=True)
    p.add_argument("--target-name", default="sum_mnu_ev")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--n-validation-observations", type=int, default=32)
    p.add_argument("--posterior-samples", type=int, default=512)
    p.add_argument("--allow-toy", action="store_true")
    p.add_argument("--evaluate-test-once", action="store_true")
    a = p.parse_args(argv)
    names = [x.strip() for x in a.parameter_names.split(",") if x.strip()]
    lo = _csv(a.prior_low)
    hi = _csv(a.prior_high)
    if not (len(names) == len(lo) == len(hi)):
        p.error("parameter names and prior bounds must have equal length")
    if not 0.5 <= a.train_fraction <= 0.85:
        p.error("train-fraction must be in [0.5, .85]")
    if not 0.05 <= a.validation_fraction <= 0.3:
        p.error("validation-fraction must be in [0.05, .3]")
    if a.train_fraction + a.validation_fraction >= 1:
        p.error("train+validation fraction must be < 1 to leave a locked test partition")
    cfg = MDNConfig(
        n_components=a.components,
        hidden_dim=a.hidden_dim,
        n_hidden_layers=a.hidden_layers,
        dropout=a.dropout,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        batch_size=a.batch_size,
        max_epochs=a.epochs,
        patience=a.patience,
        seed=a.seed,
        device=a.device,
    )
    report = train_mdn_from_npz(
        a.dataset,
        a.output_dir,
        observation_key=a.observation_key,
        theta_key=a.theta_key,
        group_id_key=a.group_id_key,
        simulation_id_key=a.simulation_id_key,
        parameter_names=names,
        prior_low=lo,
        prior_high=hi,
        target_name=a.target_name,
        seed=a.seed,
        fractions=(
            a.train_fraction,
            a.validation_fraction,
            1 - a.train_fraction - a.validation_fraction,
        ),
        config=cfg,
        n_validation_observations=a.n_validation_observations,
        posterior_samples=a.posterior_samples,
        evaluate_test_once=a.evaluate_test_once,
        allow_toy=a.allow_toy,
    )
    print(
        json.dumps(
            {
                "report": f"{a.output_dir}/mdn_report.json",
                "best_epoch": report["training"]["best_epoch"],
                "validation_metrics_written": True,
                "test_evaluated_once": report["test_evaluated_once"],
                "scientific_approval": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
