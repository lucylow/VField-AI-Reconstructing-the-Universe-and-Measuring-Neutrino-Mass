"""CLI for a compute-bounded conditional MDN model grid."""

from __future__ import annotations

import argparse
import json

from nufield.training.mdn_grid import run_mdn_grid


def _csv(s):
    try:
        return [float(t.strip()) for t in s.split(",") if t.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-mdn-grid", description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/mdn-grid")
    p.add_argument("--parameter-names", required=True)
    p.add_argument("--prior-low", required=True, type=_csv)
    p.add_argument("--prior-high", required=True, type=_csv)
    p.add_argument("--target-name", default="sum_mnu_ev")
    p.add_argument("--observation-key", default="features")
    p.add_argument("--theta-key", default="theta")
    p.add_argument("--group-id-key", default="group_id")
    p.add_argument("--simulation-id-key", default="simulation_id")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-trials", type=int, default=6)
    p.add_argument("--budget", choices=["laptop", "workstation", "hpc"], default="workstation")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--n-validation-observations", type=int, default=32)
    p.add_argument("--posterior-samples", type=int, default=512)
    p.add_argument("--max-coverage-gap", type=float, default=0.25)
    p.add_argument("--minimum-validation-groups", type=int, default=20)
    p.add_argument("--allow-toy", action="store_true")
    a = p.parse_args(argv)
    names = [v.strip() for v in a.parameter_names.split(",") if v.strip()]
    if len(names) != len(a.prior_low) or len(names) != len(a.prior_high):
        p.error("parameter names and prior bounds must have equal length")
    report = run_mdn_grid(
        a.dataset,
        a.output_dir,
        parameter_names=names,
        prior_low=a.prior_low,
        prior_high=a.prior_high,
        target_name=a.target_name,
        observation_key=a.observation_key,
        theta_key=a.theta_key,
        group_id_key=a.group_id_key,
        simulation_id_key=a.simulation_id_key,
        split_seed=a.split_seed,
        max_trials=a.max_trials,
        budget=a.budget,
        epochs=a.epochs,
        patience=a.patience,
        n_validation_observations=a.n_validation_observations,
        posterior_samples=a.posterior_samples,
        max_coverage_gap=a.max_coverage_gap,
        minimum_validation_groups=a.minimum_validation_groups,
        allow_toy=a.allow_toy,
    )
    print(
        json.dumps(
            {
                "report": f"{a.output_dir}/mdn_grid_report.json",
                "trial_count": report["trial_count"],
                "selected_candidate": report["selected_candidate"]["trial_id"]
                if report["selected_candidate"]
                else None,
                "test_used": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
