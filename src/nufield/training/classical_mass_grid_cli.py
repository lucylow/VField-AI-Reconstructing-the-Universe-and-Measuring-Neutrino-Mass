"""CLI for a transparent classical discrete neutrino-mass posterior baseline."""

from __future__ import annotations

import argparse
import json

from nufield.training.classical_mass_grid_workflow import run_classical_mass_grid


def _csv(s):
    try:
        return [float(v.strip()) for v in s.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-classical-mass-grid", description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/classical-grid")
    p.add_argument("--observation-key", default="features")
    p.add_argument("--theta-key", default="theta")
    p.add_argument("--group-id-key", default="group_id")
    p.add_argument("--simulation-id-key", default="simulation_id")
    p.add_argument("--parameter-names", required=True)
    p.add_argument("--target-name", default="sum_mnu_ev")
    p.add_argument("--prior-bounds", required=True, help="target lower,upper in target units")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--shrinkage", type=float, default=0.15)
    p.add_argument("--diagonal-floor", type=float, default=1e-5)
    p.add_argument("--min-groups-per-mass", type=int, default=2)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--posterior-samples", type=int, default=2000)
    p.add_argument("--evaluate-test-once", action="store_true")
    p.add_argument("--allow-toy", action="store_true")
    a = p.parse_args(argv)
    names = [x.strip() for x in a.parameter_names.split(",") if x.strip()]
    bounds = _csv(a.prior_bounds)
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        p.error("--prior-bounds must be lower,upper")
    if (
        not 0.5 <= a.train_fraction <= 0.85
        or not 0.05 <= a.validation_fraction <= 0.3
        or a.train_fraction + a.validation_fraction >= 1
    ):
        p.error("invalid train/validation split fractions")
    r = run_classical_mass_grid(
        a.dataset,
        a.output_dir,
        observation_key=a.observation_key,
        theta_key=a.theta_key,
        parameter_names=names,
        target_name=a.target_name,
        group_id_key=a.group_id_key,
        simulation_id_key=a.simulation_id_key,
        prior_bounds=(bounds[0], bounds[1]),
        seed=a.seed,
        fractions=(
            a.train_fraction,
            a.validation_fraction,
            1 - a.train_fraction - a.validation_fraction,
        ),
        shrinkage=a.shrinkage,
        diagonal_floor=a.diagonal_floor,
        min_groups_per_mass=a.min_groups_per_mass,
        n_eval=a.n_eval,
        posterior_samples=a.posterior_samples,
        evaluate_test_once=a.evaluate_test_once,
        allow_toy=a.allow_toy,
    )
    print(
        json.dumps(
            {
                "report": f"{a.output_dir}/classical_mass_grid_report.json",
                "mass_grid": r["model"]["mass_grid_ev"],
                "test_evaluated_once": r["test_evaluated_once"],
                "scientific_approval": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
