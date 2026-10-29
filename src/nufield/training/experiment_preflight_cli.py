"""CLI wrapper for the complete data/model preflight report."""

from __future__ import annotations

import argparse
import json

from nufield.training.experiment_preflight import run_experiment_preflight


def _floats(text):
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def _ints(text):
    try:
        return tuple(int(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated dimensions") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-preflight",
        description="Inspect dataset contract and block unsuitable model choices",
    )
    p.add_argument("dataset")
    p.add_argument("--observation-key", required=True)
    p.add_argument("--theta-key", default="theta")
    p.add_argument(
        "--parameter-names", required=True, help="comma-separated columns in exact theta order"
    )
    p.add_argument("--simulation-id-key", default="simulation_id")
    p.add_argument("--group-id-key", default="group_id")
    p.add_argument(
        "--observation-kind",
        choices=["spectrum", "field3d", "multimodal", "catalog", "image2d"],
        required=True,
    )
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--release", required=True)
    p.add_argument(
        "--physics-level", choices=["toy", "prototype", "validated", "unknown"], default="unknown"
    )
    p.add_argument("--field-shape", type=_ints)
    p.add_argument("--prior-low", type=_floats, required=True)
    p.add_argument("--prior-high", type=_floats, required=True)
    p.add_argument("--budget", choices=["laptop", "workstation", "hpc"], default="workstation")
    p.add_argument("--periodic-box", action="store_true")
    p.add_argument("--source-schema-verified", action="store_true")
    p.add_argument("--lensing-key")
    p.add_argument("--output", default="outputs/preflight/report.json")
    a = p.parse_args(argv)
    report = run_experiment_preflight(
        a.dataset,
        observation_key=a.observation_key,
        theta_key=a.theta_key,
        parameter_names=[s.strip() for s in a.parameter_names.split(",") if s.strip()],
        simulation_id_key=a.simulation_id_key,
        group_id_key=a.group_id_key,
        observation_kind=a.observation_kind,
        dataset_id=a.dataset_id,
        release=a.release,
        physics_level=a.physics_level,
        field_shape=a.field_shape,
        prior_low=a.prior_low,
        prior_high=a.prior_high,
        budget=a.budget,
        periodic_box=a.periodic_box,
        source_schema_verified=a.source_schema_verified,
        has_lensing=bool(a.lensing_key),
        lensing_key=a.lensing_key,
        output_path=a.output,
    )
    print(
        json.dumps(
            {
                "output": report["output_path"],
                "n_rows": report["n_rows"],
                "n_groups": report["n_parent_groups"],
                "can_run_software_smoke_test": report["can_run_software_smoke_test"],
                "physical_claim_allowed": False,
                "blocking_errors": report["blocking_error_codes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
