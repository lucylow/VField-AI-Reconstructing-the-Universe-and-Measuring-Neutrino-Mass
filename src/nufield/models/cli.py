"""CLI for explainable model-family selection."""

from __future__ import annotations

import argparse
import json

from nufield.models.selector import recommend_models


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-model", description="Rank candidate AI models for a declared cosmology task"
    )
    p.add_argument(
        "--task",
        default="parameter_posterior",
        choices=[
            "parameter_posterior",
            "field_reconstruction",
            "survey_summary",
            "multimodal_posterior",
        ],
    )
    p.add_argument("--n-simulations", type=int, required=True)
    p.add_argument("--field-shape", help="comma-separated D,H,W or C,D,H,W")
    p.add_argument("--n-parameters", type=int, default=5)
    p.add_argument("--no-power-spectrum", action="store_true")
    p.add_argument("--has-lensing", action="store_true")
    p.add_argument("--has-mock-domain-shift", action="store_true")
    p.add_argument(
        "--periodic-box",
        action="store_true",
        help="declare periodic uniform-grid simulation box; enables FNO recommendation",
    )
    p.add_argument("--budget", default="laptop", choices=["laptop", "workstation", "hpc"])
    p.add_argument("--no-uncertainty", action="store_true")
    args = p.parse_args(argv)
    shape = tuple(int(s) for s in args.field_shape.split(",")) if args.field_shape else None
    results = recommend_models(
        task=args.task,
        n_simulations=args.n_simulations,
        field_shape=shape,
        n_parameters=args.n_parameters,
        has_power_spectrum=not args.no_power_spectrum,
        has_lensing=args.has_lensing,
        has_mock_domain_shift=args.has_mock_domain_shift,
        budget=args.budget,
        target_uncertainty=not args.no_uncertainty,
        periodic_box=args.periodic_box,
    )
    print(
        json.dumps(
            {
                "heuristic_only": True,
                "candidates": [r.to_dict() for r in results],
                "warning": "Rankings are planning heuristics. Benchmark all viable candidates under matched priors, simulations and splits.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
