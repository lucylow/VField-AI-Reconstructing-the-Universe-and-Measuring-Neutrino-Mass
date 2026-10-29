"""Command line research planner that combines data-source and model-family decision support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.research_planner import plan_research_project


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-plan", description="Generate a gate-driven data/model research plan"
    )
    p.add_argument(
        "--objective",
        default="neutrino_mass_inference",
        choices=[
            "neutrino_mass_inference",
            "field_reconstruction",
            "survey_validation",
            "baryonic_robustness",
            "multimodal_extension",
            "software_smoke_test",
        ],
    )
    p.add_argument("--n-simulations", type=int, default=1500)
    p.add_argument(
        "--field-shape", default="32,32,32", help="D,H,W; use empty string if fields unavailable"
    )
    p.add_argument("--budget", choices=["laptop", "workstation", "hpc"], default="workstation")
    p.add_argument("--periodic-box", action="store_true")
    p.add_argument("--has-lensing", action="store_true")
    p.add_argument(
        "--mass-grid-verified",
        action="store_true",
        help="only after source metadata and actual parameter arrays are inspected",
    )
    p.add_argument("--no-uncertainty", action="store_true")
    p.add_argument("--output", default="outputs/research_plan.json")
    args = p.parse_args(argv)
    shape = (
        tuple(int(x.strip()) for x in args.field_shape.split(",") if x.strip())
        if args.field_shape.strip()
        else None
    )
    if shape and len(shape) not in {3, 4}:
        p.error("--field-shape must contain three or four positive dimensions")
    plan = plan_research_project(
        objective=args.objective,
        n_simulations=args.n_simulations,
        field_shape=shape,
        budget=args.budget,
        periodic_box=args.periodic_box,
        has_lensing=args.has_lensing,
        mass_grid_verified=args.mass_grid_verified,
        target_uncertainty=not args.no_uncertainty,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "readiness": plan["readiness"],
                "blockers": plan["blockers"],
                "top_dataset": plan["dataset_candidates"][0]["source_id"],
                "top_model": plan["model_candidates"][0]["name"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
