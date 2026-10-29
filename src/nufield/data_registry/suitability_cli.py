"""CLI for a role-based dataset plan that separates training from validation roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.data_registry.suitability import build_suitability_plan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-dataset-plan", description="Plan the data stack by scientific role"
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
    p.add_argument("--budget", default="workstation", choices=["laptop", "workstation", "hpc"])
    p.add_argument(
        "--exact-product-mass-grid-verified",
        action="store_true",
        help="set only after checking actual product metadata and theta labels",
    )
    p.add_argument("--physics-baseline-validated", action="store_true")
    p.add_argument("--survey-observation-model-validated", action="store_true")
    p.add_argument("--output", default="outputs/dataset_suitability_plan.json")
    a = p.parse_args(argv)
    result = build_suitability_plan(
        a.objective,
        budget=a.budget,
        exact_product_mass_grid_verified=a.exact_product_mass_grid_verified,
        physics_baseline_validated=a.physics_baseline_validated,
        survey_observation_model_validated=a.survey_observation_model_validated,
    )
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "primary_mass_training_source": result["primary_mass_training_source"],
                "blocking_conditions": result["blocking_conditions"],
                "scientific_approval": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
