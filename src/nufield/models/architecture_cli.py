"""Explain architecture eligibility from a declared observation and compute contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.models.architecture_selector import DataModelContract, eligible_architectures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-architecture", description="Hard-gated model family selection"
    )
    p.add_argument("--n-independent-groups", type=int, required=True)
    p.add_argument(
        "--sample-shape", required=True, help="dimensions per sample, e.g. 32,32,32 or 1,32,32,32"
    )
    p.add_argument(
        "--observation-kind",
        choices=["spectrum", "field3d", "multimodal", "catalog", "image2d"],
        required=True,
    )
    p.add_argument("--target-dimension", type=int, default=1)
    p.add_argument("--has-mass-variation", action="store_true")
    p.add_argument("--mass-grid-verified", action="store_true")
    p.add_argument("--train-val-test-disjoint", action="store_true")
    p.add_argument(
        "--physics-level", choices=["toy", "prototype", "validated", "unknown"], default="unknown"
    )
    p.add_argument(
        "--non-mnu-target",
        action="store_true",
        help="disable neutrino-mass-specific target variation gate when the target is not sum(m_nu)",
    )
    p.add_argument("--regular-grid", action="store_true")
    p.add_argument("--periodic-box", action="store_true")
    p.add_argument("--no-power-spectrum", action="store_true")
    p.add_argument("--has-lensing", action="store_true")
    p.add_argument("--modalities-matched", action="store_true")
    p.add_argument("--prior-declared", action="store_true")
    p.add_argument("--source-schema-verified", action="store_true")
    p.add_argument("--budget", choices=["laptop", "workstation", "hpc"], default="workstation")
    p.add_argument("--output")
    a = p.parse_args(argv)
    shape = tuple(int(s.strip()) for s in a.sample_shape.split(","))
    c = DataModelContract(
        a.n_independent_groups,
        shape,
        a.observation_kind,
        a.target_dimension,
        a.has_mass_variation,
        a.mass_grid_verified,
        a.train_val_test_disjoint,
        a.physics_level,
        not a.non_mnu_target,
        a.regular_grid,
        a.periodic_box,
        not a.no_power_spectrum,
        a.has_lensing,
        a.modalities_matched,
        a.prior_declared,
        a.source_schema_verified,
    )
    result = {
        "contract": c.to_dict(),
        "contract_errors": c.validate(),
        "decisions": [x.to_dict() for x in eligible_architectures(c, budget=a.budget)],
        "decision_rule": "Eligibility is not a performance ranking. Run matched benchmarks and calibration before choosing a winner.",
    }
    if a.output:
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
