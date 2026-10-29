"""Review one exact data product against a declared scientific contract and local label arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from nufield.data_registry.product_contract import DatasetProductContract, review_dataset_product


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-product-review", description=__doc__)
    p.add_argument(
        "contract_file",
        help="JSON/YAML DatasetProductContract authored after reading official release docs",
    )
    p.add_argument("arrays_file", help="NPZ containing parameter matrix and group IDs")
    p.add_argument("--theta-key", default="theta")
    p.add_argument("--group-id-key", default="group_id")
    p.add_argument("--parameter-names", required=True, help="comma-separated in theta column order")
    p.add_argument(
        "--prior-bounds",
        required=True,
        help="target lower,upper in the target's explicitly documented unit",
    )
    p.add_argument("--min-groups-per-target-value", type=int, default=2)
    p.add_argument("--output", default="outputs/product_review.json")
    a = p.parse_args(argv)
    contract_data = yaml.safe_load(Path(a.contract_file).read_text(encoding="utf-8"))
    contract = DatasetProductContract(**contract_data)
    prior = [float(x.strip()) for x in a.prior_bounds.split(",") if x.strip()]
    names = [x.strip() for x in a.parameter_names.split(",") if x.strip()]
    if len(prior) != 2 or prior[0] >= prior[1]:
        p.error("--prior-bounds must be lower,upper")
    with np.load(a.arrays_file, allow_pickle=False) as z:
        if a.theta_key not in z or a.group_id_key not in z:
            raise KeyError(f"required keys missing; keys={z.files}")
        theta = z[a.theta_key]
        groups = z[a.group_id_key]
    report = review_dataset_product(
        contract,
        theta,
        parameter_names=names,
        group_ids=groups,
        prior_bounds=(prior[0], prior[1]),
        min_groups_per_target_value=a.min_groups_per_target_value,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out.resolve()),
                "eligible_for_declared_role": report["eligible_for_declared_role"],
                "errors": report["errors"],
                "warnings": report["warnings"],
                "scientific_approval": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
