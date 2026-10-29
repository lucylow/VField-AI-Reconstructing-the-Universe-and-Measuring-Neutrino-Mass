"""CLI for scoring an actual candidate dataset against explicit mass-inference gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nufield.data_registry.cards import load_dataset_card
from nufield.data_registry.decision_gates import evaluate_dataset_gates


def _read_npz(path: str, theta_key: str, id_key: str, group_key: str):
    with np.load(path, allow_pickle=False) as z:
        if theta_key not in z:
            raise KeyError(f"theta key {theta_key!r} not present; keys={z.files}")
        theta = np.asarray(z[theta_key])
        ids = np.asarray(z[id_key]) if id_key in z else None
        groups = np.asarray(z[group_key]) if group_key in z else None
    return theta, ids, groups


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-data-gates", description="Fail-closed dataset eligibility preflight"
    )
    p.add_argument("--card", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--theta-key")
    p.add_argument("--id-key", default="simulation_id")
    p.add_argument("--group-key", default="group_id")
    p.add_argument("--prior-bounds", required=True, help="target prior bounds in eV, e.g. 0.0,0.4")
    p.add_argument(
        "--source-schema-verified",
        action="store_true",
        help="set only after manually inspecting the official release schema",
    )
    p.add_argument("--expected-release")
    p.add_argument("--output", default="outputs/data_gate_report.json")
    args = p.parse_args(argv)
    card = load_dataset_card(args.card)
    key = args.theta_key or card.parameter_key
    bounds = [float(x.strip()) for x in args.prior_bounds.split(",")]
    if len(bounds) != 2:
        p.error("--prior-bounds must be lower,upper")
    theta, ids, groups = _read_npz(args.dataset, key, args.id_key, args.group_key)
    report = evaluate_dataset_gates(
        card,
        theta=theta,
        observed_parameter_names=card.parameter_names,
        simulation_ids=ids,
        group_ids=groups,
        expected_source_release=args.expected_release,
        prior_bounds=tuple(bounds),
        source_schema_verified=args.source_schema_verified,
    )
    out = report.to_dict()
    out["dataset_path"] = str(Path(args.dataset).resolve())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "eligible_for_preliminary_mass_training": report.eligible_for_preliminary_mass_training,
                "eligible_for_observed_data_application": report.eligible_for_observed_data_application,
                "failed_gates": [g.gate_id for g in report.gates if not g.passed],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
