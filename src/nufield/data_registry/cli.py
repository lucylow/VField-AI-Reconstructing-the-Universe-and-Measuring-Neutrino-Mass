"""Dataset decision-support CLI; recommendation scores are heuristics, not approvals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nufield.benchmarking.mass_response import paired_mass_response
from nufield.data_registry.assessment import assess_arrays
from nufield.data_registry.cards import load_dataset_card
from nufield.data_registry.catalog import DATASETS
from nufield.data_registry.quality import audit_numeric_dataset
from nufield.data_registry.resource_estimator import estimate_storage, estimate_training_memory
from nufield.data_registry.science_readiness import assess_science_readiness
from nufield.data_registry.selector import recommend_datasets


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-data", description="Choose and audit νField data sources"
    )
    sub = p.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recommend", help="rank sources for a research objective")
    rec.add_argument(
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
    rec.add_argument("--budget", default="laptop", choices=["laptop", "workstation", "hpc"])
    rec.add_argument("--top-k", type=int, default=6)
    rec.add_argument("--output")
    cat = sub.add_parser("catalog", help="export all curated source records")
    cat.add_argument("--output")
    ass = sub.add_parser("assess-npz", help="assess observations/theta arrays before training")
    ass.add_argument("dataset")
    ass.add_argument("--observation-key", required=True)
    ass.add_argument("--theta-key", default="theta")
    ass.add_argument("--simulation-id-key", default="simulation_id")
    ass.add_argument(
        "--parameter-names", required=True, help="comma-separated canonical column order"
    )
    ass.add_argument("--mass-parameter-name", default="sum_mnu_ev")
    ass.add_argument("--output")
    audit = sub.add_parser(
        "audit-npz", help="audit array hygiene, labels, IDs, and metadata before model selection"
    )
    audit.add_argument("dataset")
    audit.add_argument("--observation-key", required=True)
    audit.add_argument("--theta-key", default="theta")
    audit.add_argument("--simulation-id-key", default="simulation_id")
    audit.add_argument("--group-id-key", default="group_id")
    audit.add_argument(
        "--parameter-names", required=True, help="comma-separated canonical column order"
    )
    audit.add_argument("--dataset-id", default="unspecified")
    audit.add_argument(
        "--physics-level", default="unknown", choices=["toy", "prototype", "validated", "unknown"]
    )
    audit.add_argument("--source-release", default="")
    audit.add_argument("--units", default="")
    audit.add_argument("--output")
    resp = sub.add_parser(
        "mass-response", help="compare a feature matrix across matched neutrino-mass simulations"
    )
    resp.add_argument("dataset")
    resp.add_argument("--feature-key", required=True, help="2-D P(k)/summary feature matrix key")
    resp.add_argument("--mass-key", default="sum_mnu_ev")
    resp.add_argument(
        "--mass-column", type=int, default=0, help="column if mass is stored inside theta"
    )
    resp.add_argument(
        "--phase-id-key", required=True, help="verified matched-phase / initial-condition ID"
    )
    resp.add_argument(
        "--strata-id-key", help="optional ID for matching all other cosmology/nuisance choices"
    )
    resp.add_argument("--low-mass-ev", type=float, required=True)
    resp.add_argument("--high-mass-ev", type=float, required=True)
    resp.add_argument("--output", required=True)
    ready = sub.add_parser(
        "science-readiness",
        help="check a dataset card and parameter arrays against a conservative science checklist",
    )
    ready.add_argument("--card", required=True)
    ready.add_argument("--dataset", help="NPZ override; otherwise uses card.local_path")
    ready.add_argument("--theta-key", help="override card.parameter_key")
    ready.add_argument("--simulation-id-key", default="simulation_id")
    ready.add_argument("--group-id-key", default="group_id")
    ready.add_argument("--output")
    storage = sub.add_parser("estimate-storage", help="estimate raw/derived field storage")
    storage.add_argument("--n-simulations", type=int, required=True)
    storage.add_argument(
        "--shape", required=True, help="comma-separated dimensions per simulation, e.g. C,D,H,W"
    )
    storage.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64", "int16", "int32", "int64"],
    )
    storage.add_argument("--derived-copies", type=int, default=3)
    storage.add_argument("--compression-factor", type=float, default=1.0)
    memory = sub.add_parser(
        "estimate-memory", help="rough training memory estimate for a 3-D model"
    )
    memory.add_argument("--model", choices=["cnn3d", "fno3d"], required=True)
    memory.add_argument("--field-shape", required=True, help="C,D,H,W")
    memory.add_argument("--batch-size", type=int, default=2)
    memory.add_argument("--width", type=int, default=16)
    memory.add_argument("--blocks", type=int, default=4)
    memory.add_argument(
        "--dtype", default="float32", choices=["float16", "bfloat16", "float32", "float64"]
    )
    args = p.parse_args(argv)
    if args.command == "recommend":
        output = {
            "objective": args.objective,
            "budget": args.budget,
            "ranking_is_heuristic": True,
            "recommendations": [
                x.to_dict()
                for x in recommend_datasets(
                    args.objective, max_results=args.top_k, compute_budget=args.budget
                )
            ],
        }
    elif args.command == "catalog":
        output = {
            "sources": [d.to_dict() for d in DATASETS.values()],
            "warning": "Source metadata is curated planning guidance; verify actual product documentation before use.",
        }
    elif args.command == "assess-npz":
        with np_load(args.dataset) as z:
            for key in (args.observation_key, args.theta_key):
                if key not in z:
                    raise KeyError(f"{key!r} missing; available={z.files}")
            x = z[args.observation_key]
            y = z[args.theta_key]
            ids = z[args.simulation_id_key] if args.simulation_id_key in z else None
        assessment = assess_arrays(
            x,
            y,
            parameter_names=[s.strip() for s in args.parameter_names.split(",") if s.strip()],
            simulation_ids=ids,
            mass_parameter_name=args.mass_parameter_name,
        )
        output = assessment.to_dict()
    elif args.command == "audit-npz":
        with np_load(args.dataset) as z:
            for key in (args.observation_key, args.theta_key):
                if key not in z:
                    raise KeyError(f"{key!r} missing; available={z.files}")
            x = z[args.observation_key]
            y = z[args.theta_key]
            ids = z[args.simulation_id_key] if args.simulation_id_key in z else None
            groups = z[args.group_id_key] if args.group_id_key in z else None
        audit_report = audit_numeric_dataset(
            x,
            y,
            parameter_names=[s.strip() for s in args.parameter_names.split(",") if s.strip()],
            dataset_id=args.dataset_id,
            simulation_ids=ids,
            group_ids=groups,
            metadata={
                "physics_level": args.physics_level,
                "source_release": args.source_release,
                "units": args.units,
            },
        )
        output = audit_report.to_dict()
    elif args.command == "science-readiness":
        card = load_dataset_card(args.card)
        dataset_path = Path(args.dataset or card.local_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"dataset not found: {dataset_path}; provide --dataset or fix card.local_path"
            )
        with np_load(str(dataset_path)) as z:
            theta_key = args.theta_key or card.parameter_key
            if theta_key not in z:
                raise KeyError(f"{theta_key!r} missing; available={z.files}")
            theta = np.asarray(z[theta_key])
            ids = np.asarray(z[args.simulation_id_key]) if args.simulation_id_key in z else None
            groups = np.asarray(z[args.group_id_key]) if args.group_id_key in z else None
        report = assess_science_readiness(
            card,
            theta=theta,
            observed_parameter_names=card.parameter_names,
            simulation_ids=ids,
            group_ids=groups,
        )
        output = report.to_dict()
        output["dataset_path"] = str(dataset_path.resolve())
    elif args.command == "estimate-storage":
        shape = tuple(int(x.strip()) for x in args.shape.split(",") if x.strip())
        output = estimate_storage(
            args.n_simulations,
            shape,
            dtype=args.dtype,
            derived_copies=args.derived_copies,
            compression_factor=args.compression_factor,
        ).to_dict()
    elif args.command == "estimate-memory":
        shape = tuple(int(x.strip()) for x in args.field_shape.split(",") if x.strip())
        output = estimate_training_memory(
            args.model,
            batch_size=args.batch_size,
            field_shape=shape,
            width=args.width,
            n_blocks=args.blocks,
            dtype=args.dtype,
        ).to_dict()
    elif args.command == "mass-response":
        with np_load(args.dataset) as z:
            for key in (args.feature_key, args.phase_id_key):
                if key not in z:
                    raise KeyError(f"{key!r} missing; available={z.files}")
            x = z[args.feature_key]
            if args.mass_key in z:
                m = z[args.mass_key]
            elif "theta" in z:
                theta = z["theta"]
                if theta.ndim != 2 or args.mass_column < 0 or args.mass_column >= theta.shape[1]:
                    raise ValueError("theta must be [N,P] and mass-column index must be in bounds")
                m = theta[:, args.mass_column]
            else:
                raise KeyError(f"mass key {args.mass_key!r} absent and no theta array available")
            phase = z[args.phase_id_key]
            strata = (
                z[args.strata_id_key] if args.strata_id_key and args.strata_id_key in z else None
            )
            if args.strata_id_key and strata is None:
                raise KeyError(f"strata key {args.strata_id_key!r} missing")
        report = paired_mass_response(
            x,
            m,
            phase,
            low_mass_ev=args.low_mass_ev,
            high_mass_ev=args.high_mass_ev,
            strata_ids=strata,
        )
        report_dict = report.to_dict()
        report_dict["dataset_path"] = str(Path(args.dataset).resolve())
        report_dict["feature_key"] = args.feature_key
        report_dict["phase_id_key"] = args.phase_id_key
        output = report_dict
    else:  # pragma: no cover - argparse restricts the command set
        raise ValueError(f"Unsupported command: {args.command}")
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    if getattr(args, "output", None):
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    print(rendered)
    return 0


def np_load(path: str):
    return np.load(path, allow_pickle=False)


if __name__ == "__main__":
    raise SystemExit(main())
