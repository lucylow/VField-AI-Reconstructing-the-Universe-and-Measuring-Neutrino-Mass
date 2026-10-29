"""Command-line interface for νField starter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nufield.config import SimulationConfig
from nufield.experiments.runner import run_benchmark, run_demo
from nufield.features.extract import extract_observation_features
from nufield.parameters import CosmologyParameters
from nufield.provenance import TOY_WARNING, write_json
from nufield.randomness import rng_for_id
from nufield.simulation.dataset import create_dataset, load_dataset
from nufield.simulation.pipeline import simulate_observation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nufield", description="νField toy research software starter"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("demo", help="run a complete small synthetic benchmark")
    p.add_argument("--output-dir", default="outputs/demo")
    p.add_argument("--n-sims", type=int, default=160)
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p = subs.add_parser("simulate", help="generate one toy simulation")
    p.add_argument("--output", default="outputs/one-sim.npz")
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--box-size", type=float, default=500.0)
    p.add_argument("--seed", type=int, default=13)
    p = subs.add_parser("make-dataset", help="generate a toy dataset")
    p.add_argument("--output", default="outputs/dataset.npz")
    p.add_argument("--n-sims", type=int, default=160)
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p = subs.add_parser("inspect", help="inspect dataset schema and dimensions")
    p.add_argument("dataset")
    p = subs.add_parser("benchmark", help="fit the kernel baseline and evaluate held-out rows")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/benchmark")
    p.add_argument("--seed", type=int, default=107)
    p = subs.add_parser("validate", help="check dataset data-contract invariants")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        result = run_demo(
            args.output_dir, n_sims=args.n_sims, grid_size=args.grid_size, seed=args.seed
        )
        print(json.dumps(result, indent=2))
    elif args.command == "simulate":
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        params = CosmologyParameters()
        obs = simulate_observation(
            simulation_id="single-toy",
            params=params,
            grid_size=args.grid_size,
            box_size_mpc_h=args.box_size,
            mean_counts_per_cell=4.0,
            rng=rng_for_id(args.seed, "single-toy"),
            projection_noise_std=0.03,
        )
        features = extract_observation_features(
            obs, args.box_size, n_bins=min(10, max(4, args.grid_size))
        )
        np.savez_compressed(
            out,
            initial_field=obs.initial_field,
            counts=obs.counts,
            galaxy_overdensity=obs.galaxy_overdensity,
            toy_projection=obs.toy_projection,
            features=features,
            theta=np.asarray(params.as_array(), np.float32),
        )
        write_json(
            out.with_suffix(".json"),
            {
                "physics_level": "toy",
                "scientific_warning": TOY_WARNING,
                "simulation_id": obs.simulation_id,
                "parameters": params.to_dict(),
                "feature_shape": list(features.shape),
            },
        )
        print(f"Wrote {out} (toy physics only)")
    elif args.command == "make-dataset":
        config = SimulationConfig(
            n_sims=args.n_sims,
            grid_size=args.grid_size,
            seed=args.seed,
            n_k_bins=min(10, max(4, args.grid_size)),
        )
        print(json.dumps(create_dataset(config, args.output), indent=2))
    elif args.command == "inspect":
        data = load_dataset(args.dataset)
        result = {
            "path": args.dataset,
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()},
            "scientific_warning": TOY_WARNING,
        }
        meta = Path(args.dataset).with_suffix(".json")
        if meta.exists():
            result["metadata"] = json.loads(meta.read_text())
        print(json.dumps(result, indent=2))
    elif args.command == "benchmark":
        report = run_benchmark(args.dataset, args.output_dir, seed=args.seed)
        print(
            json.dumps(
                {
                    "report": str(Path(args.output_dir) / "benchmark_report.json"),
                    "n_test": report["n_test"],
                    "warning": report["scientific_interpretation"],
                },
                indent=2,
            )
        )
    elif args.command == "validate":
        data = load_dataset(args.dataset)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "valid": True,
            "n_rows": int(len(data["features"])),
            "feature_shape": list(data["features"].shape),
            "theta_shape": list(data["theta"].shape),
            "unique_ids": int(len(set(data["simulation_id"].tolist()))),
            "all_finite": bool(
                np.isfinite(data["features"]).all() and np.isfinite(data["theta"]).all()
            ),
            "scientific_warning": TOY_WARNING,
        }
        write_json(out / "data_contract_report.json", report)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
