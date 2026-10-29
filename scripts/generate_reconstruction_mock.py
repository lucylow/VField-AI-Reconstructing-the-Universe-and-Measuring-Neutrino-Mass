#!/usr/bin/env python3
"""Generate a deterministic MOCK paired-field NPZ for software smoke tests only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/reconstruction_mock.npz")
    parser.add_argument("--n-sims", type=int, default=48)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.n_sims < 6 or args.grid_size < 4:
        parser.error("n-sims must be >=6 and grid-size >=4")
    rng = np.random.default_rng(args.seed)
    shape = (args.n_sims, args.grid_size, args.grid_size, args.grid_size)
    truth = np.empty(shape, dtype=np.float32)
    galaxy = np.empty_like(truth)
    lensing = np.empty((args.n_sims, args.grid_size, args.grid_size), dtype=np.float32)
    cmb = np.empty((args.n_sims, 2), dtype=np.float32)
    groups, sim_ids = [], []
    for i in range(args.n_sims):
        if i % 2 == 0:
            field = rng.normal(size=shape[1:]).astype(np.float32)
            field -= field.mean()
            field /= max(float(field.std()), 1e-8)
        truth[i] = field
        galaxy[i] = field + rng.normal(0, 0.35, shape[1:]).astype(np.float32)
        lensing[i] = field.mean(axis=2) + rng.normal(0, 0.5, (args.grid_size, args.grid_size)).astype(np.float32)
        cmb[i] = [float(field.mean() + rng.normal(0, 0.1)), float(field.std() + rng.normal(0, 0.1))]
        groups.append(f"mock-parent-{i // 2:04d}")
        sim_ids.append(f"mock-sim-{i:05d}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, initial_field=truth, galaxy_overdensity=galaxy,
                        weak_lensing_map=lensing, cmb_features=cmb,
                        simulation_id=np.asarray(sim_ids), group_id=np.asarray(groups))
    out.with_suffix(".json").write_text(json.dumps({
        "schema_version": "1.0", "dataset_name": "synthetic-reconstruction-smoke",
        "data_type": "synthetic software mock", "license": "generated locally; no external data",
        "is_synthetic": True, "physics_level": "toy", "random_seed": args.seed,
        "n_simulations": args.n_sims, "grid_size": args.grid_size,
        "scientific_warning": "Synthetic signal/noise fixture for software testing only; not a cosmological simulation or observation.",
        "array_units": {"initial_field": "arbitrary dimensionless", "galaxy_overdensity": "arbitrary dimensionless",
                        "weak_lensing_map": "arbitrary dimensionless", "cmb_features": "arbitrary dimensionless"},
        "coordinate_system": "grid indices; no cosmological coordinate mapping",
        "coordinate_unit": "dimensionless grid cells",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(out), "n_sims": args.n_sims, "grid_size": args.grid_size,
                      "is_synthetic": True, "scientific_approval": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
