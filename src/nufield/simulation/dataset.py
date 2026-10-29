"""Dataset generation, persistence, and split assignments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from nufield.config import SimulationConfig, SplitConfig
from nufield.features.extract import extract_observation_features
from nufield.parameters import PARAMETER_NAMES, PARAMETER_UNITS, CosmologyParameters
from nufield.provenance import build_manifest, write_json
from nufield.randomness import rng_for_id
from nufield.simulation.pipeline import simulate_observation


def draw_parameters(rng: np.random.Generator) -> CosmologyParameters:
    """Draw from a broad toy prior used only for software testing."""
    return CosmologyParameters(
        sum_mnu_ev=float(rng.uniform(0.06, 0.30)),
        omega_m=float(rng.uniform(0.27, 0.36)),
        sigma8=float(rng.uniform(0.74, 0.88)),
        h=float(rng.uniform(0.60, 0.75)),
        n_s=float(rng.uniform(0.92, 1.01)),
        bias=float(rng.uniform(0.8, 1.8)),
    )


def create_dataset(config: SimulationConfig, output_path: str | Path) -> dict:
    """Generate dataset NPZ and adjacent JSON metadata using per-ID deterministic RNGs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(config.seed)
    rows = []
    params_rows = []
    ids = []
    projections = []
    fields = []
    counts = []
    for i in range(config.n_sims):
        sid = f"toy-{i:06d}"
        params = draw_parameters(master_rng)
        sim_rng = rng_for_id(config.seed, sid)
        obs = simulate_observation(
            simulation_id=sid,
            params=params,
            grid_size=config.grid_size,
            box_size_mpc_h=config.box_size_mpc_h,
            mean_counts_per_cell=config.mean_counts_per_cell,
            rng=sim_rng,
            projection_noise_std=0.03,
        )
        rows.append(extract_observation_features(obs, config.box_size_mpc_h, config.n_k_bins))
        params_rows.append(params.as_array())
        ids.append(sid)
        projections.append(obs.toy_projection)
        fields.append(obs.galaxy_overdensity)
        counts.append(obs.counts)
    features = np.asarray(rows, dtype=np.float32)
    theta = np.asarray(params_rows, dtype=np.float32)
    # Toy fields are included only for small-grid examples; for production-like grid sizes, use a
    # separate chunked artifact store instead of stacking 3-D arrays in RAM.
    np.savez_compressed(
        output_path,
        features=features,
        theta=theta,
        simulation_id=np.asarray(ids),
        galaxy_overdensity=np.asarray(fields, dtype=np.float32),
        counts=np.asarray(counts, dtype=np.int32),
        toy_projection=np.asarray(projections, dtype=np.float32),
    )
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "schema_version": "1.0",
        "dataset_name": "nufield-toy-dataset",
        "physics_level": "toy",
        "scientific_warning": (
            "Phenomenological toy simulation only; not a neutrino-aware cosmology simulator."
        ),
        "n_simulations": config.n_sims,
        "grid_size": config.grid_size,
        "box_size_mpc_h": config.box_size_mpc_h,
        "mean_counts_per_cell": config.mean_counts_per_cell,
        "random_seed": config.seed,
        "parameter_names": PARAMETER_NAMES,
        "parameter_units": PARAMETER_UNITS,
        "feature_names": [f"log_pk_bin_{j:02d}" for j in range(features.shape[1])],
        "feature_units": "log arbitrary toy power units",
        "simulation_ids": ids,
        "array_shapes": {"features": list(features.shape), "theta": list(theta.shape)},
        "limitations": [
            "No physical Boltzmann transfer functions",
            "No nonlinear gravitational evolution",
            "Toy Poisson tracer, no HOD/RSD/real survey mask",
            "Toy projection is not lensing",
        ],
    }
    write_json(metadata_path, metadata)
    manifest = build_manifest(
        config=asdict(config),
        seed=config.seed,
        dataset_path=output_path,
        extra={"metadata_path": str(metadata_path)},
    )
    write_json(output_path.with_name(output_path.stem + ".manifest.json"), manifest)
    return {
        "dataset_path": str(output_path),
        "metadata_path": str(metadata_path),
        "n_sims": config.n_sims,
        "feature_shape": list(features.shape),
        "theta_shape": list(theta.shape),
    }


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a generated NPZ dataset."""
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    required = {"features", "theta", "simulation_id"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"Dataset missing arrays: {sorted(missing)}")
    if result["features"].ndim != 2 or result["theta"].ndim != 2:
        raise ValueError("features and theta must be two-dimensional")
    if len(result["features"]) != len(result["theta"]) or len(result["theta"]) != len(
        result["simulation_id"]
    ):
        raise ValueError("features/theta/simulation_id row count mismatch")
    if len(set(result["simulation_id"].tolist())) != len(result["simulation_id"]):
        raise ValueError("simulation_id values must be unique")
    if not np.isfinite(result["features"]).all() or not np.isfinite(result["theta"]).all():
        raise ValueError("features and theta must be finite")
    return result


def make_split(ids: np.ndarray, config: SplitConfig, seed: int) -> dict[str, np.ndarray]:
    """Split by complete simulation ID and return integer row indices."""
    ids = np.asarray(ids).astype(str)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("simulation IDs must be unique before splitting")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n = len(ids)
    n_train = int(round(n * config.train_fraction))
    n_val = int(round(n * config.validation_fraction))
    # Ensure the three partitions cover every row exactly once.
    if n_train + n_val >= n:
        n_train = max(1, n_train - 1)
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("dataset too small for requested split")
    return {"train": train, "validation": val, "test": test}
