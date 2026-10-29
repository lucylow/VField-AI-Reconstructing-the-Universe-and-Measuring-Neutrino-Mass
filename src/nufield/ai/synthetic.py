"""Small deterministic synthetic regression data for testing AI infrastructure only."""

from __future__ import annotations

from typing import Any

import numpy as np

from nufield.ai.dataset import SimulationDataset


def make_toy_inverse_problem(
    *,
    n_samples: int = 1200,
    seed: int = 42,
    observation_noise: float = 0.08,
    domain_shift: float = 0.0,
) -> SimulationDataset:
    """Create a nonlinear toy inverse problem with known latent parameters.

    The features are not cosmological observables and the labels are not physical parameters. This
    generator exists solely to test training, split handling, uncertainty estimation and report code
    without network access or large simulation assets.
    """
    if n_samples < 40 or observation_noise < 0 or domain_shift < 0:
        raise ValueError("n_samples must be >=40 and noise/shift nonnegative")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-1, 1, size=(n_samples, 3))
    a, b, c = theta.T
    ideal = np.column_stack(
        [
            a,
            b,
            c,
            a * b,
            b * c,
            a * c,
            np.sin(np.pi * a),
            np.cos(np.pi * b),
            np.sin(np.pi * c / 2),
            a**2 - b**2,
            b**2 - c**2,
            np.sin(np.pi * (a + c) / 2),
            np.cos(np.pi * a * b),
            np.exp(-0.5 * c**2) * a,
            np.tanh(a + b - c),
            a * b * c,
        ]
    )
    noise = rng.normal(0, observation_noise, size=ideal.shape)
    noise[:, 8:] += domain_shift * rng.normal(0, 1, size=noise[:, 8:].shape)
    observations = ideal + noise
    group_ids = np.asarray([f"toy-sim-{seed}-{i:07d}" for i in range(n_samples)])
    return SimulationDataset(
        observations.astype(np.float32),
        theta.astype(np.float64),
        group_ids,
        tuple(f"toy_feature_{i:02d}" for i in range(observations.shape[1])),
        ("latent_a", "latent_b", "latent_c"),
        {"latent_a": "dimensionless", "latent_b": "dimensionless", "latent_c": "dimensionless"},
        {
            "simulator_status": "toy",
            "generator": "make_toy_inverse_problem",
            "seed": int(seed),
            "observation_noise": float(observation_noise),
            "domain_shift": float(domain_shift),
            "not_for_cosmological_claims": True,
        },
    )


def split_array(dataset: SimulationDataset, indices: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return feature/target arrays by indices after validating the toy dataset schema."""
    dataset.require_valid()
    ix = np.asarray(indices, int)
    if ix.ndim != 1 or np.any(ix < 0) or np.any(ix >= len(dataset.observations)):
        raise ValueError("indices must be a valid one-dimensional row index array")
    return dataset.observations[ix], dataset.parameters[ix]
