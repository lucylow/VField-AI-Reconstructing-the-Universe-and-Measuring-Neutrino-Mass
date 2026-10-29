"""Composable toy forward model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.parameters import CosmologyParameters
from nufield.simulation.galaxies import counts_to_overdensity, sample_galaxy_counts
from nufield.simulation.lensing import toy_projected_map
from nufield.simulation.linear_field import generate_toy_density_field


@dataclass(slots=True)
class ToyObservation:
    simulation_id: str
    params: CosmologyParameters
    initial_field: np.ndarray
    counts: np.ndarray
    galaxy_overdensity: np.ndarray
    toy_projection: np.ndarray
    physics_level: str = "toy"


def simulate_observation(
    *,
    simulation_id: str,
    params: CosmologyParameters,
    grid_size: int,
    box_size_mpc_h: float,
    mean_counts_per_cell: float,
    rng: np.random.Generator,
    projection_noise_std: float = 0.0,
) -> ToyObservation:
    """Generate one complete toy observation and its known latent field."""
    field = generate_toy_density_field(grid_size, box_size_mpc_h, params, rng)
    counts = sample_galaxy_counts(field, params.bias, mean_counts_per_cell, rng)
    overdensity = counts_to_overdensity(counts)
    projection = toy_projected_map(field, noise_std=projection_noise_std, rng=rng)
    return ToyObservation(simulation_id, params, field, counts, overdensity, projection)
