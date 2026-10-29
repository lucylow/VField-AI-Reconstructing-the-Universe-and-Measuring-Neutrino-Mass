import numpy as np

from nufield.parameters import CosmologyParameters
from nufield.simulation.pipeline import simulate_observation


def test_simulation_outputs_finite_and_aligned():
    obs = simulate_observation(
        simulation_id="x",
        params=CosmologyParameters(),
        grid_size=8,
        box_size_mpc_h=300,
        mean_counts_per_cell=3,
        rng=np.random.default_rng(42),
    )
    assert obs.initial_field.shape == (8, 8, 8)
    assert obs.counts.shape == obs.galaxy_overdensity.shape
    assert obs.toy_projection.shape == (8, 8)
    assert np.isfinite(obs.initial_field).all()
    assert np.all(obs.counts >= 0)
