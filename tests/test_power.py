import numpy as np

from nufield.parameters import CosmologyParameters
from nufield.simulation.power import toy_matter_power, wave_number_grid


def test_power_is_positive_for_positive_k():
    p = CosmologyParameters()
    k = np.array([0.01, 0.1, 1.0])
    out = toy_matter_power(k, p)
    assert np.all(out > 0)


def test_wave_number_grid_shape():
    assert wave_number_grid(8, 100).shape == (8, 8, 8)
