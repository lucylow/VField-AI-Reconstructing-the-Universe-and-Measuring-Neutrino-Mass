import pytest

from nufield.config import SimulationConfig, SplitConfig
from nufield.exceptions import ConfigurationError


def test_invalid_grid_rejected():
    with pytest.raises(ConfigurationError):
        SimulationConfig(grid_size=4)


def test_split_must_sum_to_one():
    with pytest.raises(ConfigurationError):
        SplitConfig(0.5, 0.2, 0.2)
