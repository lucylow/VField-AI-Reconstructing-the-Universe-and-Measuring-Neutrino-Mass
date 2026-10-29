import pytest

from nufield.exceptions import ConfigurationError
from nufield.parameters import CosmologyParameters


def test_default_parameters_are_valid():
    p = CosmologyParameters()
    assert len(p.as_array()) == 6


def test_out_of_bounds_parameter_rejected():
    with pytest.raises(ConfigurationError):
        CosmologyParameters(sum_mnu_ev=-0.1)


def test_unknown_mapping_key_rejected():
    with pytest.raises(ConfigurationError):
        CosmologyParameters.from_mapping({"nope": 1})
