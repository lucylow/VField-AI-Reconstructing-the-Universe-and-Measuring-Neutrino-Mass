import numpy as np
import pytest

from nufield.data_registry.parameter_grid import assert_target_support, inspect_parameter_grid
from nufield.data_registry.target_contract import (
    TargetColumn,
    convert_target,
)


def test_mass_unit_conversion_is_explicit():
    col = TargetColumn("sum_mnu_ev", "Mnu", "meV", "eV", 0.0, 1.0)
    np.testing.assert_allclose(convert_target(np.array([50.0, 100.0]), col), [0.05, 0.1])
    with pytest.raises(ValueError):
        convert_target(np.array([0.1]), TargetColumn("sum_mnu_ev", "Mnu", "unknown", "eV"))


def test_parameter_grid_reports_mass_support_and_constant_column():
    theta = np.column_stack([np.tile([0.1, 0.2, 0.4], 10), np.ones(30)])
    r = inspect_parameter_grid(theta, ["sum_mnu_ev", "constant"])
    assert any("constant" in e for e in r.errors)
    assert any("coarse grid" in w for w in r.warnings)


def test_target_support_checks_prior():
    theta = np.array([[0.1], [0.2], [0.3], [0.4]])
    r = assert_target_support(
        theta, ["sum_mnu_ev"], target_name="sum_mnu_ev", prior_bounds=(0.0, 0.4)
    )
    assert r["eligible_for_preliminary_target_training"]
    bad = assert_target_support(
        theta, ["sum_mnu_ev"], target_name="sum_mnu_ev", prior_bounds=(0.15, 0.35)
    )
    assert not bad["eligible_for_preliminary_target_training"]
