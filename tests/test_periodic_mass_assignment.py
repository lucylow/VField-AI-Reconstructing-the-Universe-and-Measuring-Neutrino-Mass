import numpy as np
import pytest

from nufield.data_registry.cic import cloud_in_cell, mesh_to_overdensity
from nufield.data_registry.field_features import field_summary
from nufield.features.periodic_power import estimate_periodic_power


def test_cic_preserves_mass_and_mean_overdensity():
    rng = np.random.default_rng(10)
    pos = rng.uniform(0, 100, size=(500, 3))
    weights = rng.uniform(0.1, 3, size=500)
    mesh = cloud_in_cell(pos, box_size=100, grid_size=16, weights=weights)
    assert np.isclose(mesh.sum(), weights.sum(), rtol=1e-12)
    delta = mesh_to_overdensity(mesh)
    assert abs(delta.mean()) < 1e-12


def test_cic_refuses_ambiguous_nonperiodic_or_unwrapped_data():
    with pytest.raises(ValueError):
        cloud_in_cell(np.array([[1, 2, 3.0]]), box_size=10, grid_size=8, periodic=False)
    with pytest.raises(ValueError):
        cloud_in_cell(np.array([[10, 2, 3.0]]), box_size=10, grid_size=8)


def test_periodic_spectrum_uses_physical_volume_factor_and_feature_names():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(32, 32, 32))
    x -= x.mean()
    pk = estimate_periodic_power(x, box_size=500, n_bins=12)
    assert pk.grid_size == 32 and np.isfinite(pk.power[pk.mode_count > 0]).all()
    assert np.all(pk.k > 0)
    vec, names = field_summary(x, box_size=500, power_bins=8)
    assert len(vec) == len(names) and len(vec) > 20
