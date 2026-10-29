import numpy as np

from nufield.models.architecture_selector import DataModelContract, eligible_architectures
from nufield.models.benchmark_policy import BenchmarkPolicy, compare_policies
from nufield.models.observation_quality import audit_observations


def _base(**kw):
    data = dict(
        n_independent_groups=1200,
        sample_shape=(1, 32, 32, 32),
        observation_kind="field3d",
        target_dimension=5,
        has_mass_variation=True,
        mass_grid_verified=True,
        train_validation_test_disjoint=True,
        physics_level="validated",
        regular_grid=True,
        periodic_box=True,
        has_power_spectrum=True,
        has_lensing=False,
        modalities_physically_matched=False,
        prior_declared=True,
        source_schema_verified=True,
    )
    data.update(kw)
    return DataModelContract(**data)


def test_fno_requires_periodic_regular_box_and_workstation():
    decisions = {x.model_id: x for x in eligible_architectures(_base(), budget="workstation")}
    assert decisions["fno3d_npe"].eligible
    decisions = {
        x.model_id: x
        for x in eligible_architectures(_base(periodic_box=False), budget="workstation")
    }
    assert not decisions["fno3d_npe"].eligible
    decisions = {x.model_id: x for x in eligible_architectures(_base(), budget="laptop")}
    assert not decisions["fno3d_npe"].eligible


def test_lensing_requires_matched_modalities():
    c = _base(has_lensing=True, modalities_physically_matched=False)
    assert c.validate()


def test_policy_comparison_catches_split_mismatch():
    a = BenchmarkPolicy("d", "r", "s1", "t1", ("Mnu",), (0.0,), (0.4,), "validated", "obs")
    b = BenchmarkPolicy("d", "r", "s1", "t2", ("Mnu",), (0.0,), (0.4,), "validated", "obs")
    assert not compare_policies(a, b)["comparable"]


def test_observation_quality_reports_shape_and_nan():
    x = np.ones((10, 8, 8, 8))
    x[0, 0, 0, 0] = np.nan
    r = audit_observations(x, expected_kind="field3d")
    assert not r.to_dict()["can_train"]
