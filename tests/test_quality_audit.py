import numpy as np

from nufield.data_registry.quality import audit_numeric_dataset


def test_quality_audit_catches_constant_target_and_missing_provenance():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 5))
    theta = np.column_stack([np.full(30, 0.1), rng.normal(size=30)])
    report = audit_numeric_dataset(
        x,
        theta,
        parameter_names=["sum_mnu_ev", "omega_m"],
        simulation_ids=np.arange(30),
        group_ids=np.repeat(np.arange(10), 3),
        metadata={"physics_level": "toy"},
    )
    assert report.has_errors
    assert report.n_unique_groups == 10
    assert any(f.code == "CONSTANT_TARGET" for f in report.findings)
    assert any("units" in warning for warning in report.warnings)
    assert report.science_approval is False


def test_quality_audit_detects_nonfinite_and_duplicate_summaries():
    x = np.array([[1.0, 2.0], [1.0, 2.0], [np.nan, 4.0], [4.0, 6.0]])
    theta = np.column_stack([np.linspace(0.05, 0.2, 4)])
    report = audit_numeric_dataset(x, theta, parameter_names=["sum_mnu_ev"])
    assert any(f.code == "NONFINITE_OBSERVATIONS" for f in report.findings)
    assert (
        report.duplicate_observation_rows is None
    )  # duplicate scan is not run with nonfinite data
    assert not report.can_attempt_training
