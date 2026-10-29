import numpy as np
import pytest

from nufield.benchmarking.mass_response import paired_mass_response, response_table_by_mass


def test_paired_mass_response_recovers_constructed_signal():
    rng = np.random.default_rng(21)
    rows, mass, phase, strata = [], [], [], []
    for phase_id in range(20):
        shared = rng.normal(size=4)
        for mnu in (0.06, 0.30):
            rows.append(shared + mnu * np.array([4.0, 2.0, 0.2, -1.0]) + rng.normal(0, 0.001, 4))
            mass.append(mnu)
            phase.append(phase_id)
            strata.append("cosmoA")
    report = paired_mass_response(
        np.asarray(rows),
        np.asarray(mass),
        np.asarray(phase),
        low_mass_ev=0.06,
        high_mass_ev=0.30,
        strata_ids=np.asarray(strata),
    )
    assert report.n_matched_pairs == 20
    assert report.paired_design
    assert report.paired_z_score[0] > report.paired_z_score[2]
    assert report.strongest_abs_z_feature_indices[0] == 0


def test_paired_mass_response_rejects_mismatched_or_duplicate_keys():
    x = np.arange(16, dtype=float).reshape(8, 2)
    m = np.array([0.06, 0.06, 0.06, 0.06, 0.3, 0.3, 0.3, 0.3])
    phase = np.array([1, 2, 3, 4, 1, 2, 2, 5])
    with pytest.raises(ValueError, match="duplicate row"):
        paired_mass_response(x, m, phase, low_mass_ev=0.06, high_mass_ev=0.3)


def test_response_table_reports_reference_mass():
    x = np.array([[1.0, 2.0], [1.2, 2.2], [2.0, 2.5], [2.2, 2.7]])
    m = np.array([0.06, 0.06, 0.3, 0.3])
    out = response_table_by_mass(x, m, reference_mass_ev=0.06, feature_names=["pk0", "pk1"])
    assert len(out) == 4
    target = [r for r in out if r["sum_mnu_ev"] == 0.3 and r["feature"] == "pk0"][0]
    assert target["fractional_shift_vs_reference"] > 0.5
