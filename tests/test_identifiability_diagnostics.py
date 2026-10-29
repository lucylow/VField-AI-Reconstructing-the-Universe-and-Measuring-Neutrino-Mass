import numpy as np

from nufield.benchmarking.identifiability import analyze_identifiability


def test_identifiability_finds_obvious_signal():
    rng = np.random.default_rng(8)
    target = rng.uniform(0.05, 0.4, 300)
    x = np.column_stack(
        [2.5 * target + rng.normal(0, 0.02, len(target)), rng.normal(size=len(target))]
    )
    theta = np.column_stack([target, rng.uniform(0.25, 0.35, len(target))])
    report = analyze_identifiability(x, theta, parameter_names=["sum_mnu_ev", "omega_m"])
    assert report.n_simulations == 300
    assert report.target_unique_values == 300
    assert abs(report.feature_correlations[0]) > 0.95
    assert report.linear_r2_in_sample > 0.8
    assert (
        report.scientific_disclaimer
        if hasattr(report, "scientific_disclaimer")
        else report.disclaimer
    )


def test_identifiability_flags_constant_target():
    x = np.random.default_rng(4).normal(size=(20, 5))
    y = np.column_stack([np.full(20, 0.1)])
    report = analyze_identifiability(x, y, parameter_names=["sum_mnu_ev"])
    assert any("constant" in f.lower() for f in report.findings)
