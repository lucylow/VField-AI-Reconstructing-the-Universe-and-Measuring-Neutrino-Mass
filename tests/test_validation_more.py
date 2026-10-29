import numpy as np

from nufield.features.field_stats import one_point_moments
from nufield.validation.diagnostics import posterior_summary, standardized_residual_summary
from nufield.validation.finite_sim_uncertainty import binomial_coverage_interval


def test_binomial_interval_contains_empirical_fraction():
    lo, hi = binomial_coverage_interval(68, 100)
    assert lo < 0.68 < hi


def test_posterior_summary_shapes():
    out = posterior_summary(np.random.default_rng(0).normal(size=(200, 3)))
    assert len(out["mean"]) == 3
    assert "0.95" in out["quantiles"]


def test_moments_for_constant_field_are_finite():
    result = one_point_moments(np.ones((3, 3, 3)))
    assert np.isfinite(result["skewness"])
    assert result["std"] == 0


def test_residual_summary():
    result = standardized_residual_summary(np.array([-1, 0, 1, 2, -2], dtype=float))
    assert result["fraction_abs_below_1"] == 0.2
