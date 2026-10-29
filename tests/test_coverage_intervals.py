import numpy as np
import pytest

from nufield.validation.coverage_intervals import (
    coverage_interval_report,
    rank_histogram,
    wilson_interval,
)


def test_wilson_interval_contains_observed_rate_and_is_bounded():
    lo, hi = wilson_interval(68, 100)
    assert 0 <= lo < 0.68 < hi <= 1
    assert wilson_interval(0, 10)[0] == 0
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_coverage_report_includes_finite_sample_uncertainty():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(120, 2))
    samples = rng.normal(loc=truth[:, None, :], scale=1.0, size=(120, 400, 2))
    report = coverage_interval_report(samples, truth, levels=(0.68,))
    assert report["n_simulations"] == 120
    assert len(report["results"]) == 2
    assert all(
        0 <= row["coverage_ci_low"] <= row["empirical_coverage"] <= row["coverage_ci_high"] <= 1
        for row in report["results"]
    )


def test_sbc_rank_histogram_validates_rank_range():
    ranks = np.tile(np.arange(11, dtype=int)[:, None], (1, 2))
    result = rank_histogram(ranks, n_posterior_samples=10, n_bins=5)
    assert sum(result["counts"]) == ranks.size
    with pytest.raises(ValueError):
        rank_histogram(np.array([[12]]), n_posterior_samples=10)
