import numpy as np

from nufield.inference.sbc import coverage_table, posterior_rank, rank_histogram


def test_rank_histogram_counts_all_entries():
    counts = rank_histogram(np.array([0, 1, 2, 3, 4]), n_draws=4, n_bins=4)
    assert counts.sum() == 5


def test_coverage_table_returns_expected_keys():
    samples = [np.random.default_rng(i).normal(size=(500, 2)) for i in range(5)]
    truths = np.zeros((5, 2))
    result = coverage_table(samples, truths)
    assert "coverage_0.68" in result
    assert len(result["coverage_0.68"]) == 2


def test_randomized_rank_shape():
    rng = np.random.default_rng(1)
    ranks = posterior_rank(np.zeros((50, 3)), np.zeros(3), rng)
    assert ranks.shape == (3,)
