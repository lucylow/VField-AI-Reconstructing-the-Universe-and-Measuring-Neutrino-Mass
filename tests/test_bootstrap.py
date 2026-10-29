import numpy as np

from nufield.validation.bootstrap import bootstrap_mean_interval, paired_bootstrap_difference


def test_bootstrap_interval_is_finite():
    mean, lo, hi = bootstrap_mean_interval(np.arange(20), n_bootstrap=200, seed=1)
    assert lo <= mean <= hi


def test_paired_difference_zero_for_equal_inputs():
    x = np.arange(12, dtype=float)
    result = paired_bootstrap_difference(x, x, n_bootstrap=200)
    assert result["mean_difference_a_minus_b"] == 0
