import numpy as np

from nufield.validation.stress import add_gaussian_feature_noise, stress_summary


def test_noise_is_deterministic_with_seed():
    x = np.ones((4, 3))
    a = add_gaussian_feature_noise(x, 0.2, np.random.default_rng(4))
    b = add_gaussian_feature_noise(x, 0.2, np.random.default_rng(4))
    assert np.allclose(a, b)


def test_stress_summary_zero_for_same_array():
    x = np.ones(4)
    assert stress_summary(x, x)["relative_l2_change"] == 0
