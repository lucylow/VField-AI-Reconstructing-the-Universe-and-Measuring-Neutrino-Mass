import numpy as np

from nufield.features.extract import extract_features
from nufield.features.power_spectrum import radial_power_spectrum


def test_power_spectrum_has_expected_shapes():
    x = np.random.default_rng(0).normal(size=(12, 12, 12))
    k, p, n = radial_power_spectrum(x, 300, n_bins=8)
    assert k.shape == p.shape == n.shape == (8,)
    assert np.all(n >= 0)


def test_feature_vector_finite():
    x = np.random.default_rng(0).normal(size=(12, 12, 12))
    feat = extract_features(x, 300, n_bins=8)
    assert feat.shape == (8,)
    assert np.isfinite(feat).all()
