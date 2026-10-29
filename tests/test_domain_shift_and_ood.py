import numpy as np

from nufield.data_registry.domain_shift import (
    domain_shift_report,
    maximum_mean_discrepancy,
    standardized_mean_difference,
)
from nufield.validation.ood import fit_support_envelope, score_support_envelope


def test_domain_shift_zero_for_identical_distribution_sample():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4))
    smd = standardized_mean_difference(x, x.copy())
    np.testing.assert_allclose(smd, 0, atol=1e-12)
    m = maximum_mean_discrepancy(x, x, seed=1)
    assert m["biased_mmd2"] < 1e-9
    assert domain_shift_report(x, x)["max_abs_smd"] < 1e-12


def test_ood_support_flags_far_away_point():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 3))
    envelope = fit_support_envelope(x, quantile=0.98)
    r = score_support_envelope(envelope, np.vstack([x[:5], np.full((1, 3), 12.0)]))
    assert r["out_of_support"][-1]
