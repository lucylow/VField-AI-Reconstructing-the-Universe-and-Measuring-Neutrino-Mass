import numpy as np

from nufield.training.active_design import (
    disagreement_design,
    latin_hypercube_candidates,
    maximin_design,
)


def test_latin_hypercube_within_bounds_and_reproducible():
    a = latin_hypercube_candidates([0.0, -1.0], [1.0, 1.0], n_candidates=50, seed=4)
    b = latin_hypercube_candidates([0.0, -1.0], [1.0, 1.0], n_candidates=50, seed=4)
    assert np.array_equal(a, b)
    assert np.all(a[:, 0] >= 0) and np.all(a[:, 0] <= 1)
    assert np.all(a[:, 1] >= -1) and np.all(a[:, 1] <= 1)


def test_maximin_and_disagreement_design_return_distinct_points():
    existing = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = maximin_design(existing, [0.0, 0.0], [1.0, 1.0], n_select=5, n_candidates=50, seed=3)
    assert len({tuple(round(v, 7) for v in x.parameters) for x in result}) == 5
    candidates = latin_hypercube_candidates([0.0, 0.0], [1.0, 1.0], n_candidates=20, seed=5)
    predictions = np.random.default_rng(5).normal(size=(4, 20, 2))
    uncertain = disagreement_design(candidates, predictions, [0.0, 0.0], [1.0, 1.0], n_select=4)
    assert len({x.index for x in uncertain}) == 4
