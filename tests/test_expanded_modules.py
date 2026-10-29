import numpy as np
import pytest

from nufield.compute.chunked import iter_batches
from nufield.exceptions import ScientificValidityError
from nufield.experiments.parameter_design import latin_hypercube
from nufield.features.schema import FeatureDefinition, FeatureSchema
from nufield.inference.gaussian_likelihood import GaussianSummaryLikelihood
from nufield.inference.importance import importance_diagnostics, normalized_importance_weights
from nufield.inference.parameter_transforms import from_unit_interval, to_unit_interval
from nufield.simulation.positions import counts_to_positions
from nufield.simulation.response import paired_field_response
from nufield.simulation.window import apply_selection_probability
from nufield.validation.domain_shift import nearest_neighbor_distances
from nufield.validation.observed_data_guard import require_research_grade_physics
from nufield.validation.spectrum_metrics import compare_spectra


def test_latin_hypercube_shape_and_bounds():
    x = latin_hypercube(20, seed=12)
    assert x.shape == (20, 6)
    assert np.all(x >= 0.06) and np.all(x[:, 0] <= 0.30)


def test_gaussian_likelihood_finite():
    like = GaussianSummaryLikelihood(np.zeros(2), np.eye(2))
    assert np.isfinite(like.logpdf(np.ones(2)))
    assert np.allclose(like.whitened_residual(np.ones(2)), np.ones(2))


def test_importance_weights_normalize():
    w = normalized_importance_weights(np.zeros(4), np.zeros(4), np.array([0.0, 1.0, 0.0, -1.0]))
    assert np.isclose(w.sum(), 1)
    assert importance_diagnostics(w)["effective_sample_size"] >= 1


def test_parameter_transform_roundtrip():
    lo = np.array([0.0, 10.0])
    hi = np.array([1.0, 20.0])
    x = np.array([[0.3, 17.0]])
    assert np.allclose(from_unit_interval(to_unit_interval(x, lo, hi), lo, hi), x)


def test_feature_schema_has_stable_hash():
    schema = FeatureSchema("1", (FeatureDefinition("a", "one feature", "dimensionless", "mean"),))
    assert len(schema.sha256()) == 64


def test_position_sampling_respects_box():
    counts = np.zeros((4, 4, 4), int)
    counts[1, 2, 3] = 5
    pos = counts_to_positions(counts, 10, np.random.default_rng(1))
    assert pos.shape == (5, 3)
    assert np.all(pos >= 0) and np.all(pos < 10)


def test_selection_probabilities_are_applied():
    c = np.full((3, 3, 3), 20)
    out = apply_selection_probability(c, np.zeros_like(c, dtype=float), np.random.default_rng(0))
    assert out.sum() == 0


def test_response_identical_fields_zero():
    x = np.random.default_rng(0).normal(size=(5, 5, 5))
    assert paired_field_response(x, x)["rms_difference"] == 0


def test_spectrum_metrics_identical_spectra():
    k = np.arange(1, 5, dtype=float)
    p = np.array([2.0, 3.0, 4.0, 5.0])
    result = compare_spectra(k, p, p)
    assert result["fractional_bias_rms"] == 0


def test_domain_distance_is_zero_for_same_row():
    x = np.random.default_rng(0).normal(size=(8, 3))
    d = nearest_neighbor_distances(x, x)
    assert np.allclose(d, 0)


def test_toy_guard_blocks_real_measurement():
    with pytest.raises(ScientificValidityError):
        require_research_grade_physics("toy", "scientific_measurement")


def test_batch_iterator_yields_all_rows():
    x = np.arange(13)
    batches = list(iter_batches(x, 5))
    assert sum(len(batch) for _, batch in batches) == 13
