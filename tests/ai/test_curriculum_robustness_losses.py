import numpy as np
import pytest

from nufield.ai.curriculum import (
    bounded_sample_weights,
    curriculum_weights,
    training_residual_difficulty,
    update_ema_weights,
)
from nufield.ai.domain_adaptation import domain_balance_report
from nufield.ai.losses import gaussian_nll_numpy


def test_curriculum_weights_expand_active_fraction_and_stay_positive():
    d = np.linspace(0, 1, 100)
    early, state_early = curriculum_weights(d, epoch=1, total_epochs=10)
    late, state_late = curriculum_weights(d, epoch=10, total_epochs=10)
    assert state_late.active_fraction > state_early.active_fraction
    assert np.all(early > 0) and np.all(late > 0)
    assert np.isclose(early.mean(), 1) and np.isclose(late.mean(), 1)


def test_hard_example_weights_are_bounded_and_smoothed():
    loss = np.arange(1, 20, dtype=float)
    w = bounded_sample_weights(loss, minimum_weight=0.2, maximum_weight=3)
    smoothed = update_ema_weights(np.ones(19), w, momentum=0.8)
    assert np.isclose(w.mean(), 1)
    assert np.isclose(smoothed.mean(), 1)
    assert smoothed.min() > 0
    assert training_residual_difficulty(loss).shape == loss.shape


def test_numpy_gaussian_nll_rejects_bad_variance():
    with pytest.raises(ValueError):
        gaussian_nll_numpy(np.zeros(4), np.zeros(4), np.ones(4))


def test_domain_balance_report_exposes_confounding():
    domains = np.array([0] * 20 + [1] * 20)
    theta = np.concatenate([np.zeros((20, 1)), np.ones((20, 1)) * 2])
    report = domain_balance_report(domains, theta)
    assert report["n_domains"] == 2
    assert report["target_means_by_domain"]["1"] == [2.0]
