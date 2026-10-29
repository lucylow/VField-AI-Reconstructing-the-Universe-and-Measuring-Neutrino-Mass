import numpy as np
import pytest

from nufield.ai.conformal import ConformalRegressor, conformal_quantile, empirical_coverage
from nufield.ai.group_conformal import GroupConformalRegressor
from nufield.ai.metrics import (
    calibration_curve,
    energy_score,
    gaussian_crps,
    regression_metrics,
    sbc_rank_histograms,
    simulation_based_calibration_ranks,
    stratified_error_report,
)
from nufield.ai.uncertainty import decompose_ensemble, gaussian_entropy, predictive_interval
from nufield.ai.variance_calibration import VarianceTemperature, reliability_by_variance_bin


def test_ensemble_law_total_variance():
    means = np.array([[[0.0], [1.0]], [[2.0], [3.0]], [[1.0], [2.0]]])
    variances = np.ones_like(means) * 0.25
    result = decompose_ensemble(means, variances)
    assert np.allclose(result.mean.ravel(), [1, 2])
    assert np.allclose(result.epistemic_variance.ravel(), [1, 1])
    assert np.allclose(result.aleatoric_variance.ravel(), [0.25, 0.25])
    assert np.allclose(result.total_variance, result.epistemic_variance + result.aleatoric_variance)


def test_conformal_and_empirical_coverage():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(150, 2))
    pred = y + rng.normal(0, 0.2, size=y.shape)
    cal = ConformalRegressor.fit(y[:70], pred[:70], alpha=0.1, simultaneous=True)
    lo, hi = cal.predict(pred[70:])
    report = empirical_coverage(y[70:], lo, hi)
    assert report["simultaneous_coverage"] > 0.75
    assert cal.quantile >= 0


def test_conformal_quantile_finite_sample_correction():
    assert conformal_quantile([1, 2, 3, 4, 5], alpha=0.1) == 5
    with pytest.raises(ValueError):
        conformal_quantile([], alpha=0.1)


def test_group_conformal_scores_parent_groups():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(80, 1))
    pred = y + rng.normal(0, 0.2, size=(80, 1))
    groups = np.array([f"g{i // 4}" for i in range(80)])
    cal = GroupConformalRegressor.fit(y[:40], pred[:40], groups[:40], alpha=0.1)
    assert cal.n_calibration_groups == 10
    lo, hi = cal.predict(pred[40:])
    assert lo.shape == hi.shape == (40, 1)
    with pytest.raises(ValueError):
        GroupConformalRegressor.fit(y[:10], pred[:10], np.array(["same"] * 10))


def test_regression_metrics_variance_and_calibration_curve():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(120, 2))
    mean = y + rng.normal(0, 0.2, size=y.shape)
    v = np.ones_like(y) * 0.05
    metrics = regression_metrics(y, mean, variance=v)
    assert metrics["rmse_mean"] > 0 and "gaussian_nll_mean" in metrics
    curve = calibration_curve(mean, v, y, n_bins=6)
    assert len(curve["nominal"]) == 6


def test_variance_temperature_and_reliability_report():
    rng = np.random.default_rng(5)
    y = rng.normal(size=(100, 2))
    mean = y + rng.normal(0, 0.5, size=y.shape)
    var = np.ones_like(y) * 0.1
    cal = VarianceTemperature.fit(mean, var, y, grid_size=31)
    assert cal.temperature > 0 and cal.n_calibration == 100
    report = reliability_by_variance_bin(y, mean, cal.apply(var), n_bins=5)
    assert report["n"] == 100


def test_sbc_ranks_and_histogram():
    rng = np.random.default_rng(6)
    truths = rng.normal(size=(40, 2))
    draws = rng.normal(size=(40, 100, 2))
    ranks = simulation_based_calibration_ranks(draws, truths)
    assert ranks.shape == truths.shape
    assert len(sbc_rank_histograms(draws, truths)["histograms"]) == 2


def test_energy_score_crps_and_stratified_report():
    rng = np.random.default_rng(7)
    y = rng.normal(size=(20, 2))
    samples = rng.normal(size=(20, 50, 2))
    assert energy_score(samples, y).shape == (20,)
    assert gaussian_crps(np.zeros_like(y), np.ones_like(y), y).shape == y.shape
    report = stratified_error_report(y, y + 0.1, stratifier=np.arange(20), n_bins=4)
    assert len(report["bins"]) == 4
    assert np.all(np.isfinite(gaussian_entropy(np.ones((4, 2)))))
    lo, hi = predictive_interval(y, np.ones_like(y))
    assert np.all(lo < hi)
