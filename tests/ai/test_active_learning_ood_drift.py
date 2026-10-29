import numpy as np
import pytest

from nufield.ai.active_learning import (
    candidate_pool,
    ensemble_disagreement,
    predictive_entropy_score,
    select_candidates,
)
from nufield.ai.drift import DriftMonitor, feature_drift_report, population_stability_index
from nufield.ai.ood import MahalanobisOOD, NearestReferenceOOD, ood_separation_report
from nufield.ai.robustness import groupwise_metrics, perturbation_sweep_report


def test_candidate_pool_and_maximin_selection_are_reproducible():
    pool = candidate_pool([0, 0], [1, 2], n_candidates=500, seed=1)
    a = select_candidates(pool, [0, 0], [1, 2], n_select=10, method="maximin", seed=1)
    b = select_candidates(pool, [0, 0], [1, 2], n_select=10, method="maximin", seed=1)
    assert [r.parameters for r in a.selected] == [r.parameters for r in b.selected]
    assert len({r.candidate_index for r in a.selected}) == 10


def test_ensemble_disagreement_and_entropy_scores():
    rng = np.random.default_rng(11)
    pred = rng.normal(size=(4, 50, 3))
    var = np.ones_like(pred) * 0.2
    d = ensemble_disagreement(pred, [1, 1, 1])
    e = predictive_entropy_score(pred, var, [1, 1, 1])
    assert d.shape == e.shape == (50,)
    records = select_candidates(
        rng.uniform(size=(50, 2)),
        [0, 0],
        [1, 1],
        n_select=5,
        method="ensemble_disagreement",
        predictions=pred,
        existing_parameters=np.empty((0, 2)),
    )
    assert len(records.selected) == 5


def test_active_learning_rejects_invalid_prior_bounds():
    with pytest.raises(ValueError):
        candidate_pool([1, 0], [0, 1])


def test_ood_detector_requires_independent_calibration():
    rng = np.random.default_rng(12)
    train = rng.normal(size=(200, 4))
    val = rng.normal(size=(100, 4))
    shifted = rng.normal(4, 0.2, size=(30, 4))
    det = MahalanobisOOD().fit(train)
    with pytest.raises(RuntimeError):
        det.predict_ood(shifted)
    det.calibrate(val, quantile=0.95)
    assert det.predict_ood(shifted).mean() > 0.8
    assert (
        ood_separation_report(det.score(val), det.score(shifted), threshold=det.threshold_)[
            "shifted_detection_rate"
        ]
        > 0.8
    )


def test_nearest_reference_ood():
    rng = np.random.default_rng(13)
    train = rng.normal(size=(100, 3))
    model = NearestReferenceOOD.fit(train).calibrate(rng.normal(size=(30, 3)), quantile=0.9)
    assert model.predict_ood(rng.normal(3, 0.1, size=(10, 3))).all()


def test_feature_drift_report_flags_shift():
    rng = np.random.default_rng(14)
    reference = rng.normal(size=(500, 3))
    current = rng.normal(size=(500, 3))
    current[:, 1] += 2.0
    monitor = DriftMonitor.fit(reference, feature_names=["a", "b", "c"])
    report = monitor.evaluate(current)
    assert report["features"][1]["psi"] > report["features"][0]["psi"]
    assert population_stability_index(reference, current).shape == (3,)
    assert len(feature_drift_report(reference, current)["features"]) == 3


def test_groupwise_and_perturbation_metrics():
    y = np.zeros((40, 2))
    pred = y.copy()
    pred[20:] += 1
    groups = np.array(["domain-a"] * 20 + ["domain-b"] * 20)
    report = groupwise_metrics(y, pred, groups)
    assert report["worst_group"] == 1
    perturb = perturbation_sweep_report(y, pred, {"noise": pred + 0.2})
    assert perturb["perturbations"]["noise"]["rmse_scaled"] > 0.8
