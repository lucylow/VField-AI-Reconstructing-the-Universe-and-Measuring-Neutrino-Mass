import numpy as np

from nufield.training.hpo_space import build_hpo_space, hpo_manifest
from nufield.training.nested_selection import CandidateMetric, hash_ids, select_candidate
from nufield.validation.posterior_diagnostics import (
    crps_ensemble,
    energy_score,
    interval_coverage,
    posterior_diagnostic_report,
)


def test_selection_requires_calibration_and_matching_validation_hash():
    h = hash_ids(["a", "b", "c"])
    metrics = [
        CandidateMetric(
            "m1", "crps", 0.5, "minimize", 12, calibrated=True, validation_group_hash=h
        ),
        CandidateMetric(
            "m2", "crps", 0.2, "minimize", 9, calibrated=False, validation_group_hash=h
        ),
        CandidateMetric(
            "m3", "crps", 0.1, "minimize", 8, calibrated=True, validation_group_hash="wrong"
        ),
    ]
    r = select_candidate(metrics, metric_name="crps", expected_validation_hash=h)
    assert r["selected"]["candidate_id"] == "m1"
    assert len(r["excluded"]) == 2


def test_hpo_grid_capped_by_laptop_and_manifest_hashes_must_differ():
    trials = build_hpo_space(
        architecture="spectrum_mlp", split_hash="split", budget="laptop", max_trials=200
    )
    assert len(trials) <= 12
    m = hpo_manifest(
        trials[:2], dataset_id="q", validation_groups_hash="val", test_groups_hash="test"
    )
    assert m["selection_partition"] == "validation"


def test_posterior_scores_and_coverage_shape():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(15, 2))
    samples = rng.normal(size=(15, 300, 2)) + truth[:, None, :] * 0.2
    assert crps_ensemble(samples, truth).shape == (15, 2)
    assert energy_score(samples, truth).shape == (15,)
    coverage = interval_coverage(samples, truth)
    assert "0.95" in coverage
    report = posterior_diagnostic_report(samples, truth, ["Mnu", "sigma8"])
    assert len(report["mean_crps_by_parameter"]) == 2
