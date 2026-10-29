import json

import numpy as np
import pytest

from nufield.ai.audit import heldout_regression_report, scientific_claim_gates
from nufield.ai.benchmark import (
    cluster_bootstrap_metric,
    paired_cluster_bootstrap_difference,
    repeated_seed_summary,
)
from nufield.ai.checkpoints import (
    save_checkpoint_manifest,
    verify_checkpoint_manifest,
)
from nufield.ai.contracts import AIExperimentContract
from nufield.ai.experiment import (
    AIExperimentManifest,
    fingerprint_array,
    fingerprint_json,
)
from nufield.ai.model_selection import (
    CandidateEvidence,
    rank_candidates,
    validation_one_standard_error,
)
from nufield.ai.posterior_diagnostics import (
    chains_diagnostic_report,
    covariance_condition_report,
    effective_sample_size_1d,
    split_rhat,
)
from nufield.ai.predictive_checks import (
    featurewise_predictive_checks,
    posterior_predictive_rank,
    standardized_discrepancy,
)


def test_experiment_manifest_and_fingerprint(tmp_path):
    manifest = AIExperimentManifest(
        "exp",
        "regression",
        "test-model",
        7,
        {"depth": 2},
        {"train": "a" * 64},
        {"train": [0, 1], "validation": [2]},
        ["mnu"],
        {"mnu": "eV"},
        "toy vector",
        "toy",
    )
    out = manifest.to_dict()
    assert len(out["manifest_fingerprint"]) == 64 and not out["scientific_claim_eligible"]
    manifest.save(tmp_path / "manifest.json")
    assert json.loads((tmp_path / "manifest.json").read_text())["experiment_name"] == "exp"
    assert fingerprint_array(np.arange(10)) == fingerprint_array(np.arange(10))
    assert fingerprint_json({"x": 1, "y": 2}) == fingerprint_json({"y": 2, "x": 1})


def test_checkpoint_manifest_hash_and_tamper_detection(tmp_path):
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"some reproducible model bytes")
    kwargs = dict(
        model_name="tiny",
        model_config={"hidden": 8},
        training_config={"seed": 1},
        data_fingerprint="a" * 64,
        split_fingerprint="b" * 64,
        parameter_names=["mnu"],
        parameter_units={"mnu": "eV"},
        best_epoch=4,
        best_validation_loss=0.4,
        simulator_status="toy",
    )
    path = save_checkpoint_manifest(checkpoint, **kwargs)
    verified = verify_checkpoint_manifest(checkpoint, path)
    assert verified["valid"] and not verified["claim_eligible"]
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint_manifest(checkpoint, path)


def test_model_selection_uses_validation_and_mandatory_gates():
    good = CandidateEvidence("candidate_a", 0.1, 0.04, 0.8, 2.0, 300, True, True, "validated")
    bad = CandidateEvidence("candidate_b", 0.01, 0.3, 0.9, 1.0, 200, True, True, "validated")
    result = rank_candidates(
        [good, bad], max_calibration_error=0.1, require_validated_simulator=True
    )
    assert result["winner"] == "candidate_a"
    assert result["rejected_or_gated"][0]["candidate"] == "candidate_b"
    folds = validation_one_standard_error(
        np.array([[1.0, 1.2, 0.9], [1.02, 1.0, 1.1], [0.8, 0.8, 0.8]]), complexity=[50, 10, 100]
    )
    assert folds["one_se_candidate_index"] in {0, 1, 2}


def test_cluster_bootstrap_and_paired_model_difference():
    rng = np.random.default_rng(30)
    n_groups = 20
    group = np.repeat([f"g{i}" for i in range(n_groups)], 3)
    y = rng.normal(size=(len(group), 2))
    a = y + 0.5
    b = y + 0.1
    report = cluster_bootstrap_metric(y, b, group, n_bootstrap=100)
    paired = paired_cluster_bootstrap_difference(y, a, b, group, n_bootstrap=100)
    assert report["n_groups"] == n_groups
    assert paired["difference_a_minus_b"] > 0
    assert paired["confidence_interval"][0] > 0
    assert repeated_seed_summary([1, 2, 3])["n_seeds"] == 3


def test_posterior_chain_diagnostics_and_covariance():
    rng = np.random.default_rng(31)
    chains = rng.normal(size=(4, 300, 2))
    assert np.all(split_rhat(chains) < 1.1)
    report = chains_diagnostic_report(chains)
    assert report["minimum_ess"] > 10
    assert effective_sample_size_1d(rng.normal(size=100)) > 10
    cov = covariance_condition_report(rng.normal(size=(100, 3)))
    assert cov["positive_definite"]


def test_posterior_predictive_and_scientific_gates():
    obs = np.array([1.0, 2.0, 3.0])
    rep = np.random.default_rng(32).normal(loc=obs, scale=0.1, size=(40, 3))
    check = standardized_discrepancy(obs, rep)
    assert check["max_abs_z"] < 5
    assert len(featurewise_predictive_checks(obs, rep, feature_names=["a", "b", "c"])) == 3
    assert "empirical_lower_tail" in posterior_predictive_rank(
        0.5, np.random.default_rng(33).normal(size=50)
    )
    contract = AIExperimentContract(
        "regression",
        "summary",
        100,
        100,
        1,
        True,
        True,
        True,
        True,
        "toy",
        target_observation_mapping_verified=True,
    )
    gates = scientific_claim_gates(
        contract=contract,
        group_leakage_detected=False,
        calibration_error=0.04,
        systematic_bias_fraction=0.1,
    )
    assert not gates["all_gates_pass"]


def test_locked_test_report_rejects_validation_partition():
    y = np.zeros(10)
    with pytest.raises(ValueError, match="held-out test"):
        heldout_regression_report(y, y, partition_name="validation")
