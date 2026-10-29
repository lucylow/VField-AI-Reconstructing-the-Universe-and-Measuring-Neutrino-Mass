import numpy as np
import pytest

from nufield.ai.contracts import (
    AIExperimentContract,
    ArrayContract,
    assert_group_disjoint,
    assert_same_rows,
)
from nufield.ai.dataset import SimulationDataset, dataset_quality_report
from nufield.ai.grouped_splits import (
    grouped_train_validation_calibration_test_split,
    split_distribution_report,
)
from nufield.ai.normalization import BoxTransform, RobustStandardizer, Standardizer
from nufield.ai.synthetic import make_toy_inverse_problem


def test_array_contract_and_experiment_claim_gates():
    assert ArrayContract("pk", (32,), "float64", "(Mpc/h)^3").validate() == []
    c = AIExperimentContract(
        "posterior",
        "summary",
        100,
        90,
        3,
        True,
        True,
        True,
        True,
        "toy",
        parameter_variation_verified=True,
        target_observation_mapping_verified=True,
    )
    assert any("validated" in e for e in c.validate(for_scientific_claim=True))
    c2 = AIExperimentContract(
        "posterior",
        "summary",
        100,
        90,
        3,
        True,
        True,
        True,
        True,
        "validated",
        parameter_variation_verified=True,
        target_observation_mapping_verified=True,
    )
    assert c2.validate(for_scientific_claim=True) == []


def test_group_disjoint_raises_on_overlap():
    with pytest.raises(ValueError, match="overlap"):
        assert_group_disjoint(["a", "b"], ["b", "c"], ["d"])


def test_same_rows_checks_arrays():
    assert assert_same_rows(x=np.ones((8, 2)), y=np.ones((8, 1))) == 8
    with pytest.raises(ValueError):
        assert_same_rows(x=np.ones((8, 2)), y=np.ones((7, 1)))


def test_grouped_split_disjoint_and_full_coverage():
    n = 200
    groups = np.asarray([f"g{i:03d}" for i in range(n)])
    values = np.linspace(0, 1, n)
    split = grouped_train_validation_calibration_test_split(groups, values, seed=19, n_bins=5)
    split.validate()
    all_i = np.concatenate(
        [
            split.train_indices,
            split.validation_indices,
            split.calibration_indices,
            split.test_indices,
        ]
    )
    assert sorted(all_i.tolist()) == list(range(n))
    assert len(set(split.train_groups) & set(split.test_groups)) == 0
    report = split_distribution_report(split, groups, values)
    assert report["partitions"]["train"]["groups"] > report["partitions"]["calibration"]["groups"]


def test_grouped_split_rejects_varying_label_within_parent_group():
    groups = np.array([f"g{i // 2}" for i in range(40)])
    values = np.arange(40, dtype=float)
    with pytest.raises(ValueError, match="varies within parent group"):
        grouped_train_validation_calibration_test_split(groups, values)


def test_dataset_roundtrip_and_fingerprint(tmp_path):
    dataset = make_toy_inverse_problem(n_samples=80, seed=4)
    path = tmp_path / "toy.npz"
    saved = dataset.save(path)
    loaded = SimulationDataset.load(path)
    assert loaded.fingerprint() == dataset.fingerprint() == saved["dataset_fingerprint"]
    report = dataset_quality_report(loaded)
    assert report["valid"] and report["n_rows"] == 80


def test_dataset_detects_manifest_tamper(tmp_path):
    dataset = make_toy_inverse_problem(n_samples=60, seed=1)
    path = tmp_path / "toy.npz"
    dataset.save(path)
    with np.load(path, allow_pickle=False) as src:
        arrays = {k: src[k] for k in src.files}
    arrays["parameters"][0, 0] += 0.1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        SimulationDataset.load(path)


def test_standardizers_roundtrip_and_constant_columns():
    x = np.column_stack([np.arange(8), np.ones(8)])
    scaler = Standardizer().fit(x)
    y = scaler.transform(x)
    assert np.allclose(scaler.inverse_transform(y), x, atol=2e-6)
    assert scaler.scale_[1] == 1
    robust = RobustStandardizer().fit(x)
    assert np.isfinite(robust.transform(x)).all()


def test_box_transform_round_trip():
    tr = BoxTransform((0.0, -2.0), (1.0, 2.0))
    x = np.array([[0.2, 0.0], [0.8, 1.5]])
    assert np.allclose(tr.inverse(tr.forward(x)), x, atol=2e-6)
    with pytest.raises(ValueError):
        tr.forward(np.array([[2.0, 0.0]]))
