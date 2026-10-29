import json

import numpy as np
import pytest

pytest.importorskip("torch")

from nufield.data_registry.splits import group_train_validation_split
from nufield.training.mdn_grid import build_mdn_trials, run_mdn_grid


def _synthetic_dataset(path, n=120):
    rng = np.random.default_rng(122)
    mass = np.tile(np.array([0.1, 0.2, 0.3, 0.4]), n // 4)
    omega = rng.uniform(0.25, 0.35, n)
    features = np.column_stack(
        [
            mass + rng.normal(0, 0.03, n),
            mass**2 + rng.normal(0, 0.02, n),
            omega + rng.normal(0, 0.01, n),
            rng.normal(size=n),
        ]
    ).astype(np.float32)
    theta = np.column_stack([mass, omega]).astype(np.float64)
    groups = np.array([f"parent-{i}" for i in range(n)])
    ids = np.array([f"sim-{i}" for i in range(n)])
    np.savez_compressed(path, features=features, theta=theta, group_id=groups, simulation_id=ids)
    path.with_suffix(".json").write_text(
        json.dumps({"physics_level": "toy", "dataset_id": "mdn-grid-test"})
    )
    return path


def test_mdn_trial_grid_deterministic_and_budget_bounded():
    a = build_mdn_trials(max_trials=3, budget="laptop", fit_seeds=(7, 8))
    b = build_mdn_trials(max_trials=3, budget="laptop", fit_seeds=(7, 8))
    assert len(a) <= 3
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert len({x.trial_id for x in a}) == len(a)


def test_inner_group_split_never_separates_parent_groups():
    groups = np.repeat([f"parent-{i}" for i in range(30)], 2)
    split = group_train_validation_split(groups, validation_fraction=0.2, seed=13)
    train_groups = set(groups[split["train"]])
    val_groups = set(groups[split["validation"]])
    assert train_groups.isdisjoint(val_groups)
    assert len(split["train"]) + len(split["validation"]) == len(groups)


def test_mdn_grid_ranks_validation_only_and_does_not_open_test(tmp_path):
    dataset = _synthetic_dataset(tmp_path / "toy.npz", n=120)
    report = run_mdn_grid(
        dataset,
        tmp_path / "grid-out",
        parameter_names=["sum_mnu_ev", "omega_m"],
        prior_low=[0.1, 0.2],
        prior_high=[0.4, 0.4],
        split_seed=51,
        max_trials=2,
        budget="laptop",
        epochs=4,
        patience=2,
        n_validation_observations=4,
        posterior_samples=24,
        max_coverage_gap=1.0,
        minimum_validation_groups=2,
        allow_toy=True,
    )
    assert report["trial_count"] == 2
    assert report["test_policy"].startswith("No trial evaluates")
    assert all(not trial["test_used"] for trial in report["trials"])
    assert all(
        trial["selection_partition"] == "outer validation only" for trial in report["trials"]
    )
    on_disk = json.loads((tmp_path / "grid-out" / "mdn_grid_report.json").read_text())
    assert on_disk["scientific_approval"] is False
    for trial in report["trials"]:
        output = tmp_path / "grid-out" / trial["trial_id"]
        assert (output / "mdn_report.json").exists()
        assert not (output / "heldout_posterior_samples.npz").exists()
