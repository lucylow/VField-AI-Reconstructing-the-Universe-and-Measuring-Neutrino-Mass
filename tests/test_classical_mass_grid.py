import json

import numpy as np
import pytest

from nufield.benchmarking.classical_mass_grid import ClassicalMassGridModel
from nufield.training.classical_mass_grid_workflow import run_classical_mass_grid


def test_classical_mass_grid_returns_discrete_normalized_posterior():
    rng = np.random.default_rng(1)
    mass = np.repeat([0.1, 0.2, 0.3, 0.4], 8)
    x = np.column_stack(
        [
            mass * 2 + rng.normal(0, 0.08, len(mass)),
            rng.normal(size=len(mass)),
            rng.normal(size=len(mass)),
        ]
    )
    groups = np.array([f"phase-{i}" for i in range(len(mass))])
    model = ClassicalMassGridModel.fit(
        x, mass, group_ids=groups, prior_bounds=(0.1, 0.4), min_replicates_per_mass=3
    )
    weights = model.posterior_weights(np.array([0.75, 0.0, 0.0]))
    assert np.isclose(weights.sum(), 1)
    assert model.mass_grid.tolist() == [0.1, 0.2, 0.3, 0.4]
    samples = model.sample_posterior(np.array([0.75, 0.0, 0.0]), n_samples=50, seed=4)
    assert samples.shape == (50, 1)
    assert set(np.unique(samples[:, 0])).issubset(set(model.mass_grid))
    assert model.fit_summary["posterior_type"] == "discrete mass grid; no interpolation"


def test_classical_baseline_rejects_varying_unmodelled_nuisance():
    rng = np.random.default_rng(2)
    mass = np.repeat([0.1, 0.2, 0.3], 5)
    nuisance = np.linspace(0.25, 0.4, len(mass))
    x = rng.normal(size=(len(mass), 4))
    theta = np.column_stack([mass, nuisance])
    with pytest.raises(ValueError, match="nuisance parameters"):
        ClassicalMassGridModel.fit(
            x, mass, nuisance_parameters=nuisance[:, None], prior_bounds=(0.1, 0.3)
        )


def test_classical_workflow_uses_group_split_and_keeps_test_closed_by_default(tmp_path):
    rng = np.random.default_rng(3)
    mass = np.repeat([0.1, 0.2, 0.3, 0.4], 16)
    n = len(mass)
    features = np.column_stack(
        [2 * mass + rng.normal(0, 0.12, n), rng.normal(size=n), rng.normal(size=n)]
    )
    theta = mass[:, None]
    ids = np.array([f"s{i}" for i in range(n)])
    groups = np.array([f"p{i}" for i in range(n)])
    path = tmp_path / "dataset.npz"
    np.savez_compressed(path, features=features, theta=theta, simulation_id=ids, group_id=groups)
    path.with_suffix(".json").write_text(json.dumps({"physics_level": "toy"}))
    report = run_classical_mass_grid(
        path,
        tmp_path / "out",
        parameter_names=["sum_mnu_ev"],
        prior_bounds=(0.1, 0.4),
        allow_toy=True,
        min_groups_per_mass=2,
        n_eval=6,
        posterior_samples=64,
        evaluate_test_once=False,
    )
    assert report["test_evaluated_once"] is False
    assert report["final_test_metrics"] is None
    assert not (tmp_path / "out" / "heldout_posterior_samples.npz").exists()
    val = np.load(tmp_path / "out" / "validation_posterior_samples.npz")
    assert val["samples"].shape == (6, 64, 1)
    assert (
        report["model"]["fit_summary"]["posterior_type"] == "discrete mass grid; no interpolation"
    )
