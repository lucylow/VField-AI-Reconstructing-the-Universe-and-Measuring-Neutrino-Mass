import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from nufield.models.bounded_transform import BoundedParameterTransform
from nufield.models.conditional_mdn import (
    MDNConfig,
    draw_physical_posterior,
    fit_conditional_mdn,
)
from nufield.training.mdn_workflow import train_mdn_from_npz


def test_bounded_transform_roundtrip_and_support():
    bounds = BoundedParameterTransform(np.array([0.0, 0.1]), np.array([1.0, 0.5]))
    x = np.array([[0.2, 0.2], [0.8, 0.4]])
    np.testing.assert_allclose(bounds.inverse(bounds.forward(x)), x, atol=1e-6)
    with pytest.raises(ValueError):
        bounds.forward(np.array([[1.1, 0.2]]))
    z = np.array([[1000.0, -1000.0]])
    y = bounds.inverse(z)
    assert y[0, 0] <= 1 and y[0, 0] > 0.99
    assert y[0, 1] >= 0.1 and y[0, 1] < 0.11


def test_mdn_fit_returns_finite_validation_history_and_bounded_samples():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(90, 5)).astype("f4")
    y = np.column_stack(
        [0.1 + 0.3 / (1 + np.exp(-x[:, 0])) * 0.95, 0.25 + 0.04 / (1 + np.exp(-x[:, 1]))]
    ).astype("f4")
    train = np.arange(0, 68)
    val = np.arange(68, 90)
    from nufield.data_registry.preprocessing import Standardizer
    from nufield.models.bounded_transform import BoundedParameterTransform

    bt = BoundedParameterTransform(np.array([0.1, 0.25]), np.array([0.4, 0.29]))
    fs = Standardizer().fit(x[train])
    ys = Standardizer().fit(bt.forward(y[train]))
    xs = fs.transform(x)
    yt = ys.transform(bt.forward(y))
    cfg = MDNConfig(
        n_components=3,
        hidden_dim=32,
        n_hidden_layers=2,
        batch_size=16,
        max_epochs=12,
        patience=4,
        seed=11,
    )
    model, report = fit_conditional_mdn(xs[train], yt[train], xs[val], yt[val], config=cfg)
    samples = draw_physical_posterior(
        model,
        x[val[:3]],
        n_samples=32,
        seed=5,
        feature_state=fs.state_dict(),
        target_state=ys.state_dict(),
        bounds_state=bt.state_dict(),
    )
    assert samples.shape == (3, 32, 2)
    assert np.isfinite(samples).all()
    assert samples[..., 0].min() >= 0.1 and samples[..., 0].max() <= 0.4
    assert report["best_epoch"] >= 1 and np.isfinite(
        report["best_validation_nll_transformed_space"]
    )


def test_mdn_grouped_npz_workflow_never_touches_test_by_default(tmp_path):
    rng = np.random.default_rng(9)
    n = 120
    mass = np.tile(np.array([0.1, 0.2, 0.3, 0.4]), n // 4)
    x = np.column_stack(
        [mass + rng.normal(0, 0.03, n), rng.normal(size=n), mass**2 + rng.normal(0, 0.02, n)]
    ).astype("f4")
    omega = rng.uniform(0.25, 0.35, n)
    theta = np.column_stack([mass, omega])
    ids = np.array([f"sim{i}" for i in range(n)])
    groups = np.array([f"parent{i}" for i in range(n)])
    dataset = tmp_path / "toy.npz"
    np.savez_compressed(dataset, features=x, theta=theta, simulation_id=ids, group_id=groups)
    dataset.with_suffix(".json").write_text(
        json.dumps({"physics_level": "toy", "dataset_id": "toy-test"})
    )
    report = train_mdn_from_npz(
        dataset,
        tmp_path / "result",
        parameter_names=["sum_mnu_ev", "omega_m"],
        prior_low=[0.1, 0.2],
        prior_high=[0.4, 0.4],
        seed=7,
        fractions=(0.7, 0.15, 0.15),
        config=MDNConfig(
            n_components=3,
            hidden_dim=24,
            n_hidden_layers=1,
            batch_size=16,
            max_epochs=8,
            patience=3,
            seed=7,
        ),
        n_validation_observations=5,
        posterior_samples=32,
        allow_toy=True,
        evaluate_test_once=False,
    )
    assert report["test_evaluated_once"] is False
    assert report["final_test_posterior_metrics"] is None
    assert (tmp_path / "result" / "mdn_checkpoint.pt").is_file()
    heldout = tmp_path / "result" / "heldout_posterior_samples.npz"
    assert not heldout.exists()
    val = np.load(tmp_path / "result" / "validation_posterior_samples.npz")
    assert val["samples"].shape == (5, 32, 2)
    assert val["samples"][..., 0].min() >= 0.1 and val["samples"][..., 0].max() <= 0.4
    cfg = json.loads((tmp_path / "result" / "mdn_report.json").read_text())
    assert cfg["scientific_approval"] is False
    assert cfg["preprocessing"]["fit_on"] == "inner gradient-fit groups only"
    assert cfg["model_selection_validation_is_not_used_for_early_stopping"] is True
    assert (
        cfg["inner_training_group_hashes"]["fit"]
        != cfg["inner_training_group_hashes"]["early_stopping"]
    )


def test_mdn_workflow_rejects_toy_without_explicit_smoke_test_override(tmp_path):
    n = 20
    x = np.ones((n, 2))
    m = np.tile([0.1, 0.2, 0.3, 0.4], 5)
    theta = m[:, None]
    ds = tmp_path / "data.npz"
    np.savez(
        ds,
        features=x,
        theta=theta,
        group_id=np.array([str(i) for i in range(n)]),
        simulation_id=np.array([str(i) for i in range(n)]),
    )
    ds.with_suffix(".json").write_text('{"physics_level":"toy"}')
    with pytest.raises(ValueError, match="toy"):
        train_mdn_from_npz(
            ds, tmp_path / "o", parameter_names=["sum_mnu_ev"], prior_low=[0.1], prior_high=[0.4]
        )


def test_mdn_checkpoint_roundtrip_and_prediction_cli(tmp_path, capsys):
    from nufield.models.conditional_mdn import draw_physical_posterior, load_mdn_checkpoint
    from nufield.training.predict_mdn import main as predict_main

    rng = np.random.default_rng(44)
    n = 80
    mass = np.tile([0.1, 0.2, 0.3, 0.4], 20)
    x = np.column_stack([mass + rng.normal(0, 0.02, n), rng.normal(size=n)]).astype("f4")
    theta = np.column_stack([mass, rng.uniform(0.25, 0.35, n)])
    ds = tmp_path / "train.npz"
    np.savez_compressed(
        ds,
        features=x,
        theta=theta,
        group_id=np.array([f"g{i}" for i in range(n)]),
        simulation_id=np.array([f"s{i}" for i in range(n)]),
    )
    ds.with_suffix(".json").write_text('{"physics_level":"toy"}')
    result = tmp_path / "fit"
    train_mdn_from_npz(
        ds,
        result,
        parameter_names=["sum_mnu_ev", "omega_m"],
        prior_low=[0.1, 0.2],
        prior_high=[0.4, 0.4],
        allow_toy=True,
        config=MDNConfig(
            n_components=2,
            hidden_dim=16,
            n_hidden_layers=1,
            batch_size=16,
            max_epochs=4,
            patience=2,
            seed=8,
        ),
        n_validation_observations=3,
        posterior_samples=16,
    )
    model, ckpt = load_mdn_checkpoint(str(result / "mdn_checkpoint.pt"))
    samples = draw_physical_posterior(
        model,
        x[:2],
        n_samples=12,
        seed=2,
        feature_state=ckpt["feature_scaler"],
        target_state=ckpt["target_scaler"],
        bounds_state=ckpt["bounded_transform"],
    )
    assert samples.shape == (2, 12, 2)
    pred = tmp_path / "pred_input.npz"
    np.savez(pred, features=x[:3], simulation_id=np.array(["o0", "o1", "o2"]))
    out = tmp_path / "posterior.npz"
    assert (
        predict_main(
            [
                "--checkpoint",
                str(result / "mdn_checkpoint.pt"),
                "--observations",
                str(pred),
                "--n-samples",
                "16",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert np.load(out)["samples"].shape == (3, 16, 2)
