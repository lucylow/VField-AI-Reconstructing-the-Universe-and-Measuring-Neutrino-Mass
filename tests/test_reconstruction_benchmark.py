import json

import numpy as np
import pytest

from nufield.reconstruction.benchmark import (
    _scores,
    _standardize_modalities,
    load_reconstruction_dataset,
    run_reconstruction_benchmark,
)
from nufield.reconstruction.config import ReconstructionConfig, load_reconstruction_config


def _fixture(path, *, bad=False, masked=False):
    rng = np.random.default_rng(14)
    n, g = 48, 8
    truth = rng.normal(size=(n, g, g, g)).astype("f4")
    noise = rng.normal(scale=0.2, size=truth.shape).astype("f4")
    galaxy = truth + noise
    lens = truth.mean(axis=3) + rng.normal(scale=0.3, size=(n, g, g)).astype("f4")
    cmb = rng.normal(size=(n, 2)).astype("f4")
    groups = np.asarray([f"parent-{i // 2}" for i in range(n)])
    sim = np.asarray([f"sim-{i}" for i in range(n)])
    if bad:
        galaxy[0, 0, 0, 0] = np.nan
    arrays = dict(initial_field=truth, galaxy_overdensity=galaxy, weak_lensing_map=lens,
                  cmb_features=cmb, simulation_id=sim, group_id=groups)
    if masked:
        mask = np.ones((g, g, g), dtype="u1")
        mask[:, :, 0] = 0
        arrays["survey_mask"] = mask
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps({
        "schema_version": "1.0", "data_type": "synthetic test fixture",
        "is_synthetic": True, "license": "generated for tests",
        "array_units": {"initial_field": "arbitrary dimensionless", "galaxy_overdensity": "arbitrary dimensionless",
                        "weak_lensing_map": "arbitrary dimensionless", "cmb_features": "arbitrary dimensionless"},
        "coordinate_system": "grid indices", "coordinate_unit": "dimensionless cells",
        "physics_level": "test mock",
    }))
    return truth


def test_modalities_are_normalized_on_train_rows_only():
    x = np.zeros((8, 1, 2, 2, 2), dtype="f4")
    x[:6] = 2
    x[6:] = 1000
    z, state = _standardize_modalities({"galaxy": x}, np.arange(6))
    assert state["galaxy"]["mean"] == [2.0]
    assert np.all(z["galaxy"][:6] == 0)
    assert np.all(z["galaxy"][6:] > 100)


def test_masked_out_voxels_do_not_change_test_metrics():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(1, 8, 8, 8)).astype("f4")
    pred = (truth + rng.normal(scale=0.1, size=truth.shape)).astype("f4")
    mask = np.ones_like(truth, dtype=bool)
    mask[:, :, :, :3] = False
    baseline = _scores(pred, truth, mask)
    changed = pred.copy()
    changed[~mask] = 1e5
    after = _scores(changed, truth, mask)
    assert after == baseline


def test_load_rejects_nan_and_run_writes_group_safe_ablation_artifacts(tmp_path):
    bad = tmp_path / "bad.npz"
    _fixture(bad, bad=True)
    with pytest.raises(ValueError, match="NaN or infinity"):
        load_reconstruction_dataset(ReconstructionConfig(dataset=str(bad), run_neural=False))

    path = tmp_path / "paired.npz"
    _fixture(path, masked=True)
    cfg = ReconstructionConfig(dataset=str(path), output_dir=str(tmp_path / "out"), seed=9, run_neural=False)
    report = run_reconstruction_benchmark(cfg)
    assert report["scientific_approval"] is False
    assert report["group_disjoint"] is True
    assert set(report["results"]) == {"linear_galaxy", "linear_galaxy+lensing", "linear_galaxy+cmb", "linear_galaxy+lensing+cmb"}
    assert all("rmse" in r["metrics"] for r in report["results"].values())
    assert all("noise_stress" in r for r in report["results"].values())
    assert (tmp_path / "out" / "metrics.csv").is_file()
    with np.load(tmp_path / "out" / "predictions.npz", allow_pickle=False) as z:
        assert "truth" in z and "linear_galaxy" in z
        assert len(z["truth"]) == report["n_rows"]["test"]


def test_repeated_benchmark_reuses_identical_split_and_metrics(tmp_path):
    path = tmp_path / "paired.npz"
    _fixture(path)
    cfg = ReconstructionConfig(dataset=str(path), output_dir=str(tmp_path / "one"), seed=11, run_neural=False)
    first = run_reconstruction_benchmark(cfg)
    cfg2 = ReconstructionConfig(dataset=str(path), output_dir=str(tmp_path / "two"), seed=11, run_neural=False)
    second = run_reconstruction_benchmark(cfg2)
    assert first["split_ids"] == second["split_ids"]
    assert first["results"]["linear_galaxy"]["metrics"] == second["results"]["linear_galaxy"]["metrics"]


def test_config_validation_and_yaml_unknown_key(tmp_path):
    with pytest.raises(ValueError):
        ReconstructionConfig(dataset="x", train_fraction=0.5)
    p = tmp_path / "bad.yaml"
    p.write_text("reconstruction:\n  dataset: x.npz\n  typo: true\n")
    with pytest.raises(Exception, match="Unknown reconstruction setting"):
        load_reconstruction_config(p)


def test_loader_rejects_bad_mask_and_malformed_field_shape(tmp_path):
    p = tmp_path / "data.npz"
    _fixture(p)
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["survey_mask"] = np.full((8, 8, 8), 2, dtype="u1")
    np.savez_compressed(p, **arrays)
    with pytest.raises(ValueError, match="binary"):
        load_reconstruction_dataset(ReconstructionConfig(dataset=str(p), run_neural=False))


def test_loader_requires_provenance_and_config_requires_box_for_power_loss(tmp_path):
    p = tmp_path / "data.npz"
    _fixture(p)
    p.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="Missing provenance sidecar"):
        load_reconstruction_dataset(ReconstructionConfig(dataset=str(p), run_neural=False))
    with pytest.raises(ValueError, match="box_size_mpc_h"):
        ReconstructionConfig(dataset="x", lambda_pk=0.1)


def test_optional_unet_checkpoint_can_resume(tmp_path):
    pytest.importorskip("torch")
    from nufield.reconstruction.train import train_unet3d

    rng = np.random.default_rng(22)
    x = rng.normal(size=(2, 8, 8, 8)).astype("f4")
    y = (x * 0.7).astype("f4")
    checkpoint = tmp_path / "resume.pt"
    first = train_unet3d(x, y, epochs=1, batch_size=2, seed=5, checkpoint_path=checkpoint,
                         provenance={"dataset_sha256": "fixture"})
    resumed = train_unet3d(x, y, epochs=2, batch_size=2, seed=5, resume_from=checkpoint,
                           checkpoint_path=checkpoint, provenance={"dataset_sha256": "fixture"})
    assert len(first["history"]) == 1
    assert resumed["start_epoch"] == 1 and len(resumed["history"]) == 2
    assert {"train_mse", "train_loss"} <= set(resumed["history"][-1])
    with pytest.raises(ValueError, match="provenance differs"):
        train_unet3d(x, y, epochs=2, batch_size=2, seed=5, resume_from=checkpoint,
                     provenance={"dataset_sha256": "changed"})
