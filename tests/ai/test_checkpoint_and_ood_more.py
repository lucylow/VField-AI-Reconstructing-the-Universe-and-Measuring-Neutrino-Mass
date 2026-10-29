import numpy as np
import pytest

from nufield.ai.checkpoints import create_checkpoint_manifest, file_sha256
from nufield.ai.ood import MahalanobisOOD
from nufield.ai.variance_calibration import VarianceTemperature


def test_checkpoint_manifest_requires_complete_scientific_metadata(tmp_path):
    path = tmp_path / "state.pt"
    path.write_bytes(b"weight bytes")
    with pytest.raises(ValueError, match="units"):
        create_checkpoint_manifest(
            path,
            model_name="test",
            model_config={},
            training_config={},
            data_fingerprint="a",
            split_fingerprint="b",
            parameter_names=["mnu"],
            parameter_units={},
            best_epoch=1,
            best_validation_loss=0.2,
            simulator_status="toy",
        )
    assert len(file_sha256(path)) == 64


def test_mahalanobis_ood_rejects_too_small_calibration():
    det = MahalanobisOOD().fit(np.random.default_rng(50).normal(size=(30, 2)))
    with pytest.raises(ValueError, match="at least 10"):
        det.calibrate(np.zeros((5, 2)))


def test_variance_calibration_never_changes_mean():
    rng = np.random.default_rng(51)
    y = rng.normal(size=(100, 1))
    mean = y + rng.normal(size=(100, 1)) * 0.4
    var = np.ones_like(y) * 0.1
    fit = VarianceTemperature.fit(mean, var, y, grid_size=15)
    adjusted = fit.apply(var)
    assert np.all(adjusted > 0)
    assert np.allclose(mean, mean)
