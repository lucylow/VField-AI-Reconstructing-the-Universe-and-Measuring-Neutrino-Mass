import numpy as np
import pytest

from nufield.ai.feature_engineering import (
    SpectralFeaturePipeline,
    SpectrumContract,
    smooth_broadband_component,
    spectral_slopes,
)
from nufield.ai.gaussian_process_emulator import GaussianProcessEmulator, kernel_matrix
from nufield.ai.priors import ParameterPrior, PriorSpecification, design_quality_summary


def test_prior_sampling_log_density_and_design_quality():
    prior = PriorSpecification(
        (
            ParameterPrior("mnu", 0.06, 0.5, "eV"),
            ParameterPrior("sigma8", 0.6, 1.0, "dimensionless"),
        ),
        "test",
    )
    samples = prior.sample(128, method="sobol", seed=42)
    assert samples.shape == (128, 2)
    assert np.all(samples >= prior.low) and np.all(samples <= prior.high)
    lp = prior.log_prob(samples[:3])
    assert np.all(np.isfinite(lp))
    assert design_quality_summary(samples, prior)["n_points"] == 128
    log_prior = PriorSpecification((ParameterPrior("mass", 0.01, 1.0, "eV", "log_uniform"),))
    assert np.all(np.isfinite(log_prior.log_prob(log_prior.sample(5, seed=1))))
    with pytest.raises(ValueError):
        prior.validate([[0.0, 0.8]])


def test_spectrum_contract_and_train_only_feature_pipeline():
    rng = np.random.default_rng(60)
    k = np.geomspace(0.01, 1.0, 32)
    power = np.exp(-k)[None, :] * rng.lognormal(0, 0.1, size=(80, 32))
    power[0, 4] = -0.2
    contract = SpectrumContract(tuple(k), "h/Mpc", "(Mpc/h)^3", "isotropic P(k)", box_size=500)
    pipeline = SpectralFeaturePipeline(include_validity_mask=True).fit(power[:60], contract)
    features = pipeline.transform(power[60:])
    assert features.shape == (20, 64) and np.isfinite(features).all()
    assert len(pipeline.to_dict()["floor"]) == 32
    assert spectral_slopes(k, np.exp(-k)).shape == (32,)
    assert smooth_broadband_component(np.exp(-k)).shape == (32,)
    with pytest.raises(ValueError):
        SpectralFeaturePipeline().fit(power, SpectrumContract(tuple(k[::-1]), "k", "P", "bad"))


def test_gp_emulator_recovers_smooth_toy_function():
    rng = np.random.default_rng(61)
    x = rng.uniform(-1, 1, size=(45, 2))
    y = np.column_stack([np.sin(2 * x[:, 0]) + x[:, 1] ** 2, x[:, 0] * x[:, 1]])
    gp = GaussianProcessEmulator(length_scale=0.7, amplitude=1.0, noise_variance=1e-5).fit(x, y)
    test = rng.uniform(-0.8, 0.8, size=(12, 2))
    mean, var = gp.predict(test)
    truth = np.column_stack([np.sin(2 * test[:, 0]) + test[:, 1] ** 2, test[:, 0] * test[:, 1]])
    assert mean.shape == var.shape == truth.shape
    assert np.sqrt(np.mean((mean - truth) ** 2)) < 0.25
    assert np.all(var >= 0) and gp.summary()["n_train"] == 45
    _, cov = gp.predict(test[:4], return_covariance=True)
    assert cov.shape == (2, 4, 4)


def test_kernel_matrix_and_gp_reject_bad_inputs():
    with pytest.raises(ValueError):
        kernel_matrix([[0]], [[1]], length_scale=0)
    with pytest.raises(ValueError):
        GaussianProcessEmulator().fit(np.ones((2, 3)), np.ones(4))


def test_hdf5_streaming_dataset_is_lazy_and_worker_safe(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("torch")
    from nufield.ai.data_loading import (
        ArraySimulationDataset,
        HDF5SimulationDataset,
        estimate_array_memory_bytes,
    )

    path = tmp_path / "sim.h5"
    with h5py.File(path, "w") as h:
        h.create_dataset("x", data=np.arange(60, dtype=np.float32).reshape(20, 3))
        h.create_dataset("y", data=np.arange(20, dtype=np.float32).reshape(20, 1))
        h.create_dataset("g", data=np.asarray([f"g{i}" for i in range(20)], dtype="S8"))
    ds = HDF5SimulationDataset(path, input_key="x", target_key="y", group_key="g")
    item = ds[2]
    assert len(item) == 3 and item[0].shape == (3,) and item[2] == "g2"
    ds.close()
    arr_ds = ArraySimulationDataset(np.zeros((10, 4)), np.ones((10, 2)))
    assert len(arr_ds) == 10 and arr_ds[0][0].shape == (4,)
    memory = estimate_array_memory_bytes({"field": (100, 32, 32, 32)}, workers=2)
    assert memory["lower_bound_peak_gib"] > 0
