import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from nufield.data_registry.field_batch_pipeline import process_hdf5_field_batch


def _fixture(path):
    rng = np.random.default_rng(2027)
    fields = rng.normal(size=(6, 16, 16, 16)).astype("f4")
    theta = np.column_stack([np.tile([0.1, 0.2, 0.3], 2), rng.uniform(0.25, 0.35, 6)])
    with h5py.File(path, "w") as h5:
        h5.create_dataset("maps/delta", data=fields, chunks=(1, 16, 16, 16), compression="gzip")
        h5.create_dataset("theta", data=theta)
        h5.create_dataset("simulation_id", data=np.asarray([f"s{i}".encode() for i in range(6)]))
        h5.create_dataset("group_id", data=np.asarray([f"g{i}".encode() for i in range(6)]))


def test_streaming_field_pipeline_writes_ai_ready_arrays(tmp_path):
    f = tmp_path / "field.h5"
    _fixture(f)
    out = tmp_path / "processed"
    report = process_hdf5_field_batch(
        f,
        out,
        fields_key="maps/delta",
        theta_key="theta",
        parameter_names=["sum_mnu_ev", "omega_m"],
        simulation_id_key="simulation_id",
        group_id_key="group_id",
        box_size_mpc_h=500,
        redshift=0.5,
        field_kind="overdensity",
        power_bins=4,
        max_memory_gb=0.25,
    )
    x = np.load(out / "features.npy", mmap_mode="r")
    fields = np.load(out / "fields.npy", mmap_mode="r")
    assert x.shape[0] == 6 and x.shape[1] > 0
    assert fields.shape == (6, 1, 16, 16, 16)
    assert report["scientific_approval"] is False
    assert (
        json.loads((out / "dataset_manifest.json").read_text())["output_arrays"]["fields"]
        == "fields.npy"
    )


def test_streaming_field_pipeline_can_skip_field_copy(tmp_path):
    f = tmp_path / "field.h5"
    _fixture(f)
    out = tmp_path / "processed"
    result = process_hdf5_field_batch(
        f,
        out,
        fields_key="maps/delta",
        theta_key="theta",
        parameter_names=["sum_mnu_ev", "omega_m"],
        simulation_id_key="simulation_id",
        group_id_key="group_id",
        box_size_mpc_h=500,
        redshift=0.5,
        field_kind="overdensity",
        power_bins=4,
        max_memory_gb=0.25,
        copy_fields=False,
    )
    assert result["copied_fields"] is False
    assert not (out / "fields.npy").exists()
    assert (out / "features.npy").is_file()


def test_streaming_field_pipeline_requires_explicitly_valid_geometry(tmp_path):
    f = tmp_path / "field.h5"
    _fixture(f)
    with pytest.raises(ValueError, match="field_kind"):
        process_hdf5_field_batch(
            f,
            tmp_path / "out",
            fields_key="maps/delta",
            theta_key="theta",
            parameter_names=["sum_mnu_ev", "omega_m"],
            simulation_id_key="simulation_id",
            group_id_key="group_id",
            box_size_mpc_h=500,
            redshift=0.5,
            field_kind="observed_desi",
            power_bins=4,
        )
