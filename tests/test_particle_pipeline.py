import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from nufield.data_registry.particle_pipeline import process_hdf5_particle_batch
from nufield.data_registry.particle_pipeline_cli import main


def _write_fixture(path):
    rng = np.random.default_rng(2026)
    positions = rng.uniform(0.0, 20.0, size=(4, 120, 3)).astype("f4")
    theta = np.asarray([[0.1, 0.30], [0.2, 0.31], [0.3, 0.32], [0.4, 0.33]], dtype="f8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("sim/positions", data=positions)
        h5.create_dataset("params/theta", data=theta)
        h5.create_dataset("metadata/simulation_id", data=np.asarray([b"s0", b"s1", b"s2", b"s3"]))
        h5.create_dataset("metadata/group_id", data=np.asarray([b"g0", b"g1", b"g2", b"g3"]))


def test_particle_pipeline_converts_a_mapped_periodic_hdf5_product(tmp_path):
    source = tmp_path / "fixture.h5"
    _write_fixture(source)
    out = tmp_path / "derived"
    manifest = process_hdf5_particle_batch(
        source,
        out,
        positions_key="sim/positions",
        theta_key="params/theta",
        parameter_names=["sum_mnu_ev", "omega_m"],
        simulation_id_key="metadata/simulation_id",
        group_id_key="metadata/group_id",
        box_size_mpc_h=20.0,
        grid_size=16,
        redshift=0.5,
        power_bins=4,
        max_memory_gb=0.25,
    )
    fields = np.load(out / "fields.npy", mmap_mode="r")
    features = np.load(out / "features.npy", mmap_mode="r")
    theta = np.load(out / "theta.npy", mmap_mode="r")
    assert fields.shape == (4, 16, 16, 16)
    assert features.shape[0] == 4 and features.shape[1] > 0
    assert theta.shape == (4, 2)
    assert manifest["scientific_approval"] is False
    assert len(manifest["source_sha256"]) == 64
    saved = json.loads((out / "dataset_manifest.json").read_text())
    assert saved["source_keys"]["positions"] == "sim/positions"
    assert saved["physics_level"] == "prototype_requires_reference_validation"


def test_particle_pipeline_refuses_implicit_overwrite(tmp_path):
    source = tmp_path / "fixture.h5"
    _write_fixture(source)
    out = tmp_path / "derived"
    out.mkdir()
    (out / "keep.txt").write_text("existing")
    with pytest.raises(FileExistsError):
        process_hdf5_particle_batch(
            source,
            out,
            positions_key="sim/positions",
            theta_key="params/theta",
            parameter_names=["sum_mnu_ev", "omega_m"],
            simulation_id_key="metadata/simulation_id",
            group_id_key="metadata/group_id",
            box_size_mpc_h=20.0,
            grid_size=16,
            redshift=0.5,
            power_bins=4,
            max_memory_gb=0.25,
        )


def test_particle_pipeline_cli_argument_mapping(tmp_path, capsys):
    source = tmp_path / "fixture.h5"
    _write_fixture(source)
    out = tmp_path / "cli-derived"
    code = main(
        [
            str(source),
            "--output-dir",
            str(out),
            "--positions-key",
            "sim/positions",
            "--theta-key",
            "params/theta",
            "--parameter-names",
            "sum_mnu_ev,omega_m",
            "--simulation-id-key",
            "metadata/simulation_id",
            "--group-id-key",
            "metadata/group_id",
            "--box-size-mpc-h",
            "20",
            "--grid-size",
            "16",
            "--redshift",
            "0.5",
            "--power-bins",
            "4",
            "--max-memory-gb",
            "0.25",
        ]
    )
    parsed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert parsed["n_realizations"] == 4
    assert parsed["scientific_approval"] is False
