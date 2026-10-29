import numpy as np

from nufield.data_registry.cards import DatasetCard
from nufield.data_registry.manifest import inventory_files, sha256_file
from nufield.data_registry.preflight import preflight


def make_card(status="arrays_checked"):
    return DatasetCard(
        dataset_id="example",
        source_id="quijote_mnu",
        title="test",
        release="r1",
        source_url="https://example.org",
        license_name="verify",
        local_path="/tmp/f.npz",
        storage_format="npz",
        observable_kind="3d_field",
        observable_key="fields",
        parameter_key="theta",
        parameter_names=["Mnu", "Omega_m"],
        neutrino_mass_parameter="Mnu",
        mass_range_ev=[0.06, 0.3],
        neutrino_mass_varies=True,
        status=status,
    )


def test_file_manifest_hashes_source(tmp_path):
    f = tmp_path / "raw.bin"
    f.write_bytes(b"nufield-test")
    assert len(sha256_file(f)) == 64
    row = inventory_files([f])[0]
    assert row["size_bytes"] == len(b"nufield-test")
    assert row["sha256"] == sha256_file(f)


def test_preflight_never_automatically_approves_physical_claim():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(300, 6))
    theta = np.column_stack([np.linspace(0.06, 0.3, 300), rng.uniform(0.27, 0.36, 300)])
    ids = np.array([f"sim-{i}" for i in range(len(theta))])
    report = preflight(
        make_card("arrays_checked"),
        x,
        theta,
        simulation_ids=ids,
        group_ids=ids,
        prior_low=[0.06, 0.27],
        prior_high=[0.3, 0.36],
    )
    assert report.ready_for_neural_training
    assert not report.ready_for_physical_mnu_claim
