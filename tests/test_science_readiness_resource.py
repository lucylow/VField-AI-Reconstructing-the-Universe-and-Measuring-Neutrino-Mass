import numpy as np
import pytest

from nufield.data_registry.cards import DatasetCard
from nufield.data_registry.resource_estimator import estimate_storage, estimate_training_memory
from nufield.data_registry.science_readiness import assess_science_readiness


def _card(**updates):
    values = dict(
        dataset_id="q-test",
        source_id="quijote_mnu",
        title="Quijote mass grid",
        release="release-a",
        source_url="https://example.org",
        license_name="check access",
        local_path="/tmp/not-used.npz",
        storage_format="npz",
        observable_kind="density field",
        observable_key="fields",
        parameter_key="theta",
        parameter_names=["Mnu", "Omega_m"],
        neutrino_mass_parameter="Mnu",
        mass_range_ev=[0.06, 0.4],
        neutrino_mass_varies=True,
        redshift=0.0,
        box_size_mpc_h=1000.0,
        simulation_code="documented N-body",
        density_definition="matter overdensity",
        units={"field": "delta", "box": "Mpc/h"},
        citation="Cite source",
    )
    values.update(updates)
    return DatasetCard(**values)


def test_science_readiness_never_auto_approves():
    c = _card()
    theta = np.column_stack([np.tile([0.06, 0.2, 0.4], 10), np.linspace(0.25, 0.35, 30)])
    ids = np.array([f"sim{i}" for i in range(30)])
    groups = np.array([f"g{i}" for i in range(30)])
    report = assess_science_readiness(
        c,
        theta=theta,
        observed_parameter_names=c.parameter_names,
        simulation_ids=ids,
        group_ids=groups,
    )
    assert report.scientific_approval is False
    assert report.checklist_stage == "array_contract_met_physics_unverified"
    assert any(i.key == "mass_varies_in_arrays" and i.satisfied for i in report.items)


def test_science_readiness_blocks_fixed_mass_and_missing_ids():
    c = _card(neutrino_mass_varies=False, mass_range_ev=[0.06, 0.06])
    theta = np.column_stack([np.full(12, 0.06), np.linspace(0.25, 0.35, 12)])
    report = assess_science_readiness(c, theta=theta, observed_parameter_names=c.parameter_names)
    assert report.blockers
    assert any("mass_varies_in_card" in b for b in report.blockers)
    assert not report.scientific_approval


def test_storage_and_memory_estimates_scale_with_inputs():
    small = estimate_storage(10, (1, 8, 8, 8), compression_factor=1.0)
    large = estimate_storage(20, (1, 8, 8, 8), compression_factor=1.0)
    assert large.raw_bytes == 2 * small.raw_bytes
    cnn = estimate_training_memory("cnn3d", batch_size=2, field_shape=(1, 16, 16, 16))
    fno = estimate_training_memory("fno3d", batch_size=2, field_shape=(1, 16, 16, 16))
    assert cnn.rough_total_bytes > 0 and fno.rough_total_bytes > 0
    assert fno.rough_total_bytes != cnn.rough_total_bytes
    with pytest.raises(ValueError):
        estimate_storage(5, (0, 8, 8))
