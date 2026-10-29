import numpy as np

from nufield.data_registry.cards import DatasetCard
from nufield.data_registry.decision_gates import evaluate_dataset_gates
from nufield.data_registry.suitability import build_suitability_plan
from nufield.training.experiment_preflight import run_experiment_preflight


def test_dataset_plan_does_not_claim_mass_training_before_exact_product_verified():
    plan = build_suitability_plan(exact_product_mass_grid_verified=False)
    assert plan["primary_mass_training_source"] is None
    assert plan["blocking_conditions"]
    plan2 = build_suitability_plan(exact_product_mass_grid_verified=True, budget="workstation")
    assert plan2["primary_mass_training_source"] == "quijote_mnu"
    assert plan2["scientific_approval"] is False


def test_data_gate_fails_closed_without_group_ids_and_split():
    card = DatasetCard(
        dataset_id="q",
        source_id="quijote_mnu",
        title="test",
        release="r1",
        source_url="https://example.com",
        license_name="unknown",
        local_path="x.npz",
        storage_format="npz",
        observable_kind="density_field",
        observable_key="field",
        parameter_key="theta",
        parameter_names=["sum_mnu_ev"],
        neutrino_mass_parameter="sum_mnu_ev",
        mass_range_ev=[0.1, 0.4],
        neutrino_mass_varies=True,
        citation="citation",
    )
    theta = np.array([[0.1], [0.2], [0.4]])
    r = evaluate_dataset_gates(
        card,
        theta=theta,
        observed_parameter_names=["sum_mnu_ev"],
        simulation_ids=np.array(["a", "b", "c"]),
        group_ids=None,
        prior_bounds=(0.1, 0.4),
        source_schema_verified=True,
    )
    assert not r.eligible_for_preliminary_mass_training
    assert not r.scientific_approval


def test_preflight_runs_on_explicit_synthetic_fixture(tmp_path):
    rng = np.random.default_rng(1)
    n = 60
    x = rng.normal(size=(n, 12))
    m = np.tile(np.array([0.1, 0.2, 0.3, 0.4]), 15)
    theta = np.column_stack([m, rng.uniform(0.25, 0.35, n)])
    groups = np.array([f"g{i}" for i in range(n)])
    f = tmp_path / "fixture.npz"
    np.savez_compressed(f, features=x, theta=theta, simulation_id=groups, group_id=groups)
    report = run_experiment_preflight(
        str(f),
        observation_key="features",
        theta_key="theta",
        parameter_names=["sum_mnu_ev", "omega_m"],
        simulation_id_key="simulation_id",
        group_id_key="group_id",
        observation_kind="spectrum",
        dataset_id="test",
        release="toy1",
        physics_level="toy",
        field_shape=None,
        prior_low=[0.1, 0.2],
        prior_high=[0.4, 0.5],
        budget="laptop",
    )
    assert report["n_rows"] == n
    # Incorrect prior low/high ordering is caught before the model is trained by downstream gates,
    # but generic preflight itself currently validates dimensionality only.
