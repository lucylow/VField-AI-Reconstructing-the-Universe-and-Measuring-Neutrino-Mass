import numpy as np

from nufield.data_registry.product_contract import DatasetProductContract, review_dataset_product


def _contract(**changes):
    fields = dict(
        dataset_id="q-mnu",
        source_family="Quijote",
        exact_product_id="Mnu-nwLH-z0.5",
        release="docs-pinned-2026-10",
        source_url="https://example.org/data",
        data_kind="simulation",
        intended_role="primary_target_training",
        license_or_terms="verified research use terms",
        product_schema_verified=True,
        source_cosmology_documented=True,
        parameter_names=("sum_mnu_ev", "omega_m"),
        parameter_units=("eV", "dimensionless"),
        target_name="sum_mnu_ev",
        target_unit="eV",
        target_values_verified=True,
        target_grid_kind="continuous",
        independent_group_definition="independent phase IDs",
        source_has_known_truth=True,
        observed_product_has_target_truth=False,
        field_geometry="periodic_box",
        coordinate_and_length_units_documented=True,
        redshift_or_snapshot_documented=True,
        product_notes=(),
    )
    fields.update(changes)
    return DatasetProductContract(**fields)


def test_exact_product_with_variable_mass_and_groups_is_eligible_for_preliminary_role():
    c = _contract()
    mass = np.repeat([0.1, 0.2, 0.3, 0.4], 4)
    omega = np.linspace(0.25, 0.35, len(mass))
    theta = np.column_stack([mass, omega])
    groups = np.array([f"g{i}" for i in range(len(mass))])
    r = review_dataset_product(
        c, theta, parameter_names=list(c.parameter_names), group_ids=groups, prior_bounds=(0.1, 0.4)
    )
    assert r["eligible_for_declared_role"]
    assert r["scientific_approval"] is False


def test_observed_data_cannot_become_supervised_training_by_contract_flag():
    c = _contract(data_kind="observed", source_has_known_truth=False)
    theta = np.column_stack([np.tile([0.1, 0.2, 0.3, 0.4], 3), np.full(12, 0.3)])
    r = review_dataset_product(
        c,
        theta,
        parameter_names=list(c.parameter_names),
        group_ids=np.arange(12).astype(str),
        prior_bounds=(0.1, 0.4),
    )
    assert not r["eligible_for_declared_role"]
    assert any("observed data" in e for e in r["errors"])


def test_sparse_mass_labels_warn_even_if_basic_schema_is_valid():
    c = _contract(target_grid_kind="discrete")
    theta = np.column_stack([np.repeat([0.1, 0.4], 6), np.linspace(0.25, 0.35, 12)])
    r = review_dataset_product(
        c,
        theta,
        parameter_names=list(c.parameter_names),
        group_ids=np.arange(12).astype(str),
        prior_bounds=(0.1, 0.4),
    )
    assert any("need >= 3" in e for e in r["errors"])
