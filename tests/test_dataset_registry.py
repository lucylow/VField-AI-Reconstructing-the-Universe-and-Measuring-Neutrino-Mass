import numpy as np
import pytest

from nufield.data_registry.assessment import assess_arrays
from nufield.data_registry.cards import DatasetCard, load_dataset_card, write_dataset_card
from nufield.data_registry.catalog import get_dataset
from nufield.data_registry.selector import recommend_datasets
from nufield.data_registry.splits import (
    SplitError,
    assert_group_disjoint,
    group_train_val_test_split,
)


def test_quijote_is_first_recommendation_for_mnu():
    rows = recommend_datasets("neutrino_mass_inference", max_results=5)
    assert rows[0].source_id == "quijote_mnu"
    assert rows[0].suitable_for_direct_mnu_training
    assert get_dataset("camels").mass_variation == "not_default"
    q = get_dataset("quijote_mnu")
    assert any("nwLH" in item and "2,000" in item for item in q.suggested_products)
    assert any("not a recommendation for the inference prior" in item for item in q.avoid_claiming)
    camels = get_dataset("camels")
    assert any("Mnu = 0.0 eV" in item for item in camels.limitations)
    assert any("parameters.html" in item for item in camels.citations)


def test_desi_is_survey_validation_not_direct_mnu_training():
    desi = get_dataset("desi_dr1_mocks")
    assert "survey_validation" in desi.objectives
    assert desi.mass_variation == "limited"
    assert any(
        "not" in warning.lower() or "grid" in warning.lower() for warning in desi.limitations
    )


def test_assessment_rejects_constant_mass_target():
    x = np.random.default_rng(1).normal(size=(20, 2, 4, 4, 4))
    theta = np.column_stack([np.full(20, 0.1), np.linspace(0.2, 0.4, 20)])
    report = assess_arrays(x, theta, parameter_names=["Mnu", "Omega_m"])
    assert not report.can_train_mnu_posterior
    assert any(f.code == "MNU_NOT_IDENTIFIABLE" for f in report.findings)


def test_assessment_passes_basic_varying_mass_contract():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(250, 12))
    theta = np.column_stack([np.linspace(0.06, 0.3, 250), rng.uniform(0.27, 0.36, 250)])
    report = assess_arrays(x, theta, parameter_names=["Mnu", "Omega_m"])
    assert report.can_train_mnu_posterior
    assert not report.scientific_approval


def test_card_round_trip(tmp_path):
    card = DatasetCard(
        dataset_id="small",
        source_id="quijote_mnu",
        title="test",
        release="r1",
        source_url="https://example.org",
        license_name="verify",
        local_path="/tmp/data.npz",
        storage_format="npz",
        observable_kind="field",
        observable_key="fields",
        parameter_key="theta",
        parameter_names=["Mnu"],
        mass_range_ev=[0.06, 0.3],
        neutrino_mass_varies=True,
    )
    path = write_dataset_card(card, tmp_path / "card.yaml")
    loaded = load_dataset_card(path)
    assert loaded.dataset_id == "small"
    assert loaded.neutrino_mass_varies


def test_group_split_never_crosses_group():
    groups = np.repeat([f"g{i}" for i in range(12)], 3)
    split = group_train_val_test_split(groups, seed=8)
    assert_group_disjoint(groups, split)
    with pytest.raises(SplitError):
        assert_group_disjoint(groups, {"train": np.array([0]), "test": np.array([1])})
