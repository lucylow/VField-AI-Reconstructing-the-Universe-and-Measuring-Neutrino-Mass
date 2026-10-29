import pytest

from nufield.models.selector import recommend_models


def test_small_dataset_prefers_baseline_and_summary_model():
    rows = recommend_models(n_simulations=150, field_shape=(16, 16, 16), budget="laptop")
    assert rows[0].name.startswith("Power-spectrum")
    assert any("Spectrum MLP" in row.name for row in rows)
    assert all(row.required_ablation for row in rows)


def test_multimodal_recommendation_requires_task_and_lensing():
    rows = recommend_models(
        task="multimodal_posterior",
        n_simulations=5000,
        field_shape=(32, 32, 32),
        has_power_spectrum=True,
        has_lensing=True,
        budget="hpc",
    )
    assert any("Mask-aware multimodal" in row.name for row in rows)


def test_bad_field_shape_rejected():
    with pytest.raises(ValueError, match="field dimensions"):
        recommend_models(n_simulations=20, field_shape=(0, 8, 8))


def test_fno_is_only_recommended_for_declared_periodic_boxes():
    nonperiodic = recommend_models(
        n_simulations=4000, field_shape=(32, 32, 32), budget="workstation"
    )
    periodic = recommend_models(
        n_simulations=4000, field_shape=(32, 32, 32), budget="workstation", periodic_box=True
    )
    assert not any("Fourier Neural Operator" in row.name for row in nonperiodic)
    assert any("Fourier Neural Operator" in row.name for row in periodic)


def test_learned_model_score_accounts_for_simulations_per_parameter():
    many_parameters = recommend_models(
        n_simulations=1000, n_parameters=50, field_shape=(16, 16, 16), budget="workstation"
    )
    few_parameters = recommend_models(
        n_simulations=1000, n_parameters=5, field_shape=(16, 16, 16), budget="workstation"
    )
    low_data_model = next(row for row in many_parameters if "Spectrum MLP" in row.name)
    better_sampled_model = next(row for row in few_parameters if "Spectrum MLP" in row.name)

    assert low_data_model.fit_score < better_sampled_model.fit_score
    assert "simulations per inferred parameter" in low_data_model.data_regime_note
    assert low_data_model.to_dict()["data_regime_note"] == low_data_model.data_regime_note
