from nufield.models.model_cards import get_model_card, model_registry
from nufield.training.model_grid import build_model_matrix


def test_model_cards_are_explicit_about_assumptions():
    fno = get_model_card("fno3d_npe")
    assert any("periodic" in x.lower() for x in fno.data_minimums)
    assert any("masked" in x.lower() for x in fno.avoid_when)
    assert len(model_registry()) >= 5


def test_model_grid_only_emits_coherent_model_view_pairs():
    configs = build_model_matrix(
        dataset_id="q",
        seeds=[1, 2],
        split_id="fixed-v1",
        model_ids=("pk_classical", "mlp_npe", "cnn3d_npe"),
        observation_views=("pk", "field"),
    )
    assert configs
    assert len({c.experiment_id for c in configs}) == len(configs)
    assert all(c.split_id == "fixed-v1" for c in configs)
    assert all(not (c.model_id == "cnn3d_npe" and c.observation_view == "pk") for c in configs)


def test_embedding_factory_shapes_for_spectrum_cnn_and_fno():
    import pytest

    torch = pytest.importorskip("torch")
    from nufield.models.observation_adapters import build_embedding

    # Summary-only encoder.
    mlp, spec = build_embedding(
        "spectrum_mlp",
        field_shape=(1, 8, 8, 8),
        spectrum_dim=6,
        lensing_shape=(1, 8, 8),
        input_width=6,
        output_dim=10,
    )
    assert mlp(torch.randn(3, 6)).shape == (3, 10)
    # Flat input is explicitly restored to its 3-D field shape.
    cnn, _ = build_embedding(
        "cnn3d",
        field_shape=(1, 8, 8, 8),
        spectrum_dim=6,
        lensing_shape=(1, 8, 8),
        input_width=8**3,
        output_dim=10,
        width=4,
    )
    assert cnn(torch.randn(2, 8**3)).shape == (2, 10)
    fno, fno_spec = build_embedding(
        "fno3d",
        field_shape=(1, 8, 8, 8),
        spectrum_dim=6,
        lensing_shape=(1, 8, 8),
        input_width=8**3,
        output_dim=10,
        width=4,
        modes=(2, 2, 2),
        n_blocks=1,
    )
    assert fno(torch.randn(2, 8**3)).shape == (2, 10)
    assert fno_spec.boundary_condition == "periodic simulation box only"
