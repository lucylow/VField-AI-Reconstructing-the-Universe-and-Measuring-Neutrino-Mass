import numpy as np
import pytest

from nufield.models.hybrid_encoder import pack_observations


def test_pack_has_explicit_missing_lensing_mask():
    fields = np.ones((5, 8, 8, 8), dtype=np.float32)
    pk = np.ones((5, 7), dtype=np.float32)
    packed = pack_observations(fields, pk, lensing=None, lensing_shape=(8, 8))
    assert packed.values.shape == (5, 8**3 + 7 + 8**2 + 3)
    assert np.all(packed.values[:, -1] == 0.0)
    assert packed.field_shape == (1, 8, 8, 8)
    assert packed.lensing_shape == (1, 8, 8)


def test_pack_rejects_misaligned_rows():
    with pytest.raises(ValueError):
        pack_observations(np.ones((5, 8, 8, 8)), np.ones((4, 7)), lensing_shape=(8, 8))


def test_hybrid_encoder_forward_if_torch_installed():
    torch = pytest.importorskip("torch")
    from nufield.models.hybrid_encoder import HybridObservationEncoder

    fields = np.random.default_rng(0).normal(size=(4, 8, 8, 8)).astype(np.float32)
    pk = np.random.default_rng(1).normal(size=(4, 6)).astype(np.float32)
    batch = pack_observations(fields, pk, lensing=None, lensing_shape=(8, 8))
    encoder = HybridObservationEncoder(
        batch.field_shape,
        batch.spectrum_dim,
        batch.lensing_shape,
        output_dim=32,
        branch_dim=8,
        width=8,
    )
    result = encoder(torch.from_numpy(batch.values))
    assert result.shape == (4, 32)
    assert torch.isfinite(result).all()
