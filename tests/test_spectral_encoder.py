import pytest

torch = pytest.importorskip("torch")
from nufield.models.spectral_encoder import FNO3DEmbedding, SpectralConv3d


def test_spectral_conv_shapes_and_gradient():
    layer = SpectralConv3d(2, 3, modes=(3, 3, 3))
    x = torch.randn(2, 2, 10, 12, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 3, 10, 12, 8)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_fno_embedding_shape_and_gradient():
    model = FNO3DEmbedding(in_channels=1, width=6, output_dim=11, modes=(3, 3, 3), n_blocks=2)
    x = torch.randn(2, 1, 12, 12, 12, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 11)
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_fno_rejects_non_5d():
    model = FNO3DEmbedding(width=4, output_dim=8, modes=(2, 2, 2), n_blocks=1)
    with pytest.raises(ValueError):
        model(torch.randn(3, 12, 12, 12))
