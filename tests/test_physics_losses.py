import pytest

torch = pytest.importorskip("torch")
from nufield.models.physics_losses import (
    composite_reconstruction_loss,
    gradient_matching_loss,
    isotropic_power_spectrum,
    log_power_spectrum_loss,
    mass_conservation_loss,
)


def test_isotropic_power_spectrum_and_identical_loss():
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    k, pk, occupied = isotropic_power_spectrum(x, box_size_mpc_h=500.0, n_bins=8)
    assert k.shape == (8,)
    assert pk.shape == (2, 8)
    assert occupied.shape == (8,)
    loss = log_power_spectrum_loss(x, x.detach(), box_size_mpc_h=500.0, n_bins=8)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_composite_loss_components_and_gradients():
    truth = torch.randn(2, 1, 8, 8, 8)
    pred = (truth + 0.03 * torch.randn_like(truth)).requires_grad_()
    report = composite_reconstruction_loss(pred, truth, box_size_mpc_h=250, n_bins=7)
    assert set(report) == {
        "total",
        "field_mse",
        "log_pk_huber",
        "gradient_mse",
        "mean_conservation",
        "weights",
    }
    report["total"].backward()
    assert torch.isfinite(pred.grad).all()
    assert gradient_matching_loss(truth, truth).item() == pytest.approx(0.0)
    assert mass_conservation_loss(truth, truth).item() == pytest.approx(0.0)
