import numpy as np
import pytest


def test_variational_autoencoder_and_beta_warmup():
    torch = pytest.importorskip("torch")
    from nufield.ai.autoencoder import VariationalFieldAutoencoder3D, beta_warmup, vae_loss

    model = VariationalFieldAutoencoder3D((8, 8, 8), in_channels=1, base_channels=4, latent_dim=5)
    x = torch.randn(2, 1, 8, 8, 8)
    out = model(x)
    assert out["reconstruction"].shape == x.shape
    loss, terms = vae_loss(out, x, beta=0.2)
    assert torch.isfinite(loss) and terms["kl_divergence"].item() >= 0
    assert beta_warmup(0, 10) == 0
    assert beta_warmup(10, 10) == 1
    with torch.no_grad():
        assert model(x, sample=False)["latent"].shape == (2, 5)


def test_model_ema_tracks_weights():
    torch = pytest.importorskip("torch")
    from torch import nn

    from nufield.ai.model_ema import ModelEMA

    model = nn.Linear(2, 1)
    ema = ModelEMA(model, decay=0.9)
    original = next(ema.shadow.parameters()).detach().clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    assert not torch.equal(next(ema.shadow.parameters()), original)
    copied = nn.Linear(2, 1)
    ema.copy_to(copied)
    assert torch.allclose(next(copied.parameters()), next(ema.shadow.parameters()))


def test_committee_metrics_capture_diversity():
    from nufield.ai.committee import ensemble_diversity_report, pairwise_prediction_correlation

    rng = np.random.default_rng(70)
    target = rng.normal(size=(50, 2))
    pred = np.stack([target + rng.normal(0, 0.2, target.shape) for _ in range(4)])
    report = ensemble_diversity_report(pred, target)
    assert report["ensemble_mean_rmse"] < report["member_rmse_mean"]
    assert pairwise_prediction_correlation(pred)["correlation_matrix"]
