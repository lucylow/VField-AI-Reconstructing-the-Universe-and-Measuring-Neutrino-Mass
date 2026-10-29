import numpy as np
import pytest


def _torch():
    return pytest.importorskip("torch")


def test_residual_gaussian_and_mixture_networks():
    torch = _torch()
    from nufield.ai.networks import (
        HeteroscedasticRegressor,
        MixtureDensityRegressor,
        ResidualSummaryEncoder,
        count_trainable_parameters,
    )

    encoder = ResidualSummaryEncoder(10, embedding_dim=16, width=24, depth=2)
    gaussian = HeteroscedasticRegressor(encoder, 16, 3)
    out = gaussian(torch.randn(5, 10))
    assert out["mean"].shape == out["log_variance"].shape == (5, 3)
    mdn = MixtureDensityRegressor(ResidualSummaryEncoder(10, 16, 24, 2), 16, 3, n_components=4)
    params = mdn(torch.randn(5, 10))
    target = torch.randn(5, 3)
    logp = mdn.log_prob(params, target)
    samples = mdn.sample(params, 7)
    assert logp.shape == (5,) and samples.shape == (5, 7, 3)
    assert count_trainable_parameters(mdn) > 0


def test_set_encoder_is_permutation_invariant_and_masked():
    torch = _torch()
    from nufield.ai.networks import MaskedPointSetEncoder

    model = MaskedPointSetEncoder(point_dim=4, embedding_dim=12, hidden_dim=16).eval()
    x = torch.randn(2, 9, 4)
    mask = torch.tensor([[1] * 7 + [0] * 2, [1] * 9], dtype=torch.bool)
    perm = torch.randperm(9)
    with torch.no_grad():
        a = model(x, mask)
        b = model(x[:, perm], mask[:, perm])
    assert torch.allclose(a, b, atol=1e-5)
    with pytest.raises(ValueError, match="at least one"):
        model(x, torch.zeros_like(mask))


def test_masked_multimodal_transformer_and_network():
    torch = _torch()
    from nufield.ai.multimodal_network import MultimodalPosteriorNetwork
    from nufield.ai.networks import ModalityFusionTransformer

    fusion = ModalityFusionTransformer({"a": 8, "b": 8}, width=16, heads=4, layers=1)
    out = fusion(
        {"a": torch.randn(3, 8), "b": torch.randn(3, 8)},
        torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool),
    )
    assert out.shape == (3, 16) and torch.isfinite(out).all()
    model = MultimodalPosteriorNetwork(
        {"pk": {"kind": "summary", "features": 7}, "lens": {"kind": "map2d", "channels": 1}},
        target_dim=3,
        width=16,
        heads=4,
        fusion_layers=1,
        output_kind="gaussian",
    )
    inputs = {"pk": torch.randn(2, 7), "lens": torch.randn(2, 1, 16, 16)}
    pred = model(inputs, torch.tensor([[1, 1], [1, 0]], dtype=torch.bool))
    assert pred["mean"].shape == (2, 3)


def test_gaussian_losses_and_mdn_loss():
    torch = _torch()
    from nufield.ai.losses import gaussian_nll, mixture_nll
    from nufield.ai.networks import MixtureDensityRegressor, ResidualSummaryEncoder

    y = torch.zeros(4, 2)
    output = {"mean": torch.zeros_like(y), "log_variance": torch.zeros_like(y)}
    assert torch.isfinite(gaussian_nll(output, y))
    model = MixtureDensityRegressor(ResidualSummaryEncoder(5, 8, 16, 1), 8, 2, 3)
    params = model(torch.randn(4, 5))
    assert torch.isfinite(mixture_nll(model, params, y))


def test_group_ids_prevent_train_validation_leakage(tmp_path):
    torch = _torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from nufield.ai.training import TrainConfig, fit_model

    x, y = torch.randn(32, 4), torch.randn(32, 2)
    loader = DataLoader(TensorDataset(x[:24], y[:24]), batch_size=8)
    val = DataLoader(TensorDataset(x[24:], y[24:]), batch_size=4)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
    result = fit_model(
        model,
        loader,
        val,
        config=TrainConfig(max_epochs=3, patience=2, device="cpu"),
        loss_fn=lambda out, target: torch.nn.functional.mse_loss(out, target),
        train_group_ids=np.arange(24),
        validation_group_ids=np.arange(24, 32),
        test_group_ids=np.arange(32, 40),
    )
    assert result.best_epoch >= 1 and np.isfinite(result.best_val_loss)
    with pytest.raises(ValueError, match="overlap"):
        fit_model(
            model,
            loader,
            val,
            config=TrainConfig(max_epochs=1),
            loss_fn=lambda out, target: torch.nn.functional.mse_loss(out, target),
            train_group_ids=np.arange(24),
            validation_group_ids=np.arange(20, 28),
        )


def test_torch_contrastive_loss_and_gradient_reversal():
    torch = _torch()
    from nufield.ai.contrastive import multi_positive_info_nce
    from nufield.ai.domain_adaptation import gradient_reverse, reversal_strength_schedule

    a, b = torch.randn(5, 12, requires_grad=True), torch.randn(5, 12, requires_grad=True)
    loss = multi_positive_info_nce(a, b, [1, 1, 2, 3, 4], temperature=0.2)
    loss.backward()
    assert torch.isfinite(loss) and a.grad is not None
    x = torch.tensor([1.0, 3.0], requires_grad=True)
    (gradient_reverse(x, 0.5).sum()).backward()
    assert torch.allclose(x.grad, torch.tensor([-0.5, -0.5]))
    assert reversal_strength_schedule(0, 10) == pytest.approx(0)
    assert reversal_strength_schedule(9, 10) > 0.9
