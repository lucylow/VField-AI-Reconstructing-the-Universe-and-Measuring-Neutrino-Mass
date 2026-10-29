import numpy as np
import pytest

from nufield.ai.augmentation import (
    apply_cube_symmetry,
    augment_periodic_batch,
    cube_symmetry_transforms,
    periodic_roll,
)


def test_cube_symmetries_preserve_shape_and_voxel_values():
    x = np.arange(1 * 1 * 6 * 6 * 6).reshape(1, 1, 6, 6, 6)
    assert len(cube_symmetry_transforms()) == 48
    for i in range(48):
        y = apply_cube_symmetry(x, i)
        assert y.shape == x.shape
        assert np.array_equal(np.sort(y.ravel()), np.sort(x.ravel()))


def test_periodic_roll_and_batch_aug_reproducibility():
    rng = np.random.default_rng(20)
    x = rng.normal(size=(4, 1, 8, 8, 8))
    rolled, shifts = periodic_roll(x[0], rng=rng, shifts=(1, 2, 3))
    assert shifts == (1, 2, 3) and np.allclose(rolled.mean(), x[0].mean())
    a = augment_periodic_batch(x, rng=np.random.default_rng(44))
    b = augment_periodic_batch(x, rng=np.random.default_rng(44))
    assert np.array_equal(a, b)


def test_noise_augmentation_validation():
    from nufield.ai.augmentation import additive_noise

    x = np.ones((8, 8, 8))
    y = additive_noise(x, rng=np.random.default_rng(2), standard_deviation=0.1)
    assert abs(y.mean() - x.mean()) < 1e-8
    with pytest.raises(ValueError):
        additive_noise(x, rng=np.random.default_rng(2), standard_deviation=-1)


def test_torch_gradient_attributions_and_physics_losses():
    torch = pytest.importorskip("torch")
    from torch import nn

    from nufield.ai.explainability import input_gradient_saliency, integrated_gradients
    from nufield.ai.physics_guided import (
        box_constraint_penalty,
        equivariance_consistency_loss,
        smoothness_penalty,
    )

    model = nn.Sequential(nn.Flatten(), nn.Linear(8 * 8 * 8, 2))
    x = torch.randn(3, 1, 8, 8, 8)
    grad = input_gradient_saliency(model, x)
    ig = integrated_gradients(model, x, steps=8, internal_batch_size=2)
    assert grad.shape == ig.shape == x.shape
    assert torch.isfinite(grad).all() and torch.isfinite(ig).all()
    penalty = box_constraint_penalty(torch.tensor([[-2.0, 0.5]]), [0, 0], [1, 1])
    assert penalty.item() > 0
    assert smoothness_penalty(x, periodic=True).item() >= 0
    scalar_model = nn.Sequential(nn.Flatten(), nn.Linear(8 * 8 * 8, 1))
    assert equivariance_consistency_loss(scalar_model, x, transform="roll").item() >= 0
