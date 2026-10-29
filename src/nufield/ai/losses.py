"""Probabilistic objectives and physics-aware regularizers for AI training."""

from __future__ import annotations

from typing import Any

import numpy as np

from nufield.exceptions import OptionalDependencyError

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to use differentiable AI losses")


def gaussian_nll(output: dict[str, Any], target, *, reduction: str = "mean"):
    """Diagonal Gaussian negative log likelihood with output ``mean`` and ``log_variance``."""
    _require_torch()
    mean, logvar = output["mean"], output["log_variance"]
    if mean.shape != target.shape or logvar.shape != target.shape:
        raise ValueError("mean, log_variance and target shapes must match")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none/mean/sum")
    terms = 0.5 * (logvar + (target - mean).square() * torch.exp(-logvar) + np.log(2 * np.pi))
    per_sample = terms.sum(dim=-1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


def mixture_nll(model, params: dict[str, Any], target, *, reduction: str = "mean"):
    """Negative log probability for a ``MixtureDensityRegressor`` output."""
    _require_torch()
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none/mean/sum")
    logp = model.log_prob(params, target)
    if reduction == "none":
        return -logp
    if reduction == "sum":
        return -logp.sum()
    return -logp.mean()


def mse_loss(prediction, target, *, reduction: str = "mean"):
    """Mean squared error with shape validation."""
    _require_torch()
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    return torch.nn.functional.mse_loss(prediction, target, reduction=reduction)


def spectral_log_power_loss(
    prediction, target, *, box_size_mpc_h: float, n_bins: int = 24, eps: float = 1e-8
):
    """Compare log binned power spectra of matched periodic 3-D field batches.

    This is a regularizer only; it is not a likelihood. The caller is responsible for matching
    the field definition, volume, mesh, survey window, and noise treatment. A spectrum-only loss can
    erase phase information, so combine it with a field-space or reconstruction objective.
    """
    _require_torch()
    from nufield.models.physics_losses import isotropic_power_spectrum

    kp, pp = isotropic_power_spectrum(prediction, box_size_mpc_h=box_size_mpc_h, n_bins=n_bins)
    kt, pt = isotropic_power_spectrum(target, box_size_mpc_h=box_size_mpc_h, n_bins=n_bins)
    if kp.shape != kt.shape or not torch.allclose(kp, kt):
        raise ValueError("prediction and target power grids are inconsistent")
    valid = torch.isfinite(pp) & torch.isfinite(pt) & (pp > 0) & (pt > 0)
    if not torch.any(valid):
        raise ValueError("no positive finite power bins are available")
    return (torch.log(pp[valid] + eps) - torch.log(pt[valid] + eps)).square().mean()


def total_reconstruction_loss(
    prediction,
    target,
    *,
    field_weight: float = 1.0,
    spectral_weight: float = 0.0,
    box_size_mpc_h: float | None = None,
    n_bins: int = 24,
):
    """Field MSE plus optional spectral-shape regularizer, returned with component diagnostics."""
    _require_torch()
    if field_weight < 0 or spectral_weight < 0 or field_weight + spectral_weight <= 0:
        raise ValueError("loss weights must be nonnegative and not both zero")
    field = torch.nn.functional.mse_loss(prediction, target)
    spectral = torch.zeros((), dtype=field.dtype, device=field.device)
    if spectral_weight:
        if box_size_mpc_h is None:
            raise ValueError("box_size_mpc_h is required when spectral_weight is positive")
        spectral = spectral_log_power_loss(
            prediction, target, box_size_mpc_h=box_size_mpc_h, n_bins=n_bins
        )
    return field_weight * field + spectral_weight * spectral, {
        "field_mse": field.detach(),
        "spectral_log_power": spectral.detach(),
    }


def gaussian_nll_numpy(mean: np.ndarray, variance: np.ndarray, target: np.ndarray) -> np.ndarray:
    """NumPy Gaussian NLL per sample, useful for evaluating archived predictions."""
    m, v, y = np.asarray(mean, float), np.asarray(variance, float), np.asarray(target, float)
    if m.shape != v.shape or m.shape != y.shape or m.ndim < 1:
        raise ValueError("mean, variance and target must have identical shapes")
    if (
        not np.isfinite(m).all()
        or not np.isfinite(v).all()
        or not np.isfinite(y).all()
        or np.any(v <= 0)
    ):
        raise ValueError("inputs must be finite and variances positive")
    terms = 0.5 * (np.log(2 * np.pi * v) + np.square(y - m) / v)
    return terms.reshape(len(terms), -1).sum(axis=1)
