"""Variational 3-D field autoencoder for compact representations and reconstruction baselines."""

from __future__ import annotations

from typing import Any

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch, nn, F = None, None, None


def _need_torch():
    if torch is None:
        raise OptionalDependencyError(
            "Install nufield-starter[ml] to use the variational autoencoder"
        )


if nn is not None:

    class VariationalFieldAutoencoder3D(nn.Module):
        """Convolutional VAE for regular-grid 3-D fields.

        It compresses the supplied field distribution; the latent variables should not be interpreted
        as physical cosmological parameters without a separate calibrated inference model.
        """

        def __init__(
            self,
            input_shape: tuple[int, int, int] = (32, 32, 32),
            in_channels: int = 1,
            base_channels: int = 16,
            latent_dim: int = 64,
        ):
            super().__init__()
            if (
                len(input_shape) != 3
                or min(input_shape) < 8
                or min(in_channels, base_channels, latent_dim) < 1
            ):
                raise ValueError(
                    "input_shape must be 3-D with dimensions >=8 and channels positive"
                )
            self.input_shape = tuple(map(int, input_shape))
            self.in_channels, self.latent_dim = int(in_channels), int(latent_dim)
            c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
            self.encoder_conv = nn.Sequential(
                nn.Conv3d(in_channels, c1, 4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv3d(c1, c2, 4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv3d(c2, c3, 4, stride=2, padding=1),
                nn.GELU(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, *self.input_shape)
                feature_shape = tuple(self.encoder_conv(dummy).shape[1:])
            self._encoded_shape = feature_shape
            flattened = int(torch.tensor(feature_shape).prod().item())
            self.fc_mu = nn.Linear(flattened, latent_dim)
            self.fc_logvar = nn.Linear(flattened, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, flattened)
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose3d(c1, in_channels, 4, stride=2, padding=1),
            )

        def encode(self, x):
            if (
                x.ndim != 5
                or x.shape[1] != self.in_channels
                or tuple(x.shape[-3:]) != self.input_shape
            ):
                raise ValueError(f"expected [B,{self.in_channels},{self.input_shape}]")
            h = self.encoder_conv(x).flatten(1)
            return self.fc_mu(h), self.fc_logvar(h).clamp(-20, 10)

        @staticmethod
        def reparameterize(mu, logvar, *, generator=None):
            if not torch.is_grad_enabled():
                return mu
            eps = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
            return mu + torch.exp(0.5 * logvar) * eps

        def decode(self, z):
            if z.ndim != 2 or z.shape[-1] != self.latent_dim:
                raise ValueError(f"latent z must be [B,{self.latent_dim}]")
            h = self.fc_decode(z).reshape(z.shape[0], *self._encoded_shape)
            output = self.decoder_conv(h)
            if tuple(output.shape[-3:]) != self.input_shape:
                output = F.interpolate(
                    output, size=self.input_shape, mode="trilinear", align_corners=False
                )
            return output

        def forward(self, x, *, sample: bool | None = None):
            mu, logvar = self.encode(x)
            should_sample = self.training if sample is None else bool(sample)
            z = self.reparameterize(mu, logvar) if should_sample else mu
            reconstruction = self.decode(z)
            return {"reconstruction": reconstruction, "mu": mu, "log_variance": logvar, "latent": z}

else:

    class VariationalFieldAutoencoder3D:
        def __init__(self, *args, **kwargs):
            _need_torch()


def vae_loss(
    output: dict[str, Any],
    target,
    *,
    beta: float = 1.0,
    reconstruction: str = "mse",
    reduction: str = "mean",
):
    """VAE reconstruction + beta-weighted KL divergence with returned term diagnostics."""
    _need_torch()
    if beta < 0 or reconstruction not in {"mse", "smooth_l1"}:
        raise ValueError("beta must be nonnegative and reconstruction mse/smooth_l1")
    pred = output["reconstruction"]
    if pred.shape != target.shape:
        raise ValueError("reconstruction and target shapes must match")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be mean or sum")
    if reconstruction == "mse":
        raw = F.mse_loss(pred, target, reduction="none")
    else:
        raw = F.smooth_l1_loss(pred, target, reduction="none")
    per_sample = raw.flatten(1).mean(dim=1)
    recon_loss = per_sample.mean() if reduction == "mean" else per_sample.sum()
    mu, logvar = output["mu"], output["log_variance"]
    kl_per_dim = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl_per_sample = kl_per_dim.sum(dim=1)
    kl = kl_per_sample.mean() if reduction == "mean" else kl_per_sample.sum()
    total = recon_loss + beta * kl
    return total, {
        "reconstruction_loss": recon_loss.detach(),
        "kl_divergence": kl.detach(),
        "beta": float(beta),
        "total_loss": total.detach(),
    }


def beta_warmup(
    epoch: int, warmup_epochs: int, *, maximum_beta: float = 1.0, schedule: str = "linear"
) -> float:
    """Return a reproducible KL warm-up coefficient to reduce early latent collapse."""
    import math

    if epoch < 0 or warmup_epochs < 0 or maximum_beta < 0:
        raise ValueError("epoch, warmup_epochs and maximum_beta must be nonnegative")
    progress = 1.0 if warmup_epochs == 0 else min(1.0, epoch / warmup_epochs)
    if schedule == "linear":
        scale = progress
    elif schedule == "cosine":
        scale = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif schedule == "constant":
        scale = 1.0
    else:
        raise ValueError("schedule must be linear/cosine/constant")
    return float(maximum_beta * scale)
