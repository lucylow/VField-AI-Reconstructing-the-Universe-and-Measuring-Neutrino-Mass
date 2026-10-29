"""Small heteroscedastic neural regression baseline for point/error diagnostics.

This is explicitly not a replacement for NPE: a diagonal Gaussian head cannot represent arbitrary
multimodal posterior shapes and its uncertainty may be miscalibrated. Use it as a cheap baseline.
"""

from __future__ import annotations

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
except ImportError:
    torch, nn = None, None

if nn is not None:

    class GaussianRegressor(nn.Module):
        """MLP predicting one mean and log-variance per target coordinate."""

        def __init__(
            self,
            input_dim: int,
            target_dim: int,
            hidden_dims: tuple[int, ...] = (256, 128),
            dropout: float = 0.05,
        ):
            super().__init__()
            if (
                input_dim < 1
                or target_dim < 1
                or not hidden_dims
                or any(v < 1 for v in hidden_dims)
            ):
                raise ValueError("all model dimensions must be positive")
            if not 0 <= dropout < 1:
                raise ValueError("dropout must be in [0,1)")
            layers = []
            current = input_dim
            for width in hidden_dims:
                layers.extend([nn.Linear(current, width), nn.LayerNorm(width), nn.GELU()])
                if dropout:
                    layers.append(nn.Dropout(dropout))
                current = width
            self.trunk = nn.Sequential(*layers)
            self.output = nn.Linear(current, 2 * target_dim)
            self.target_dim = target_dim

        def forward(self, x):
            if x.ndim != 2:
                raise ValueError("GaussianRegressor expects [B,input_dim]")
            raw = self.output(self.trunk(x))
            mean = raw[:, : self.target_dim]
            log_var = raw[:, self.target_dim :].clamp(min=-10.0, max=6.0)
            return mean, log_var

    def gaussian_nll(mean, log_var, target):
        if mean.shape != log_var.shape or mean.shape != target.shape:
            raise ValueError("mean/log_var/target shapes must match")
        return 0.5 * (log_var + (target - mean).square() * torch.exp(-log_var)).mean()
else:

    class GaussianRegressor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install nufield-starter[ml]")

    def gaussian_nll(*args, **kwargs):
        raise OptionalDependencyError("Install nufield-starter[ml]")
