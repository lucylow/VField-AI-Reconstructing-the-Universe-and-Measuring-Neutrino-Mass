"""Simple baseline embedders used to ablate the more expensive hybrid CNN.

A necessary comparison is an MLP embedding of power-spectrum features; full-field architectures
should only be kept if they improve held-out inference after cost and calibration are considered.
"""

from __future__ import annotations

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
except ImportError:
    torch, nn = None, None

if nn is not None:

    class SpectrumEmbeddingMLP(nn.Module):
        def __init__(
            self, input_dim: int, output_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.05
        ):
            super().__init__()
            if min(input_dim, output_dim, hidden_dim) < 1:
                raise ValueError("dimensions must be positive")
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )

        def forward(self, x):
            if x.ndim != 2:
                raise ValueError("expected [B,K]")
            return self.net(x)

    class FlattenedFieldMLP(nn.Module):
        """Low-resolution field baseline. Do not use on large cubes due to huge parameter counts."""

        def __init__(
            self, input_shape: tuple[int, ...], output_dim: int = 64, hidden_dim: int = 256
        ):
            super().__init__()
            n = 1
            for d in input_shape:
                n *= d
            if n > 32768:
                raise ValueError(
                    "FlattenedFieldMLP is for low-resolution baselines only; use a CNN for this input"
                )
            self.input_shape = tuple(input_shape)
            self.net = nn.Sequential(
                nn.Linear(n, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            if x.ndim < 2 or tuple(x.shape[1:]) != self.input_shape:
                raise ValueError("field shape mismatch")
            return self.net(x.flatten(start_dim=1))
else:

    class SpectrumEmbeddingMLP:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install nufield-starter[ml]")

    class FlattenedFieldMLP:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install nufield-starter[ml]")
