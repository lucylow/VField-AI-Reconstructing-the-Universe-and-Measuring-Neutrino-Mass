"""Optional modality encoders for a future joint inference model.

The encoders are generic ML components and do not implement cosmological observation models.
"""

from __future__ import annotations

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
except ImportError:
    torch, nn = None, None

if nn is not None:

    class MapEncoder2D(nn.Module):
        """Small convolutional encoder for already-projected 2-D maps."""

        def __init__(self, in_channels: int = 1, output_dim: int = 64):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1),
                nn.GELU(),
                nn.AvgPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.GELU(),
                nn.AvgPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(64, output_dim)

        def forward(self, x):
            return self.head(self.features(x).flatten(1))

    class FieldEncoder3D(nn.Module):
        """Compact 3-D convolutional encoder for cube fields."""

        def __init__(self, in_channels: int = 1, output_dim: int = 64):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(in_channels, 16, 3, padding=1),
                nn.GELU(),
                nn.AvgPool3d(2, ceil_mode=True),
                nn.Conv3d(16, 32, 3, padding=1),
                nn.GELU(),
                nn.AvgPool3d(2, ceil_mode=True),
                nn.Conv3d(32, 64, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
            self.head = nn.Linear(64, output_dim)

        def forward(self, x):
            return self.head(self.features(x).flatten(1))

    class MaskAwareFusionMLP(nn.Module):
        """Fuse fixed-width embeddings plus one availability bit per modality."""

        def __init__(self, embedding_dims: list[int], output_dim: int = 128):
            super().__init__()
            if not embedding_dims or any(d < 1 for d in embedding_dims):
                raise ValueError("embedding_dims must be positive")
            input_dim = sum(embedding_dims) + len(embedding_dims)
            self.network = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.GELU(),
            )
            self.embedding_dims = embedding_dims

        def forward(self, embeddings: list[torch.Tensor], availability: torch.Tensor):
            if len(embeddings) != len(self.embedding_dims):
                raise ValueError("embedding count mismatch")
            batch = embeddings[0].shape[0]
            for tensor, width in zip(embeddings, self.embedding_dims):
                if tensor.shape != (batch, width):
                    raise ValueError("embedding shape mismatch")
            if availability.shape != (batch, len(embeddings)):
                raise ValueError("availability must have shape (batch,n_modalities)")
            clean = [e * availability[:, i : i + 1] for i, e in enumerate(embeddings)]
            return self.network(torch.cat(clean + [availability.float()], dim=-1))
else:

    class MapEncoder2D:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install nufield-starter[ml]")

    class FieldEncoder3D:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install nufield-starter[ml]")

    class MaskAwareFusionMLP:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install nufield-starter[ml]")
