"""Compact 3-D Fourier Neural Operator encoder for periodic simulation boxes.

The network is optional (PyTorch). It assumes regularly sampled, approximately periodic volumes.
Do not apply it unmodified to masked survey cutouts: encode the survey window explicitly or use a
validated observation model. It is a candidate embedding architecture, not a trained posterior.
"""

from __future__ import annotations

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    torch, nn, F = None, None, None


if nn is not None:

    class SpectralConv3d(nn.Module):
        """Learn low-frequency Fourier mixing with four signed spatial-frequency corners."""

        def __init__(
            self, in_channels: int, out_channels: int, modes: tuple[int, int, int] = (8, 8, 8)
        ):
            super().__init__()
            if min(in_channels, out_channels, *modes) < 1:
                raise ValueError("channels and modes must be positive")
            self.in_channels, self.out_channels = int(in_channels), int(out_channels)
            self.modes = tuple(int(x) for x in modes)
            scale = 1.0 / max(1, in_channels * out_channels)
            shape = (in_channels, out_channels, *self.modes)
            self.w1 = nn.Parameter(scale * torch.randn(shape, dtype=torch.cfloat))
            self.w2 = nn.Parameter(scale * torch.randn(shape, dtype=torch.cfloat))
            self.w3 = nn.Parameter(scale * torch.randn(shape, dtype=torch.cfloat))
            self.w4 = nn.Parameter(scale * torch.randn(shape, dtype=torch.cfloat))

        @staticmethod
        def _mix(inp, weight):
            return torch.einsum("bixyz,ioxyz->boxyz", inp, weight)

        def forward(self, x):
            if x.ndim != 5:
                raise ValueError("SpectralConv3d expects [B,C,D,H,W]")
            if x.shape[1] != self.in_channels:
                raise ValueError("input channel count mismatch")
            b, _, d, h, w = x.shape
            mx, my, mz = (
                min(self.modes[0], d // 2),
                min(self.modes[1], h // 2),
                min(self.modes[2], w // 2 + 1),
            )
            if min(mx, my, mz) < 1:
                raise ValueError("input grid is too small for spectral modes")
            xf = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
            out = torch.zeros(
                (b, self.out_channels, d, h, w // 2 + 1), dtype=xf.dtype, device=x.device
            )
            out[:, :, :mx, :my, :mz] = self._mix(
                xf[:, :, :mx, :my, :mz], self.w1[:, :, :mx, :my, :mz]
            )
            out[:, :, -mx:, :my, :mz] = self._mix(
                xf[:, :, -mx:, :my, :mz], self.w2[:, :, :mx, :my, :mz]
            )
            out[:, :, :mx, -my:, :mz] = self._mix(
                xf[:, :, :mx, -my:, :mz], self.w3[:, :, :mx, :my, :mz]
            )
            out[:, :, -mx:, -my:, :mz] = self._mix(
                xf[:, :, -mx:, -my:, :mz], self.w4[:, :, :mx, :my, :mz]
            )
            return torch.fft.irfftn(out, s=(d, h, w), dim=(-3, -2, -1), norm="ortho")

    class FourierResidualBlock(nn.Module):
        def __init__(self, width: int, modes: tuple[int, int, int], dropout: float = 0.0):
            super().__init__()
            if not 0 <= dropout < 1:
                raise ValueError("dropout must be in [0,1)")
            self.spectral = SpectralConv3d(width, width, modes)
            self.local = nn.Conv3d(width, width, kernel_size=1)
            self.norm = nn.GroupNorm(_groups(width), width)
            self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()

        def forward(self, x):
            return F.gelu(self.dropout(self.norm(self.spectral(x) + self.local(x))))

    def _groups(channels: int) -> int:
        n = min(8, channels)
        while channels % n:
            n -= 1
        return n

    class FNO3DEmbedding(nn.Module):
        """Map a periodic 3-D field to a fixed-size embedding for SBI or downstream tasks."""

        def __init__(
            self,
            in_channels: int = 1,
            width: int = 16,
            output_dim: int = 128,
            modes: tuple[int, int, int] = (8, 8, 8),
            n_blocks: int = 4,
            dropout: float = 0.0,
        ):
            super().__init__()
            if min(in_channels, width, output_dim, n_blocks) < 1:
                raise ValueError("model dimensions must be positive")
            if len(modes) != 3 or min(modes) < 1:
                raise ValueError("modes must contain three positive integers")
            self.in_channels, self.width, self.output_dim = in_channels, width, output_dim
            self.modes, self.n_blocks = tuple(modes), n_blocks
            self.lift = nn.Conv3d(in_channels, width, kernel_size=1)
            self.blocks = nn.ModuleList(
                [FourierResidualBlock(width, tuple(modes), dropout) for _ in range(n_blocks)]
            )
            self.project = nn.Sequential(
                nn.Linear(width * 2, width * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, output_dim),
                nn.LayerNorm(output_dim),
            )

        def forward(self, x):
            if x.ndim != 5:
                raise ValueError("FNO3DEmbedding expects [batch,channels,D,H,W]")
            if x.shape[1] != self.in_channels:
                raise ValueError("input channel count mismatch")
            z = self.lift(x)
            for block in self.blocks:
                z = z + block(z) * 0.5
            pooled = torch.cat([z.mean(dim=(-3, -2, -1)), z.amax(dim=(-3, -2, -1))], dim=1)
            return self.project(pooled)
else:

    class SpectralConv3d:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install νField with the [ml] extra to use FNO")

    class FourierResidualBlock:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install νField with the [ml] extra to use FNO")

    class FNO3DEmbedding:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install νField with the [ml] extra to use FNO")
