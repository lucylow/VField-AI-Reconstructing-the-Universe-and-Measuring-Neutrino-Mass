"""Optional PyTorch 3-D U-Net scaffold.

This model is an engineering starting point only. Train it only when a validated simulator provides
physically meaningful paired inputs/targets and compare against classical reconstruction baselines.
"""

from __future__ import annotations

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
except ImportError:  # optional dependency kept genuinely optional
    torch = None
    nn = None

if nn is not None:

    class ConvBlock3D(nn.Module):
        """Two-convolution block with GroupNorm and SiLU."""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups:
                groups -= 1
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
            )

        def forward(self, x):
            return self.net(x)

    class UNet3D(nn.Module):
        """Compact 3-D U-Net for fixed-size cubic fields."""

        def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
            super().__init__()
            c = base_channels
            self.enc1 = ConvBlock3D(in_channels, c)
            self.enc2 = ConvBlock3D(c, 2 * c)
            self.enc3 = ConvBlock3D(2 * c, 4 * c)
            self.pool = nn.MaxPool3d(2)
            self.up2 = nn.ConvTranspose3d(4 * c, 2 * c, 2, stride=2)
            self.dec2 = ConvBlock3D(4 * c, 2 * c)
            self.up1 = nn.ConvTranspose3d(2 * c, c, 2, stride=2)
            self.dec1 = ConvBlock3D(2 * c, c)
            self.head = nn.Conv3d(c, out_channels, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.up2(e3)
            if d2.shape[-3:] != e2.shape[-3:]:
                d2 = torch.nn.functional.interpolate(
                    d2, size=e2.shape[-3:], mode="trilinear", align_corners=False
                )
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            if d1.shape[-3:] != e1.shape[-3:]:
                d1 = torch.nn.functional.interpolate(
                    d1, size=e1.shape[-3:], mode="trilinear", align_corners=False
                )
            return self.head(self.dec1(torch.cat([d1, e1], dim=1)))
else:

    class UNet3D:  # pragma: no cover - only used without torch installed
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install nufield-starter[ml] to use UNet3D")
