"""Hybrid field + spectrum + optional map encoder for SBI neural posterior estimation.

A single packed vector is used because SBI's embedding-network interface accepts an observation
batch tensor. `pack_observations` stores flattened modality tensors plus availability bits; this is
an implementation transport format, not a scientifically meaningful flattened representation.
The CNN branches reshape the values back into their spatial grids before feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

from nufield.exceptions import OptionalDependencyError


@dataclass(frozen=True)
class PackedObservations:
    values: np.ndarray
    field_shape: tuple[int, ...]
    spectrum_dim: int
    lensing_shape: tuple[int, ...]
    modality_order: tuple[str, ...] = ("field", "spectrum", "lensing")

    def metadata(self) -> dict:
        return {
            "field_shape": list(self.field_shape),
            "spectrum_dim": self.spectrum_dim,
            "lensing_shape": list(self.lensing_shape),
            "modality_order": list(self.modality_order),
            "packed_width": int(self.values.shape[1]),
            "warning": "packed vector is a transport encoding; model splits and encodes modalities separately",
        }


def _as_field_batch(fields: np.ndarray) -> np.ndarray:
    x = np.asarray(fields, dtype=np.float32)
    if x.ndim == 4:
        x = x[:, None, ...]
    if x.ndim != 5:
        raise ValueError("fields must have shape [N,D,H,W] or [N,C,D,H,W]")
    if any(d < 4 for d in x.shape[2:]):
        raise ValueError("each field dimension must be at least 4 voxels")
    if not np.isfinite(x).all():
        raise ValueError("field batch must be finite")
    return x


def _as_spectrum_batch(spectrum: np.ndarray) -> np.ndarray:
    x = np.asarray(spectrum, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("spectrum must have shape [N,K]")
    if not np.isfinite(x).all():
        raise ValueError("spectrum batch must be finite")
    return x


def _as_lensing_batch(
    lensing: np.ndarray | None, n: int, shape: tuple[int, int] | None
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    if lensing is None:
        if shape is None or len(shape) != 2 or min(shape) < 4:
            raise ValueError("lensing_shape=(H,W) with H,W>=4 is required when lensing is absent")
        array = np.zeros((n, 1, *shape), dtype=np.float32)
        available = np.zeros((n, 1), dtype=np.float32)
        return array, tuple(array.shape[1:]), available
    x = np.asarray(lensing, dtype=np.float32)
    if x.ndim == 3:
        x = x[:, None, ...]
    if x.ndim != 4:
        raise ValueError("lensing maps must have shape [N,H,W] or [N,C,H,W]")
    if len(x) != n or min(x.shape[-2:]) < 4:
        raise ValueError("lensing batch mismatch or grid too small")
    if not np.isfinite(x).all():
        raise ValueError("lensing batch must be finite")
    return x, tuple(x.shape[1:]), np.ones((n, 1), dtype=np.float32)


def pack_observations(
    fields: np.ndarray,
    power_spectrum: np.ndarray,
    lensing: np.ndarray | None = None,
    *,
    lensing_shape: tuple[int, int] | None = None,
    field_available: np.ndarray | None = None,
    spectrum_available: np.ndarray | None = None,
) -> PackedObservations:
    """Pack aligned modalities into a tensor suitable for an SBI embedding network.

    Missing lensing is represented by zeros plus an availability bit, never by an unmarked zero map.
    For field/spectrum availability masks, masks are per row and default to available. This function
    does not split data or normalize features; call it after creating leakage-safe train/val/test
    assignments and fitting preprocessing on training only.
    """
    f = _as_field_batch(fields)
    p = _as_spectrum_batch(power_spectrum)
    if len(f) != len(p):
        raise ValueError("field and spectrum row counts differ")
    n = len(f)
    l, lens_shape, lens_mask = _as_lensing_batch(lensing, n, lensing_shape)
    if field_available is None:
        fm = np.ones((n, 1), dtype=np.float32)
    else:
        fm = np.asarray(field_available, dtype=np.float32).reshape(-1, 1)
        if len(fm) != n or not np.isin(fm, [0, 1]).all():
            raise ValueError("field_available must be 0/1 per row")
    if spectrum_available is None:
        pm = np.ones((n, 1), dtype=np.float32)
    else:
        pm = np.asarray(spectrum_available, dtype=np.float32).reshape(-1, 1)
        if len(pm) != n or not np.isin(pm, [0, 1]).all():
            raise ValueError("spectrum_available must be 0/1 per row")
    flat = np.concatenate([f.reshape(n, -1), p, l.reshape(n, -1), fm, pm, lens_mask], axis=1)
    if not np.isfinite(flat).all():
        raise ValueError("packed observation tensor must be finite")
    return PackedObservations(flat.astype(np.float32), tuple(f.shape[1:]), p.shape[1], lens_shape)


try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # optional dependency
    torch, nn, F = None, None, None

if nn is not None:

    def _valid_groups(channels: int, maximum: int = 8) -> int:
        groups = min(maximum, channels)
        while channels % groups:
            groups -= 1
        return groups

    class ResidualBlock3D(nn.Module):
        """Residual convolution block with a projected shortcut when channels differ."""

        def __init__(
            self, in_channels: int, out_channels: int, *, stride: int = 1, dropout: float = 0.0
        ):
            super().__init__()
            if in_channels < 1 or out_channels < 1:
                raise ValueError("channel counts must be positive")
            if not 0 <= dropout < 1:
                raise ValueError("dropout must be in [0,1)")
            groups = min(8, out_channels)
            while out_channels % groups:
                groups -= 1
            self.body = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(),
                nn.Dropout3d(dropout) if dropout else nn.Identity(),
                nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels and stride == 1
                else nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False)
            )

        def forward(self, x):
            return F.silu(self.body(x) + self.skip(x))

    class FieldCNN3D(nn.Module):
        """Compact residual 3-D CNN for cubic/rectangular simulation fields."""

        def __init__(
            self, in_channels: int, output_dim: int, width: int = 16, dropout: float = 0.0
        ):
            super().__init__()
            if min(in_channels, output_dim, width) < 1:
                raise ValueError("network dimensions must be positive")
            self.net = nn.Sequential(
                nn.Conv3d(in_channels, width, 3, padding=1, bias=False),
                nn.GroupNorm(_valid_groups(width), width),
                nn.SiLU(),
                ResidualBlock3D(width, width, dropout=dropout),
                ResidualBlock3D(width, 2 * width, stride=2, dropout=dropout),
                ResidualBlock3D(2 * width, 2 * width, dropout=dropout),
                ResidualBlock3D(2 * width, 4 * width, stride=2, dropout=dropout),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(4 * width, output_dim),
                nn.LayerNorm(output_dim),
                nn.SiLU(),
            )

        def forward(self, x):
            if x.ndim != 5:
                raise ValueError("field encoder expects [batch,channels,D,H,W]")
            return self.net(x)

    class MapCNN2D(nn.Module):
        """Residual-style 2-D CNN for projected maps such as convergence fields."""

        def __init__(
            self, in_channels: int, output_dim: int, width: int = 16, dropout: float = 0.0
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
                nn.GroupNorm(_valid_groups(width), width),
                nn.GELU(),
                nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout) if dropout else nn.Identity(),
                nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(4 * width, output_dim),
                nn.GELU(),
            )

        def forward(self, x):
            if x.ndim != 4:
                raise ValueError("map encoder expects [batch,channels,H,W]")
            return self.net(x)

    class SpectrumMLP(nn.Module):
        """MLP branch for fixed-width power-spectrum / correlation-function summaries."""

        def __init__(
            self, input_dim: int, output_dim: int, hidden_dim: int = 128, dropout: float = 0.05
        ):
            super().__init__()
            if min(input_dim, output_dim, hidden_dim) < 1:
                raise ValueError("spectrum dimensions must be positive")
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
                raise ValueError("spectrum encoder expects [batch,K]")
            return self.net(x)

    class HybridObservationEncoder(nn.Module):
        """SBI-compatible embedding network for a flattened field/spectrum/lensing packet.

        `input_shape` is fixed at construction. Fields have shape (C,D,H,W), lensing maps
        (C,H,W), and a final 3-bit availability mask follows the modality arrays. The model learns
        an embedding, not a posterior directly; SBI's density estimator models p(theta | embedding).
        """

        def __init__(
            self,
            field_shape: tuple[int, int, int, int],
            spectrum_dim: int,
            lensing_shape: tuple[int, int, int],
            output_dim: int = 128,
            branch_dim: int = 48,
            width: int = 12,
            dropout: float = 0.05,
        ):
            super().__init__()
            if len(field_shape) != 4 or len(lensing_shape) != 3:
                raise ValueError("field_shape=(C,D,H,W), lensing_shape=(C,H,W)")
            if min(field_shape) < 1 or min(lensing_shape) < 1 or spectrum_dim < 1:
                raise ValueError("input dimensions must be positive")
            self.field_shape = tuple(field_shape)
            self.spectrum_dim = int(spectrum_dim)
            self.lensing_shape = tuple(lensing_shape)
            self.field_size = prod(field_shape)
            self.spectrum_size = int(spectrum_dim)
            self.lensing_size = prod(lensing_shape)
            self.expected_input_dim = self.field_size + self.spectrum_size + self.lensing_size + 3
            self.field_encoder = FieldCNN3D(
                field_shape[0], branch_dim, width=width, dropout=dropout
            )
            self.spectrum_encoder = SpectrumMLP(
                spectrum_dim, branch_dim, hidden_dim=max(2 * branch_dim, 64), dropout=dropout
            )
            self.lensing_encoder = MapCNN2D(
                lensing_shape[0], branch_dim, width=width, dropout=dropout
            )
            self.head = nn.Sequential(
                nn.Linear(3 * branch_dim + 3, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.GELU(),
            )

        def forward(self, x):
            if x.ndim != 2 or x.shape[1] != self.expected_input_dim:
                raise ValueError(
                    f"expected packed shape [B,{self.expected_input_dim}], got {tuple(x.shape)}"
                )
            b = x.shape[0]
            i = 0
            f = x[:, i : i + self.field_size].reshape(b, *self.field_shape)
            i += self.field_size
            p = x[:, i : i + self.spectrum_size]
            i += self.spectrum_size
            l = x[:, i : i + self.lensing_size].reshape(b, *self.lensing_shape)
            i += self.lensing_size
            mask = x[:, i : i + 3].clamp(0, 1)
            if not torch.isfinite(x).all():
                raise ValueError("non-finite values reached model encoder")
            ef = self.field_encoder(f) * mask[:, 0:1]
            ep = self.spectrum_encoder(p) * mask[:, 1:2]
            el = self.lensing_encoder(l) * mask[:, 2:3]
            return self.head(torch.cat([ef, ep, el, mask], dim=1))
else:

    class HybridObservationEncoder:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError(
                "Install nufield-starter[ml] to use the hybrid CNN encoder"
            )
