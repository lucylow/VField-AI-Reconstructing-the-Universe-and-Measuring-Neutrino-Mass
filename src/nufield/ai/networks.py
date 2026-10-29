"""AI architectures aimed at robust simulation-based cosmology experiments.

Every architecture is a hypothesis to benchmark, not an asserted improvement. Install the optional
``ml`` dependency group to use these models. Neural outputs are not calibrated merely because they
are probabilistic; use the calibration and held-out-group evaluation utilities as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - runtime depends on the optional extra
    torch = None
    nn = None
    F = None


def require_torch() -> None:
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to use neural architectures")


if nn is not None:

    class ResidualMLPBlock(nn.Module):
        """Pre-activation residual block with optional dropout and layer normalization."""

        def __init__(self, width: int, *, dropout: float = 0.0, expansion: int = 2):
            super().__init__()
            if width < 1 or expansion < 1 or not 0 <= dropout < 1:
                raise ValueError("invalid residual block dimensions or dropout")
            hidden = width * expansion
            self.norm = nn.LayerNorm(width)
            self.layers = nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, width),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.layers(self.norm(x))

    class ResidualSummaryEncoder(nn.Module):
        """Parameter-efficient residual MLP for power-spectrum and summary features."""

        def __init__(
            self,
            input_dim: int,
            embedding_dim: int = 128,
            width: int = 192,
            depth: int = 4,
            dropout: float = 0.05,
        ):
            super().__init__()
            if min(input_dim, embedding_dim, width, depth) < 1:
                raise ValueError("dimensions and depth must be positive")
            self.input_dim = int(input_dim)
            self.embedding_dim = int(embedding_dim)
            self.stem = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width))
            self.blocks = nn.Sequential(
                *[ResidualMLPBlock(width, dropout=dropout) for _ in range(depth)]
            )
            self.head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, embedding_dim), nn.GELU()
            )

        def forward(self, x):
            if x.ndim != 2 or x.shape[-1] != self.input_dim:
                raise ValueError(f"expected [B,{self.input_dim}] summary input")
            return self.head(self.blocks(self.stem(x)))

    class HeteroscedasticRegressor(nn.Module):
        """Predict target means and diagonal aleatoric variances.

        Output dictionary contains ``mean`` and ``log_variance`` of shape [batch, targets].
        The variance is a conditional noise/ambiguity estimate, not epistemic uncertainty by itself.
        """

        def __init__(
            self,
            encoder: nn.Module,
            embedding_dim: int,
            target_dim: int,
            log_variance_bounds: tuple[float, float] = (-10.0, 6.0),
        ):
            super().__init__()
            if (
                min(embedding_dim, target_dim) < 1
                or log_variance_bounds[1] <= log_variance_bounds[0]
            ):
                raise ValueError("invalid dimensions or variance bounds")
            self.encoder = encoder
            self.target_dim = int(target_dim)
            self.low_logvar, self.high_logvar = map(float, log_variance_bounds)
            self.mean_head = nn.Linear(embedding_dim, target_dim)
            self.logvar_head = nn.Linear(embedding_dim, target_dim)

        def forward(self, x):
            h = self.encoder(x)
            return {
                "mean": self.mean_head(h),
                "log_variance": self.logvar_head(h).clamp(self.low_logvar, self.high_logvar),
            }

    class MixtureDensityHead(nn.Module):
        """Diagonal Gaussian mixture density for multi-modal conditional targets.

        Shape convention: logits [B,K], means/log_scales [B,K,P]. This density is a useful
        baseline for degeneracies but is not a replacement for a validated neural spline flow.
        """

        def __init__(
            self,
            embedding_dim: int,
            target_dim: int,
            n_components: int = 8,
            log_scale_bounds: tuple[float, float] = (-7.0, 4.0),
        ):
            super().__init__()
            if min(embedding_dim, target_dim, n_components) < 1:
                raise ValueError("mixture dimensions must be positive")
            lo, hi = log_scale_bounds
            if hi <= lo:
                raise ValueError("log scale upper bound must exceed lower bound")
            self.target_dim = int(target_dim)
            self.n_components = int(n_components)
            self.low_log_scale, self.high_log_scale = float(lo), float(hi)
            self.logits = nn.Linear(embedding_dim, n_components)
            self.means = nn.Linear(embedding_dim, n_components * target_dim)
            self.log_scales = nn.Linear(embedding_dim, n_components * target_dim)

        def forward(self, embedding):
            batch = embedding.shape[0]
            means = self.means(embedding).reshape(batch, self.n_components, self.target_dim)
            logs = self.log_scales(embedding).reshape(batch, self.n_components, self.target_dim)
            return {
                "logits": self.logits(embedding),
                "means": means,
                "log_scales": logs.clamp(self.low_log_scale, self.high_log_scale),
            }

    class MixtureDensityRegressor(nn.Module):
        """Encoder plus diagonal Gaussian mixture head with sampling and log-probability methods."""

        def __init__(
            self, encoder: nn.Module, embedding_dim: int, target_dim: int, n_components: int = 8
        ):
            super().__init__()
            self.encoder = encoder
            self.head = MixtureDensityHead(embedding_dim, target_dim, n_components)
            self.target_dim = int(target_dim)

        def forward(self, x):
            return self.head(self.encoder(x))

        @staticmethod
        def log_prob(params: dict[str, Any], target):
            target = target.unsqueeze(1)
            means, logs = params["means"], params["log_scales"]
            standardized = (target - means) * torch.exp(-logs)
            component = (
                -0.5 * standardized.square()
                - logs
                - 0.5
                * torch.log(torch.tensor(2 * torch.pi, device=target.device, dtype=target.dtype))
            )
            component = component.sum(dim=-1) + torch.log_softmax(params["logits"], dim=-1)
            return torch.logsumexp(component, dim=-1)

        @staticmethod
        def mean(params: dict[str, Any]):
            weights = torch.softmax(params["logits"], dim=-1).unsqueeze(-1)
            return (weights * params["means"]).sum(dim=1)

        @staticmethod
        def sample(params: dict[str, Any], n_samples: int = 1, *, generator=None):
            if n_samples < 1:
                raise ValueError("n_samples must be positive")
            weights = torch.softmax(params["logits"], dim=-1)
            indices = torch.multinomial(weights, n_samples, replacement=True, generator=generator)
            gather_idx = indices.unsqueeze(-1).expand(-1, -1, params["means"].shape[-1])
            means = torch.gather(params["means"], 1, gather_idx)
            scales = torch.gather(params["log_scales"].exp(), 1, gather_idx)
            noise = torch.randn(
                means.shape, dtype=means.dtype, device=means.device, generator=generator
            )
            return means + scales * noise

    class MaskedPointSetEncoder(nn.Module):
        """Permutation-invariant encoder for padded galaxy/object catalogs.

        Input ``points`` has shape [B,N,F], and ``mask`` has shape [B,N] with True for observed
        rows. Empty catalogs are rejected rather than silently encoded as a valid observation.
        """

        def __init__(self, point_dim: int, embedding_dim: int = 128, hidden_dim: int = 128):
            super().__init__()
            if min(point_dim, embedding_dim, hidden_dim) < 1:
                raise ValueError("dimensions must be positive")
            self.point_dim = int(point_dim)
            self.point_net = nn.Sequential(
                nn.Linear(point_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2 or 1),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2 or 1, 1),
            )
            self.output = nn.Sequential(nn.Linear(hidden_dim * 2, embedding_dim), nn.GELU())

        def forward(self, points, mask):
            if points.ndim != 3 or points.shape[-1] != self.point_dim:
                raise ValueError(f"expected points [B,N,{self.point_dim}]")
            if mask.shape != points.shape[:2]:
                raise ValueError("mask must have shape [B,N]")
            mask = mask.bool()
            if torch.any(mask.sum(dim=1) == 0):
                raise ValueError("every catalog sample must contain at least one observed point")
            encoded = self.point_net(points)
            neg_inf = torch.finfo(encoded.dtype).min
            scores = self.attention(encoded).squeeze(-1).masked_fill(~mask, neg_inf)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            attentive = (encoded * weights).sum(dim=1)
            masked = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
            mean = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
            return self.output(torch.cat([attentive, mean], dim=-1))

    class ModalityFusionTransformer(nn.Module):
        """Fuse named fixed-width modality embeddings with explicit missingness handling.

        Each modality uses a projection into a shared width and a learned modality token. Missing
        tokens are masked in self-attention; an always-present learned summary token prevents an
        all-missing sample from becoming an undefined attention problem. The caller should still
        reject samples with no usable modality at the data-contract stage.
        """

        def __init__(
            self,
            modality_dims: dict[str, int],
            width: int = 128,
            heads: int = 4,
            layers: int = 2,
            dropout: float = 0.05,
        ):
            super().__init__()
            if not modality_dims or any(int(v) < 1 for v in modality_dims.values()):
                raise ValueError("modality_dims must map names to positive feature widths")
            if width % heads:
                raise ValueError("width must be divisible by heads")
            if min(width, heads, layers) < 1:
                raise ValueError("width, heads and layers must be positive")
            self.modality_names = tuple(sorted(modality_dims))
            self.projections = nn.ModuleDict(
                {name: nn.Linear(int(modality_dims[name]), width) for name in self.modality_names}
            )
            self.tokens = nn.Parameter(torch.randn(len(self.modality_names), width) * 0.02)
            self.summary = nn.Parameter(torch.zeros(1, 1, width))
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=width * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(width)

        def forward(self, modalities: dict[str, Any], availability):
            if set(modalities) != set(self.modality_names):
                raise ValueError("modality keys must exactly match the model configuration")
            batch = next(iter(modalities.values())).shape[0]
            if availability.shape != (batch, len(self.modality_names)):
                raise ValueError("availability must have shape [B,n_modalities]")
            tokens = []
            for i, name in enumerate(self.modality_names):
                value = modalities[name]
                width = self.projections[name].in_features
                if value.shape != (batch, width):
                    raise ValueError(f"modality {name} must have shape [B,{width}]")
                token = self.projections[name](value) + self.tokens[i]
                tokens.append(token)
            x = torch.stack(tokens, dim=1)
            available = availability.bool()
            if torch.any(~available.any(dim=1)):
                raise ValueError("each sample must have at least one available modality")
            summary = self.summary.expand(batch, -1, -1)
            x = torch.cat([summary, x], dim=1)
            # Summary token is always visible; subsequent flags align to the modality token sequence.
            padding = torch.cat(
                [torch.zeros((batch, 1), dtype=torch.bool, device=x.device), ~available], dim=1
            )
            encoded = self.transformer(x, src_key_padding_mask=padding)
            return self.norm(encoded[:, 0])

    class Conv3DResidualBlock(nn.Module):
        """Two-convolution residual block for local spatial features."""

        def __init__(self, channels: int, *, groups: int = 8):
            super().__init__()
            groups = min(groups, channels)
            while channels % groups:
                groups -= 1
            self.net = nn.Sequential(
                nn.GroupNorm(groups, channels),
                nn.GELU(),
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
                nn.Conv3d(channels, channels, 3, padding=1),
            )

        def forward(self, x):
            return x + self.net(x)

    class CompactFieldEncoder3D(nn.Module):
        """Residual 3-D convolutional encoder intended for modest simulation grids."""

        def __init__(
            self,
            in_channels: int = 1,
            embedding_dim: int = 128,
            channels: tuple[int, ...] = (16, 32, 64),
            dropout: float = 0.05,
        ):
            super().__init__()
            if min(in_channels, embedding_dim, *channels) < 1:
                raise ValueError("channel dimensions must be positive")
            blocks = []
            current = in_channels
            for c in channels:
                blocks.extend(
                    [
                        nn.Conv3d(current, c, 3, padding=1),
                        nn.GELU(),
                        Conv3DResidualBlock(c),
                        nn.AvgPool3d(2, ceil_mode=True),
                    ]
                )
                current = c
            self.in_channels = in_channels
            self.network = nn.Sequential(
                *blocks,
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(current, embedding_dim),
                nn.GELU(),
            )

        def forward(self, x):
            if x.ndim != 5 or x.shape[1] != self.in_channels:
                raise ValueError(f"expected [B,{self.in_channels},D,H,W]")
            if min(x.shape[-3:]) < 4:
                raise ValueError("each field dimension must be at least 4")
            return self.network(x)

    class DeepEnsembleRegressor(nn.Module):
        """Container for independent probabilistic members; train each with a different seed."""

        def __init__(self, members: list[nn.Module]):
            super().__init__()
            if len(members) < 2:
                raise ValueError("a deep ensemble needs at least two independently trained members")
            self.members = nn.ModuleList(members)

        def forward(self, x):
            return [member(x) for member in self.members]

        def member_count(self) -> int:
            return len(self.members)

else:

    class _TorchRequired:
        def __init__(self, *args, **kwargs):
            require_torch()

    ResidualMLPBlock = ResidualSummaryEncoder = HeteroscedasticRegressor = _TorchRequired
    MixtureDensityHead = MixtureDensityRegressor = _TorchRequired
    MaskedPointSetEncoder = ModalityFusionTransformer = _TorchRequired
    Conv3DResidualBlock = CompactFieldEncoder3D = DeepEnsembleRegressor = _TorchRequired


@dataclass(frozen=True)
class ArchitectureProfile:
    """Descriptive, non-performance-based metadata for an architecture candidate."""

    name: str
    intended_input: str
    inductive_bias: str
    key_assumption: str
    primary_failure_mode: str
    recommended_baseline: str


def architecture_profiles() -> list[ArchitectureProfile]:
    """Return architecture choice notes suitable for experiment reports."""
    return [
        ArchitectureProfile(
            "ResidualSummaryEncoder",
            "P(k), BAO/RSD summaries",
            "tabular residual MLP",
            "summary features are sufficient for the chosen task",
            "compression may discard field information",
            "classical likelihood on the same summaries",
        ),
        ArchitectureProfile(
            "CompactFieldEncoder3D",
            "regular 3-D simulation fields",
            "local spatial convolution",
            "grid spacing and density convention are consistent",
            "survey masks and domain shift can be learned spuriously",
            "power-spectrum inference and a reconstruction baseline",
        ),
        ArchitectureProfile(
            "MixtureDensityRegressor",
            "low-dimensional conditional targets",
            "multi-modal density",
            "diagonal component covariance is adequate",
            "component collapse or poor tail calibration",
            "kernel posterior or SBI neural spline flow",
        ),
        ArchitectureProfile(
            "MaskedPointSetEncoder",
            "variable-size object catalogs",
            "permutation invariance",
            "per-object features and selection rules are declared",
            "selection function becomes an accidental shortcut",
            "clustering-summary baseline",
        ),
        ArchitectureProfile(
            "ModalityFusionTransformer",
            "matched galaxy/lensing/CMB embeddings",
            "cross-modal attention",
            "modalities share a consistent latent cosmology",
            "missingness or cross-survey mismatch bias",
            "single-modality and late-concatenation ablations",
        ),
    ]


def count_trainable_parameters(model: Any) -> int:
    """Count trainable parameters when PyTorch is installed."""
    require_torch()
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
