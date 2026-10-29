"""End-to-end multimodal encoders for controlled cosmological simulation experiments.

This network consumes already aligned batches. It does not align survey masks, redshift kernels,
angular footprints, or mismatched latent cosmologies; those remain responsibilities of the data
contract and forward model.
"""

from __future__ import annotations

from typing import Any, Literal

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch, nn = None, None

from nufield.ai.networks import (
    CompactFieldEncoder3D,
    MixtureDensityHead,
    ModalityFusionTransformer,
    ResidualSummaryEncoder,
)

if nn is not None:

    class _MapEncoder2D(nn.Module):
        def __init__(self, in_channels: int, embedding_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 24, 3, padding=1),
                nn.GELU(),
                nn.AvgPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.GELU(),
                nn.AvgPool2d(2),
                nn.Conv2d(48, 64, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, embedding_dim),
                nn.GELU(),
            )
            self.in_channels = in_channels

        def forward(self, x):
            if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) < 4:
                raise ValueError("map input must be [B,C,H,W] with configured channels and H,W>=4")
            return self.net(x)

    class _SummaryEncoder(nn.Module):
        def __init__(self, input_dim: int, embedding_dim: int):
            super().__init__()
            self.input_dim = input_dim
            self.net = ResidualSummaryEncoder(
                input_dim, embedding_dim=embedding_dim, width=128, depth=2
            )

        def forward(self, x):
            if x.ndim != 2 or x.shape[-1] != self.input_dim:
                raise ValueError("summary input shape mismatch")
            return self.net(x)

    class MultimodalPosteriorNetwork(nn.Module):
        """Fuse aligned field/map/summary inputs, then emit Gaussian or mixture posterior parameters.

        Configuration format::

            {"galaxy": {"kind": "field3d", "channels": 1},
             "lensing": {"kind": "map2d", "channels": 1},
             "pk": {"kind": "summary", "features": 48}}

        Every training sample must use the same modality keys and configured widths. Missing values
        should be zero-filled tensors plus an explicit availability mask; never infer availability
        from the tensor contents alone.
        """

        def __init__(
            self,
            modality_config: dict[str, dict[str, Any]],
            *,
            target_dim: int,
            width: int = 128,
            heads: int = 4,
            fusion_layers: int = 2,
            output_kind: Literal["gaussian", "mixture"] = "mixture",
            mixture_components: int = 8,
            dropout: float = 0.05,
        ):
            super().__init__()
            if not modality_config or target_dim < 1:
                raise ValueError("modality_config and positive target_dim are required")
            if output_kind not in {"gaussian", "mixture"}:
                raise ValueError("output_kind must be gaussian or mixture")
            self.modality_names = tuple(sorted(modality_config))
            self.modality_config = {k: dict(modality_config[k]) for k in self.modality_names}
            encoders = {}
            for name, cfg in self.modality_config.items():
                kind = cfg.get("kind")
                if kind == "field3d":
                    encoders[name] = CompactFieldEncoder3D(
                        int(cfg.get("channels", 1)),
                        width,
                        tuple(cfg.get("channels_per_stage", (16, 32, 64))),
                        dropout=dropout,
                    )
                elif kind == "map2d":
                    encoders[name] = _MapEncoder2D(int(cfg.get("channels", 1)), width)
                elif kind == "summary":
                    encoders[name] = _SummaryEncoder(int(cfg["features"]), width)
                else:
                    raise ValueError(f"modality {name}: kind must be field3d/map2d/summary")
            self.encoders = nn.ModuleDict(encoders)
            self.fusion = ModalityFusionTransformer(
                {k: width for k in self.modality_names},
                width=width,
                heads=heads,
                layers=fusion_layers,
                dropout=dropout,
            )
            self.output_kind = output_kind
            if output_kind == "gaussian":
                self.mean_head = nn.Linear(width, target_dim)
                self.logvar_head = nn.Linear(width, target_dim)
            else:
                self.mixture_head = MixtureDensityHead(
                    width, target_dim, n_components=mixture_components
                )

        def forward(self, modalities: dict[str, Any], availability):
            if set(modalities) != set(self.modality_names):
                raise ValueError("modality keys differ from configured keys")
            embeddings = {
                name: self.encoders[name](modalities[name]) for name in self.modality_names
            }
            fused = self.fusion(embeddings, availability)
            if self.output_kind == "gaussian":
                return {
                    "mean": self.mean_head(fused),
                    "log_variance": self.logvar_head(fused).clamp(-10, 6),
                }
            return self.mixture_head(fused)
else:

    class MultimodalPosteriorNetwork:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError(
                "Install nufield-starter[ml] to use MultimodalPosteriorNetwork"
            )


def validate_modality_batch(
    modalities: dict[str, Any], availability: Any, expected_names: tuple[str, ...]
) -> dict[str, Any]:
    """Validate leading batch dimensions and availability-mask shape before device transfer."""
    import numpy as np

    if set(modalities) != set(expected_names):
        raise ValueError("modality keys must match expected names exactly")
    counts = {name: int(value.shape[0]) for name, value in modalities.items()}
    if not counts or len(set(counts.values())) != 1:
        raise ValueError(f"modality batch counts mismatch: {counts}")
    mask = np.asarray(availability, dtype=bool)
    if mask.shape != (next(iter(counts.values())), len(expected_names)):
        raise ValueError("availability must have shape [batch,n_modalities]")
    if np.any(~mask.any(axis=1)):
        raise ValueError("each row must have at least one available modality")
    return {
        "batch_size": next(iter(counts.values())),
        "modality_names": list(expected_names),
        "availability_rates": mask.mean(axis=0).tolist(),
        "valid": True,
    }
