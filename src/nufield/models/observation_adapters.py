"""Select and construct SBI embedding networks behind a flat observation-vector interface.

`sbi` passes observations as vectors into a custom embedding module. This adapter restores the
appropriate shape before a CNN/FNO is applied. The caller is responsible for train-only scaling and
for validating that the field/grid/boundary conditions match the selected architecture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from nufield.exceptions import OptionalDependencyError

Architecture = Literal["hybrid", "spectrum_mlp", "cnn3d", "fno3d"]


@dataclass(frozen=True)
class EmbeddingSpec:
    architecture: Architecture
    output_dim: int
    input_width: int
    field_shape: tuple[int, ...] | None = None
    spectrum_dim: int | None = None
    lensing_shape: tuple[int, ...] | None = None
    width: int = 12
    modes: tuple[int, int, int] = (6, 6, 6)
    n_blocks: int = 4
    dropout: float = 0.05
    boundary_condition: str = "not_applicable"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("field_shape", "lensing_shape", "modes"):
            if d[key] is not None:
                d[key] = list(d[key])
        d["notes"] = list(self.notes)
        return d


def _torch_components():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise OptionalDependencyError(
            "Install nufield-starter[ml] to create neural embedding modules"
        ) from exc
    return torch, nn


def canonical_field_shape(fields) -> tuple[int, int, int, int]:
    import numpy as np

    arr = np.asarray(fields)
    if arr.ndim == 4:
        arr = arr[:, None]
    if arr.ndim != 5:
        raise ValueError("fields must be [N,D,H,W] or [N,C,D,H,W]")
    shape = tuple(int(d) for d in arr.shape[1:])
    if min(shape) < 4:
        raise ValueError("each field dimension must be at least four")
    return shape


def build_embedding(
    architecture: Architecture,
    *,
    field_shape: tuple[int, ...],
    spectrum_dim: int,
    lensing_shape: tuple[int, ...],
    input_width: int,
    output_dim: int = 128,
    width: int = 12,
    dropout: float = 0.05,
    modes: tuple[int, int, int] = (6, 6, 6),
    n_blocks: int = 4,
):
    """Build model and `EmbeddingSpec`; model input is always a flat [B,input_width] tensor."""
    torch, nn = _torch_components()
    if output_dim < 1 or width < 1 or input_width < 1:
        raise ValueError("dimensions must be positive")
    fshape = tuple(int(v) for v in field_shape)
    lshape = tuple(int(v) for v in lensing_shape)
    if architecture not in {"hybrid", "spectrum_mlp", "cnn3d", "fno3d"}:
        raise ValueError("unsupported architecture")
    if architecture == "hybrid":
        from nufield.models.hybrid_encoder import HybridObservationEncoder

        model = HybridObservationEncoder(
            field_shape=fshape,
            spectrum_dim=spectrum_dim,
            lensing_shape=lshape,
            output_dim=output_dim,
            branch_dim=48,
            width=width,
            dropout=dropout,
        )
        spec = EmbeddingSpec(
            architecture,
            output_dim,
            input_width,
            fshape,
            spectrum_dim,
            lshape,
            width,
            modes,
            n_blocks,
            dropout,
            "modality-aware; input must be pack_observations output",
            (
                "field, P(k), optional lensing channels",
                "availability bits required for missing modalities",
            ),
        )
    elif architecture == "spectrum_mlp":
        from nufield.models.baseline_encoders import SpectrumEmbeddingMLP

        if input_width != spectrum_dim:
            raise ValueError("spectrum_mlp input_width must equal spectrum_dim")
        model = SpectrumEmbeddingMLP(
            input_dim=spectrum_dim,
            output_dim=output_dim,
            hidden_dim=max(64, width * 8),
            dropout=dropout,
        )
        spec = EmbeddingSpec(
            architecture,
            output_dim,
            input_width,
            fshape,
            spectrum_dim,
            None,
            width,
            modes,
            n_blocks,
            dropout,
            "summary-only",
            ("power-spectrum features only", "no field morphology"),
        )
    else:
        n_field = int(__import__("math").prod(fshape))
        if input_width != n_field:
            raise ValueError("field model input_width must equal product(field_shape)")
        if architecture == "cnn3d":
            from nufield.models.hybrid_encoder import FieldCNN3D

            core = FieldCNN3D(
                in_channels=fshape[0], output_dim=output_dim, width=width, dropout=dropout
            )
            bc = "boundary effects handled by zero-padded convolutions; check edge sensitivity"
            notes = ("field-only residual CNN", "expects standardized field values")
        else:
            from nufield.models.spectral_encoder import FNO3DEmbedding

            core = FNO3DEmbedding(
                in_channels=fshape[0],
                width=width,
                output_dim=output_dim,
                modes=modes,
                n_blocks=n_blocks,
                dropout=dropout,
            )
            bc = "periodic simulation box only"
            notes = (
                "field-only Fourier Neural Operator",
                "not suitable for masked survey volumes without a window model",
            )

        class FlatFieldEmbedding(nn.Module):
            def __init__(self, net, expected_shape):
                super().__init__()
                self.net = net
                self.expected_shape = tuple(expected_shape)

            def forward(self, x):
                if x.ndim != 2 or x.shape[1] != int(__import__("math").prod(self.expected_shape)):
                    raise ValueError("flattened field input width mismatch")
                return self.net(x.reshape(x.shape[0], *self.expected_shape))

        model = FlatFieldEmbedding(core, fshape)
        spec = EmbeddingSpec(
            architecture,
            output_dim,
            input_width,
            fshape,
            spectrum_dim,
            None,
            width,
            modes,
            n_blocks,
            dropout,
            bc,
            notes,
        )
    return model, spec
