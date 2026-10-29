"""Architecture size estimates used to choose models before expensive training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelBudget:
    model_id: str
    estimated_parameters: int
    estimated_activation_elements_per_sample: int
    estimated_activation_gb_per_batch: float
    input_elements_per_sample: int
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prod(vals):
    p = 1
    for v in vals:
        if int(v) < 1:
            raise ValueError("dimensions must be positive")
        p *= int(v)
    return p


def estimate_cnn3d_budget(
    *,
    field_shape: tuple[int, ...],
    width: int = 12,
    blocks: int = 4,
    embedding_dim: int = 128,
    batch_size: int = 2,
    bytes_per_value: int = 4,
    activation_multiplier: float = 4.0,
) -> ModelBudget:
    """Rough residual-CNN budget; includes a multiplier because training retains activations."""
    shape = field_shape if len(field_shape) == 4 else (1, *field_shape)
    if len(shape) != 4:
        raise ValueError("field_shape must be [D,H,W] or [C,D,H,W]")
    c, d, h, w = shape
    if min(c, d, h, w, width, blocks, embedding_dim, batch_size, bytes_per_value) < 1:
        raise ValueError("dimensions and counts must be positive")
    # Approximation for a stem + residual 3x3x3 conv pairs + projection; excludes optimizer states.
    params = (
        c * width
        + width * width * 27
        + blocks * (2 * width * width * 27 + 4 * width)
        + (width * 2) * embedding_dim
        + embedding_dim
    )
    activations = batch_size * width * d * h * w * (1 + 2 * blocks) * activation_multiplier
    gib = activations * bytes_per_value / (1024**3)
    return ModelBudget(
        "cnn3d_npe",
        int(params),
        int(activations),
        float(gib),
        int(c * d * h * w),
        (
            "Approximation not a profiler; exact parameter count depends on chosen blocks.",
            "Does not include optimizer states, posterior density estimator, CUDA workspace or data staging.",
            "Run a measured pilot before allocating hardware.",
        ),
    )


def estimate_fno3d_budget(
    *,
    field_shape: tuple[int, ...],
    width: int = 12,
    blocks: int = 4,
    modes: tuple[int, int, int] = (6, 6, 6),
    embedding_dim: int = 128,
    batch_size: int = 2,
    bytes_per_value: int = 4,
    complex_parameter_bytes: int = 8,
) -> ModelBudget:
    shape = field_shape if len(field_shape) == 4 else (1, *field_shape)
    if len(shape) != 4:
        raise ValueError("field_shape must be [D,H,W] or [C,D,H,W]")
    c, d, h, w = shape
    if min(c, d, h, w, width, blocks, embedding_dim, batch_size, *modes) < 1:
        raise ValueError("dimensions/modes must be positive")
    spectral_weights = 4 * width * width * _prod(modes) * complex_parameter_bytes
    pointwise = blocks * width * width
    params = c * width + spectral_weights + pointwise + (width * 2) * embedding_dim + embedding_dim
    spectral_activations = batch_size * width * d * h * (w // 2 + 1) * complex_parameter_bytes * 4
    spatial_activations = batch_size * width * d * h * w * bytes_per_value * (1 + 2 * blocks)
    bytes_peak = spectral_activations + spatial_activations
    return ModelBudget(
        "fno3d_npe",
        int(params),
        int(bytes_peak / bytes_per_value),
        float(bytes_peak / (1024**3)),
        int(c * d * h * w),
        (
            "Fourier activation estimate depends on backend and retained graph; measure the actual model.",
            "Only considered for regularly gridded periodic simulation boxes.",
            "Does not include posterior estimator, optimizer states, FFT workspaces or data copies.",
        ),
    )
