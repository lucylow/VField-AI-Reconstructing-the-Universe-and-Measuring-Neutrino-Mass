"""Transparent storage and model-memory estimates for dataset and architecture planning.

These formulas provide order-of-magnitude planning only. Real peaks depend on framework, activation
checkpointing, FFT workspaces, optimizer state, precision, padding, and allocator fragmentation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


@dataclass(frozen=True)
class StorageEstimate:
    n_simulations: int
    shape_per_simulation: tuple[int, ...]
    dtype: str
    raw_bytes: int
    derived_copies: int
    estimated_disk_bytes: int
    compression_factor_assumption: float
    warning: str = "Estimate only; inspect actual compression and data product layout."

    def to_dict(self):
        d = asdict(self)
        d["shape_per_simulation"] = list(self.shape_per_simulation)
        return d


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    model: str
    batch_size: int
    field_shape: tuple[int, ...]
    parameter_count_estimate: int
    input_bytes: int
    activation_bytes_rough: int
    optimizer_bytes_rough: int
    rough_total_bytes: int
    safe_device_memory_gb: float
    warning: str = "Conservative planning approximation, not measured GPU peak memory. Benchmark the first batch."

    def to_dict(self):
        d = asdict(self)
        d["field_shape"] = list(self.field_shape)
        return d


def estimate_storage(
    n_simulations: int,
    shape_per_simulation: tuple[int, ...],
    *,
    dtype: str = "float32",
    derived_copies: int = 3,
    compression_factor: float = 1.0,
) -> StorageEstimate:
    if n_simulations < 1 or not shape_per_simulation or min(shape_per_simulation) < 1:
        raise ValueError("simulation count and shape dimensions must be positive")
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; choose {sorted(_DTYPE_BYTES)}")
    if derived_copies < 1 or compression_factor < 0.05 or compression_factor > 1.0:
        raise ValueError("derived_copies must be >=1 and compression_factor in [0.05,1]")
    raw = int(n_simulations * math.prod(shape_per_simulation) * _DTYPE_BYTES[dtype])
    disk = int(raw * derived_copies * compression_factor)
    return StorageEstimate(
        n_simulations,
        tuple(shape_per_simulation),
        dtype,
        raw,
        derived_copies,
        disk,
        compression_factor,
    )


def estimate_training_memory(
    model: str,
    *,
    batch_size: int,
    field_shape: tuple[int, ...],
    width: int = 16,
    n_blocks: int = 4,
    dtype: str = "float32",
    optimizer_multiplier: float = 3.0,
    activation_factor: float = 10.0,
    safe_memory_fraction: float = 0.70,
) -> TrainingMemoryEstimate:
    """Estimate training memory for a 3-D field model with explicit conservative factors.

    `field_shape` is C,D,H,W. CNN activation estimate scales with voxels × width × a tunable factor;
    FNO adds spectral scratch space and uses a higher default activation factor. Parameter counts are
    rough and are intended only to reject obviously excessive laptop configurations early.
    """
    if batch_size < 1 or len(field_shape) != 4 or min(field_shape) < 1 or width < 1 or n_blocks < 1:
        raise ValueError("invalid batch, field shape, width, or block count")
    if dtype not in {"float16", "bfloat16", "float32", "float64"}:
        raise ValueError("unsupported floating dtype")
    if optimizer_multiplier < 1 or activation_factor < 1 or not 0 < safe_memory_fraction < 1:
        raise ValueError("invalid memory multipliers/fraction")
    bytes_per = _DTYPE_BYTES[dtype]
    n_vox = math.prod(field_shape)
    input_bytes = int(batch_size * n_vox * bytes_per)
    if model.lower() in {"cnn3d", "resnet3d", "unet3d"}:
        parameter_count = int(9 * width * width * n_blocks * 3 * 3 * 3 + 50 * width * width)
        factor = activation_factor
    elif model.lower() in {"fno3d", "fno"}:
        # Spectral weights are small relative to full-resolution feature-map activations.
        parameter_count = int(
            4 * width * width * n_blocks * 8**3 + width * math.prod(field_shape[:1])
        )
        factor = max(activation_factor, 16.0)
    else:
        raise ValueError("model must be cnn3d/resnet3d/unet3d or fno3d/fno")
    parameter_bytes = int(parameter_count * bytes_per)
    optimizer_bytes = int(parameter_bytes * optimizer_multiplier)
    activation_bytes = int(input_bytes * width * factor * n_blocks)
    total = int(input_bytes + activation_bytes + optimizer_bytes + parameter_bytes)
    safe_gb = total / (1024**3) / safe_memory_fraction
    return TrainingMemoryEstimate(
        model,
        batch_size,
        tuple(field_shape),
        parameter_count,
        input_bytes,
        activation_bytes,
        optimizer_bytes,
        total,
        safe_gb,
    )
