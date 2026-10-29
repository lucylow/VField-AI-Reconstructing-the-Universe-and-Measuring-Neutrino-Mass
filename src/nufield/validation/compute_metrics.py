"""Resource estimates for planning, clearly separated from measured run telemetry."""

from __future__ import annotations

import numpy as np


def raw_array_megabytes(shape: tuple[int, ...], dtype: str = "float32") -> float:
    nbytes = np.dtype(dtype).itemsize
    if not shape or any(int(n) < 0 for n in shape):
        raise ValueError("invalid shape")
    return float(np.prod(shape, dtype=np.int64) * nbytes / (1024**2))


def field_dataset_storage_estimate(
    n_sims: int,
    grid_size: int,
    fields_per_sim: int = 3,
    dtype: str = "float32",
    compression_factor: float = 1.0,
) -> dict[str, float]:
    """Estimate raw and assumed compressed bytes; compression factor must be user-supplied."""
    if min(n_sims, grid_size, fields_per_sim) < 1 or not 0 < compression_factor <= 1:
        raise ValueError("invalid workload or compression factor")
    raw_mb = raw_array_megabytes((n_sims, fields_per_sim, grid_size, grid_size, grid_size), dtype)
    return {
        "raw_mb": raw_mb,
        "assumed_compressed_mb": raw_mb * compression_factor,
        "assumed_compression_factor": compression_factor,
        "warning": "estimate only; excludes FFT workspaces, parameters, metadata, copies, and filesystem overhead",
    }


def memory_budget_ok(estimated_mb: float, budget_mb: float, safety_factor: float = 2.0) -> bool:
    if min(estimated_mb, budget_mb) < 0 or safety_factor < 1:
        raise ValueError("invalid memory values")
    return estimated_mb * safety_factor <= budget_mb
