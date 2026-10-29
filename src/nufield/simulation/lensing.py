"""Toy projection channel for exercising multimodal APIs; not a lensing simulator."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def toy_projected_map(
    density_field: np.ndarray,
    *,
    smoothing_sigma: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Project along the final axis and smooth to make a lensing-like 2-D map.

    This operator ignores lensing kernels, source redshift distributions, reduced shear, masks,
    shape noise, geometry, and light-cone effects. Name outputs `toy_projection`, not measured lensing.
    """
    field = np.asarray(density_field, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("density_field must be a 3-D array")
    if smoothing_sigma < 0 or noise_std < 0:
        raise ValueError("smoothing_sigma and noise_std must be non-negative")
    projection = field.sum(axis=-1) / np.sqrt(field.shape[-1])
    if smoothing_sigma:
        projection = gaussian_filter(projection, sigma=smoothing_sigma, mode="wrap")
    if noise_std:
        if rng is None:
            raise ValueError("rng must be provided when noise_std > 0")
        projection = projection + rng.normal(0, noise_std, size=projection.shape)
    return projection.astype(np.float32)
