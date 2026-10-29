"""Generate a periodic Gaussian random field using the toy spectrum."""

from __future__ import annotations

import numpy as np

from nufield.parameters import CosmologyParameters
from nufield.simulation.power import toy_matter_power, wave_number_grid


def generate_toy_density_field(
    grid_size: int, box_size_mpc_h: float, params: CosmologyParameters, rng: np.random.Generator
) -> np.ndarray:
    """Generate a finite real-valued Gaussian field.

    This uses a phenomenological spectrum and a periodic box. It does not implement growth history,
    2LPT, PM evolution, particle dynamics, baryons, or neutrino phase-space effects.
    """
    if grid_size < 4 or box_size_mpc_h <= 0:
        raise ValueError("grid_size must be >=4 and box_size_mpc_h must be positive")
    white = rng.normal(size=(grid_size, grid_size, grid_size))
    white_k = np.fft.fftn(white)
    k = wave_number_grid(grid_size, box_size_mpc_h)
    power = toy_matter_power(k, params)
    shaped_k = white_k * np.sqrt(power)
    field = np.fft.ifftn(shaped_k).real
    field -= field.mean()
    std = float(field.std())
    if not np.isfinite(std) or std < 1e-12:
        raise FloatingPointError("Toy field has a degenerate or non-finite variance")
    # Preserve sigma8-like amplitude information without claiming an actual 8 Mpc/h top-hat sigma8.
    field = field / std * params.sigma8 * 0.35
    return field.astype(np.float32)
