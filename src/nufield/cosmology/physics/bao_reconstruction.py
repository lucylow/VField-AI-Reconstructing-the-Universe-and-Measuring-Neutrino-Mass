"""Periodic-box displacement estimator used for BAO-reconstruction experiments.

This implements a simplified real-space shift field, not a survey-ready reconstruction pipeline.
It omits survey masks, selection gradients, random catalogs, boundary conditions, anisotropic RSD
corrections and the data/random displacement bookkeeping required by DESI analyses.
"""

from __future__ import annotations

import numpy as np


def reconstruction_shift_field(
    galaxy_overdensity: np.ndarray,
    box_size_mpc: float,
    *,
    bias: float,
    smoothing_radius_mpc: float = 15.0,
    growth_rate: float | None = None,
    line_of_sight_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Estimate the smoothed negative displacement field in a periodic box.

    The Fourier-space convention is s_i(k)=-i k_i/k² exp(-k²R²/2) delta_g(k)/b.
    If ``growth_rate`` is given, an approximate linear RSD correction factor
    1+f (k·n)^2/k² is applied in the denominator. Units of shift components are Mpc.
    """
    field = np.asarray(galaxy_overdensity, dtype=float)
    if field.ndim != 3 or len(set(field.shape)) != 1 or np.any(~np.isfinite(field)):
        raise ValueError("galaxy_overdensity must be a finite cubic 3-D field")
    if box_size_mpc <= 0 or bias <= 0 or smoothing_radius_mpc < 0:
        raise ValueError("box_size_mpc and bias must be positive; smoothing radius non-negative")
    if line_of_sight_axis not in (0, 1, 2):
        raise ValueError("line_of_sight_axis must be 0, 1, or 2")
    n = field.shape[0]
    freq = 2 * np.pi * np.fft.fftfreq(n, d=box_size_mpc / n)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    components = (kx, ky, kz)
    k2 = kx * kx + ky * ky + kz * kz
    fk = np.fft.fftn(field - field.mean())
    smoothing = np.exp(-0.5 * k2 * smoothing_radius_mpc**2)
    denom = bias * k2
    if growth_rate is not None:
        if not np.isfinite(growth_rate):
            raise ValueError("growth_rate must be finite when provided")
        mu_num = components[line_of_sight_axis] ** 2
        # This simple factor is only an exploratory linear RSD correction.
        denom = denom * (
            1 + growth_rate * np.divide(mu_num, k2, out=np.zeros_like(k2), where=k2 > 0)
        )
    denom[0, 0, 0] = np.inf
    shifts = []
    for ki in components:
        shift_k = -1j * ki * smoothing * fk / denom
        shift_k[0, 0, 0] = 0.0
        shifts.append(np.fft.ifftn(shift_k).real.astype(np.float32))
    return tuple(shifts)  # type: ignore[return-value]


def shift_positions_periodic(
    positions_mpc: np.ndarray,
    shift_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    box_size_mpc: float,
) -> np.ndarray:
    """Interpolate 3-D periodic shift grids at positions and return shifted positions modulo box."""
    from scipy.ndimage import map_coordinates

    pos = np.asarray(positions_mpc, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3 or box_size_mpc <= 0:
        raise ValueError("positions must have shape (N,3) and box_size_mpc positive")
    shape = np.asarray(shift_fields[0].shape)
    if any(np.asarray(s).shape != tuple(shape) for s in shift_fields) or len(set(shape)) != 1:
        raise ValueError("shift fields must be three equally-sized cubic grids")
    if np.any(~np.isfinite(pos)) or np.any(pos < 0) or np.any(pos >= box_size_mpc):
        raise ValueError("positions must be finite and inside the periodic box")
    coords = (pos / box_size_mpc * shape[0]).T
    shifts = np.column_stack(
        [
            map_coordinates(np.asarray(grid), coords, order=1, mode="grid-wrap")
            for grid in shift_fields
        ]
    )
    return np.mod(pos + shifts, box_size_mpc)


__all__ = ["reconstruction_shift_field", "shift_positions_periodic"]
