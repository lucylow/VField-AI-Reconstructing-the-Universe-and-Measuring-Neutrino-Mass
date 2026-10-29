"""Cloud-in-cell particle-to-mesh assignment for periodic cubic simulation boxes."""

from __future__ import annotations

import numpy as np


def cic_density_grid(
    positions_mpc: np.ndarray,
    box_size_mpc: float,
    grid_size: int,
    *,
    weights: np.ndarray | None = None,
    overdensity: bool = True,
) -> np.ndarray:
    """Assign particles to a periodic grid with trilinear cloud-in-cell weights.

    The returned field is mass/count density in cell units if ``overdensity=False``; otherwise it is
    rho/<rho>-1. The input catalogue must already have a consistent coordinate frame and selection.
    This utility does not correct redshift-space positions or survey incompleteness.
    """
    pos = np.asarray(positions_mpc, float)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] < 1:
        raise ValueError("positions_mpc must have shape (N,3) with N>=1")
    if np.any(~np.isfinite(pos)) or box_size_mpc <= 0 or grid_size < 2:
        raise ValueError("positions must be finite; box_size positive and grid_size >=2")
    if np.any(pos < 0) or np.any(pos >= box_size_mpc):
        raise ValueError("periodic particle coordinates must be in [0, box_size_mpc)")
    wt = np.ones(pos.shape[0], float) if weights is None else np.asarray(weights, float)
    if wt.shape != (pos.shape[0],) or np.any(~np.isfinite(wt)) or np.any(wt < 0):
        raise ValueError("weights must be finite non-negative and have one value per particle")
    scaled = pos * (grid_size / box_size_mpc)
    base = np.floor(scaled).astype(np.int64)
    frac = scaled - base
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    for dx in (0, 1):
        wx = (1 - frac[:, 0]) if dx == 0 else frac[:, 0]
        ix = (base[:, 0] + dx) % grid_size
        for dy in (0, 1):
            wy = (1 - frac[:, 1]) if dy == 0 else frac[:, 1]
            iy = (base[:, 1] + dy) % grid_size
            for dz in (0, 1):
                wz = (1 - frac[:, 2]) if dz == 0 else frac[:, 2]
                iz = (base[:, 2] + dz) % grid_size
                np.add.at(grid, (ix, iy, iz), wt * wx * wy * wz)
    if not np.isclose(grid.sum(), wt.sum(), rtol=1e-10, atol=1e-10):
        raise ArithmeticError("CIC assignment failed mass/count conservation")
    if not overdensity:
        return grid.astype(np.float32)
    mean = grid.mean()
    if mean <= 0:
        raise ValueError("cannot construct overdensity from zero total weight")
    return (grid / mean - 1.0).astype(np.float32)


def random_catalogue(n_objects: int, box_size_mpc: float, *, seed: int) -> np.ndarray:
    """Generate a uniform periodic random catalogue for pipeline tests."""
    if n_objects < 1 or box_size_mpc <= 0 or seed < 0:
        raise ValueError("n_objects and box_size must be positive; seed non-negative")
    return np.random.default_rng(seed).uniform(0.0, box_size_mpc, size=(n_objects, 3))


__all__ = ["cic_density_grid", "random_catalogue"]
