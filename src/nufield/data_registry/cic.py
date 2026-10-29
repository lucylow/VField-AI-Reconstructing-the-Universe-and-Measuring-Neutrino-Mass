"""Mass-assignment utilities for periodic simulation particle positions.

This is a compact Cloud-in-Cell (CIC) deposit for particle positions explicitly expressed in the
periodic box coordinate convention [0,L). It is intended for controlled periodic simulation boxes,
not masked survey catalogs or sky coordinates. The implementation is correct but not optimized for
very large N-body catalogs; use a verified high-performance mesh assignment package for production.
"""

from __future__ import annotations

import numpy as np


def cloud_in_cell(
    positions: np.ndarray,
    *,
    box_size: float,
    grid_size: int,
    weights: np.ndarray | None = None,
    periodic: bool = True,
) -> np.ndarray:
    """Deposit particle weights onto a cubic mesh using trilinear CIC mass assignment.

    Parameters
    ----------
    positions: shape [N,3], in the same length unit as `box_size`.
    box_size: positive side length of a cubic simulation volume.
    grid_size: number of cells along every dimension.
    weights: optional finite, non-negative per-particle weights. Total assigned weight is preserved
        up to floating-point accumulation error.
    periodic: must be True for this periodic-box implementation. Non-periodic boundaries require an
        explicitly designed mass assignment and selection treatment.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 1:
        raise ValueError("positions must have shape [N,3] with N>=1")
    if not np.isfinite(pos).all():
        raise ValueError("positions must be finite")
    if not np.isfinite(box_size) or box_size <= 0:
        raise ValueError("box_size must be positive and finite")
    if int(grid_size) != grid_size or grid_size < 2:
        raise ValueError("grid_size must be an integer >=2")
    if not periodic:
        raise ValueError(
            "this function only supports periodic boxes; provide a validated nonperiodic mesh builder"
        )
    if np.any(pos < 0) or np.any(pos >= box_size):
        raise ValueError(
            "positions must already be wrapped into [0, box_size); wrap explicitly if appropriate"
        )
    w = (
        np.ones(len(pos), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if w.shape != (len(pos),) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be finite nonnegative [N]")
    cell = box_size / grid_size
    q = pos / cell
    base = np.floor(q).astype(np.int64)
    frac = q - base
    mesh = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    # Eight corners; np.add.at handles repeated integer indices safely.
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1 - frac[:, 0]
        ix = (base[:, 0] + dx) % grid_size
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1 - frac[:, 1]
            iy = (base[:, 1] + dy) % grid_size
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1 - frac[:, 2]
                iz = (base[:, 2] + dz) % grid_size
                np.add.at(mesh, (ix, iy, iz), w * wx * wy * wz)
    return mesh


def mesh_to_overdensity(mesh: np.ndarray, *, epsilon: float = 0.0) -> np.ndarray:
    """Convert counts/mass-per-cell to delta = rho/rho_bar - 1; empty total input is rejected."""
    arr = np.asarray(mesh, dtype=np.float64)
    if arr.ndim != 3 or len(set(arr.shape)) != 1:
        raise ValueError("mesh must be a cubic 3-D array")
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("mesh must be finite and nonnegative")
    mean = float(arr.mean())
    if mean <= epsilon:
        raise ValueError("mesh mean must be positive")
    return arr / mean - 1.0


def cloud_in_cell_overdensity(
    positions: np.ndarray, *, box_size: float, grid_size: int, weights: np.ndarray | None = None
) -> np.ndarray:
    """Convenience function returning the periodic CIC overdensity field."""
    return mesh_to_overdensity(
        cloud_in_cell(positions, box_size=box_size, grid_size=grid_size, weights=weights)
    )
