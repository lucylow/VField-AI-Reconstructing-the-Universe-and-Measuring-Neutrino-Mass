"""Convert voxel counts to periodic point positions for toy catalog demos."""

from __future__ import annotations

import numpy as np


def counts_to_positions(
    counts: np.ndarray, box_size: float, rng: np.random.Generator, max_objects: int | None = None
) -> np.ndarray:
    """Sample independent uniform positions within cells according to voxel counts.

    This produces an inhomogeneous Poisson point pattern from a pre-existing count mesh. It does not
    reproduce fiber assignment, galaxy exclusion, halo occupation, redshift-space velocities, or survey
    selection. `max_objects` is a guard for exploratory visualization, not an unbiased subsampling rule.
    """
    c = np.asarray(counts)
    if c.ndim != 3 or len(set(c.shape)) != 1 or np.any(c < 0) or not np.isfinite(c).all():
        raise ValueError("counts must be a non-negative cubic 3-D array")
    if np.any(c != np.floor(c)) or box_size <= 0:
        raise ValueError("counts must be integer-valued and box_size positive")
    n_total = int(c.sum())
    if n_total == 0:
        return np.empty((0, 3), dtype=np.float32)
    if max_objects is not None and max_objects < 1:
        raise ValueError("max_objects must be positive when supplied")
    cells = np.repeat(np.argwhere(c > 0), c[c > 0].astype(int), axis=0)
    if max_objects is not None and len(cells) > max_objects:
        # Randomly subsample objects for display. Inclusion probability is uniform across objects.
        selected = rng.choice(len(cells), size=max_objects, replace=False)
        cells = cells[selected]
    cell_size = box_size / c.shape[0]
    jitter = rng.uniform(0, cell_size, size=cells.shape)
    return ((cells * cell_size + jitter) % box_size).astype(np.float32)


def periodic_pairwise_displacements(positions: np.ndarray, box_size: float) -> np.ndarray:
    """Return a matrix of minimum-image pairwise displacements for small toy catalogs."""
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or box_size <= 0:
        raise ValueError("positions must be finite (N,3), box_size positive")
    delta = p[:, None, :] - p[None, :, :]
    delta -= box_size * np.round(delta / box_size)
    return delta.astype(np.float32)
