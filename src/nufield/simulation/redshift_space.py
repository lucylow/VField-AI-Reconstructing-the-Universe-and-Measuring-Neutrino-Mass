"""Toy redshift-space perturbation utilities.

These are generic coordinate perturbations for stress testing. They do not implement a physical
redshift-space distortion model or peculiar velocities from a gravity solver.
"""

from __future__ import annotations

import numpy as np


def perturb_line_of_sight_coordinate(
    positions: np.ndarray, shifts: np.ndarray, box_size: float, los_axis: int = 2
) -> np.ndarray:
    """Apply periodic coordinate shifts along a designated line of sight."""
    pos = np.asarray(positions, dtype=np.float64)
    sh = np.asarray(shifts, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or sh.shape not in {(len(pos),), (len(pos), 1)}:
        raise ValueError("positions must be (N,3) and shifts must be (N,) or (N,1)")
    if not np.isfinite(pos).all() or not np.isfinite(sh).all() or box_size <= 0:
        raise ValueError("inputs must be finite and box_size positive")
    if los_axis not in {0, 1, 2}:
        raise ValueError("los_axis must be 0, 1, or 2")
    out = pos.copy()
    out[:, los_axis] = np.mod(out[:, los_axis] + sh.ravel(), box_size)
    return out


def gaussian_velocity_shifts(
    n_objects: int, dispersion: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw toy random LOS shifts to test robustness to coordinate perturbations."""
    if n_objects < 1 or dispersion < 0:
        raise ValueError("n_objects must be positive and dispersion non-negative")
    return rng.normal(0, dispersion, size=n_objects)
