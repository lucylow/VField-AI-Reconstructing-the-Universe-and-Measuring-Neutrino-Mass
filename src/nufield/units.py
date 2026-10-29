"""Explicit unit conversion helpers; keep use visible in code and metadata."""

from __future__ import annotations

import numpy as np


def k_hmpc_to_inv_mpc(k_h_mpc: np.ndarray, h: float) -> np.ndarray:
    """Convert numerical k from h/Mpc to 1/Mpc: k_phys = h * k_h."""
    k = np.asarray(k_h_mpc, dtype=float)
    if h <= 0 or not np.isfinite(k).all():
        raise ValueError("h must be positive and k finite")
    return k * h


def power_mpc3_to_mpc_over_h_cubed(power_mpc3: np.ndarray, h: float) -> np.ndarray:
    """Convert P in Mpc^3 to numerical (Mpc/h)^3, multiplying by h^3."""
    p = np.asarray(power_mpc3, dtype=float)
    if h <= 0 or not np.isfinite(p).all():
        raise ValueError("h must be positive and power finite")
    return p * h**3


def validate_box_units(box_size: float, unit: str) -> None:
    if box_size <= 0 or unit not in {"Mpc", "Mpc/h", "cMpc", "cMpc/h"}:
        raise ValueError("box must be positive and unit explicitly supported")
