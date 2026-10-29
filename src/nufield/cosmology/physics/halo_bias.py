"""Approximate Sheth--Tormen halo bias for controlled halo-model baselines."""

from __future__ import annotations

import numpy as np


def sheth_tormen_linear_bias(
    nu: np.ndarray | float,
    *,
    delta_c: float = 1.686,
    a: float = 0.707,
    p: float = 0.3,
) -> np.ndarray:
    r"""Return b(ν)=1+(aν²-1)/δ_c + 2p/[δ_c(1+(aν²)^p)].

    This classic fit is a baseline. It is not calibrated for a particular halo finder, cosmology,
    redshift convention, massive-neutrino prescription, or selection function.
    """
    v = np.asarray(nu, float)
    if np.any(v <= 0) or np.any(~np.isfinite(v)) or delta_c <= 0 or a <= 0 or p < 0:
        raise ValueError("nu, delta_c, a must be positive and p non-negative")
    av2 = a * v**2
    return 1 + (av2 - 1) / delta_c + 2 * p / (delta_c * (1 + av2**p))


def galaxy_halo_bias_relation(halo_bias: np.ndarray, occupation_weight: np.ndarray) -> float:
    """Occupation-weighted large-scale galaxy bias from halo-bin bias and nonnegative weights."""
    b, w = np.asarray(halo_bias, float), np.asarray(occupation_weight, float)
    if b.ndim != 1 or w.shape != b.shape or np.any(~np.isfinite(b + w)) or np.any(w < 0):
        raise ValueError(
            "halo_bias and occupation_weight must be matching finite vectors; weights>=0"
        )
    if w.sum() <= 0:
        raise ValueError("occupation weights must sum to a positive value")
    return float(np.dot(b, w) / w.sum())


__all__ = ["sheth_tormen_linear_bias", "galaxy_halo_bias_relation"]
