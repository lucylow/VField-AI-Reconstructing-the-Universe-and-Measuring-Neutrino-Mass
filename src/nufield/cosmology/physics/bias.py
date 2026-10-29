"""Simple galaxy-bias and stochastic contributions for controlled clustering tests."""

from __future__ import annotations

import numpy as np


def linear_galaxy_power(
    pk_matter: np.ndarray, bias: float, shot_noise_mpc3: float = 0.0
) -> np.ndarray:
    """P_g(k)=b1² P_m(k)+P_shot for a constant linear bias baseline."""
    p = np.asarray(pk_matter, float)
    if np.any(p < 0) or np.any(~np.isfinite(p)) or bias <= 0 or shot_noise_mpc3 < 0:
        raise ValueError(
            "P_m must be finite/non-negative, bias positive and shot noise non-negative"
        )
    return bias**2 * p + shot_noise_mpc3


def polynomial_bias(
    pk_matter: np.ndarray,
    k_mpc_inv: np.ndarray,
    b1: float,
    b_nabla: float = 0.0,
    b2_loop_proxy: float = 0.0,
    *,
    stochastic_power_mpc3: float = 0.0,
) -> np.ndarray:
    r"""Phenomenological P_g=[b1+b_nabla k²]² P_m + b2_loop_proxy P_m² + P_stoch.

    This is an engineering nuisance template, not a complete EFT-of-LSS one-loop model. Parameter
    names intentionally say ``proxy`` to discourage physical interpretation without calibration.
    """
    p, k = np.asarray(pk_matter, float), np.asarray(k_mpc_inv, float)
    if p.shape != k.shape or np.any(p < 0) or np.any(k < 0) or np.any(~np.isfinite(p + k)):
        raise ValueError("k and P must be finite non-negative arrays of equal shape")
    if b1 <= 0 or stochastic_power_mpc3 < 0:
        raise ValueError("b1 must be positive and stochastic power non-negative")
    scale_bias = b1 + b_nabla * k**2
    return scale_bias**2 * p + b2_loop_proxy * p**2 + stochastic_power_mpc3


def scale_dependent_bias_validity(k_mpc_inv: np.ndarray, k_max_mpc_inv: float) -> np.ndarray:
    """Mask modes below an explicitly analyst-selected scale cut."""
    k = np.asarray(k_mpc_inv, float)
    if k_max_mpc_inv <= 0 or np.any(k < 0) or np.any(~np.isfinite(k)):
        raise ValueError("k must be finite/non-negative and cut positive")
    return k <= k_max_mpc_inv


__all__ = ["linear_galaxy_power", "polynomial_bias", "scale_dependent_bias_validity"]
