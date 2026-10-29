"""Data-dependent linear-regime diagnostics; no universal k_max is assumed."""

from __future__ import annotations

import numpy as np

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


def dimensionless_power(spectrum: MatterPowerSpectrum) -> np.ndarray:
    """Return Δ²(k)=k³P(k)/(2π²)."""
    return spectrum.k_mpc_inv**3 * spectrum.pk_mpc3 / (2 * np.pi**2)


def linear_mode_mask(
    spectrum: MatterPowerSpectrum,
    *,
    delta2_max: float,
    k_max_mpc_inv: float | None = None,
) -> np.ndarray:
    """Select modes with Δ²(k)<=delta2_max and within optional analysis scale cut.

    This is a heuristic regime mask. Nonlinear corrections depend on cosmology and redshift; the
    threshold should be varied in robustness tests and validated for the selected observable.
    """
    if delta2_max <= 0 or not np.isfinite(delta2_max):
        raise ValueError("delta2_max must be finite and positive")
    mask = dimensionless_power(spectrum) <= delta2_max
    if k_max_mpc_inv is not None:
        if k_max_mpc_inv <= 0:
            raise ValueError("k_max_mpc_inv must be positive")
        mask &= spectrum.k_mpc_inv <= k_max_mpc_inv
    return mask


def first_k_crossing(spectrum: MatterPowerSpectrum, delta2_threshold: float = 1.0) -> float | None:
    """Return the first interpolated k where Δ² crosses threshold, or None when not crossed."""
    if delta2_threshold <= 0:
        raise ValueError("delta2_threshold must be positive")
    d2 = dimensionless_power(spectrum)
    above = np.flatnonzero(d2 >= delta2_threshold)
    if above.size == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(spectrum.k_mpc_inv[0])
    x0, x1 = np.log(spectrum.k_mpc_inv[i - 1 : i + 1])
    y0, y1 = np.log(d2[i - 1 : i + 1])
    if y1 == y0:
        return float(spectrum.k_mpc_inv[i])
    frac = (np.log(delta2_threshold) - y0) / (y1 - y0)
    return float(np.exp(x0 + frac * (x1 - x0)))


def scale_support_report(
    spectrum: MatterPowerSpectrum, required_k_min: float, required_k_max: float
) -> dict[str, object]:
    """Report whether a spectrum has explicit scale support for a requested analysis."""
    if required_k_min <= 0 or required_k_max <= required_k_min:
        raise ValueError("require 0 < required_k_min < required_k_max")
    return {
        "requested_k_range_mpc_inv": [required_k_min, required_k_max],
        "available_k_range_mpc_inv": [float(spectrum.k_mpc_inv[0]), float(spectrum.k_mpc_inv[-1])],
        "covers_range": bool(
            spectrum.k_mpc_inv[0] <= required_k_min and spectrum.k_mpc_inv[-1] >= required_k_max
        ),
        "extrapolation_allowed": False,
        "linear_tag": spectrum.linear,
    }


__all__ = ["dimensionless_power", "linear_mode_mask", "first_k_crossing", "scale_support_report"]
