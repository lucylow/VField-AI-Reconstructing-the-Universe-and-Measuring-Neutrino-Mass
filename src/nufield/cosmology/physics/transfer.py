"""Transfer-response quantities computed from a primordial spectrum and linear matter spectrum."""

from __future__ import annotations

import math

import numpy as np

from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.primordial import primordial_dimensionless_power
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


def matter_response_amplitude(
    spectrum: MatterPowerSpectrum,
    params: PhysicalCosmology,
    *,
    k_mpc_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Infer |M(k,z)| from P_m=(2π²/k³) Δ_R² |M|².

    The response is an amplitude inferred from the supplied matter spectrum, not a CLASS/CAMB transfer
    variable with a particular sign/gauge convention. Compare this ratio only at matching redshift and
    with a clearly documented definition of the total-matter variable.
    """
    k = spectrum.k_mpc_inv if k_mpc_inv is None else np.asarray(k_mpc_inv, float)
    p = spectrum.evaluate(k)
    delta_r2 = primordial_dimensionless_power(k, params)
    delta_m2 = k**3 * p / (2 * math.pi**2)
    return k, np.sqrt(delta_m2 / delta_r2)


def effective_log_slope(k: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return d ln P / d ln k using centered interior finite differences."""
    kk, pp = np.asarray(k, float), np.asarray(p, float)
    if kk.ndim != 1 or pp.shape != kk.shape or len(kk) < 3:
        raise ValueError("k and p must be matching 1-D arrays with >=3 samples")
    if (
        np.any(kk <= 0)
        or np.any(pp <= 0)
        or np.any(~np.isfinite(kk + pp))
        or np.any(np.diff(kk) <= 0)
    ):
        raise ValueError("k and p must be finite/positive, with k increasing")
    return np.gradient(np.log(pp), np.log(kk), edge_order=2)


def fractional_transfer_ratio(
    spectrum_a: MatterPowerSpectrum,
    spectrum_b: MatterPowerSpectrum,
) -> tuple[np.ndarray, np.ndarray]:
    """Return P_b/P_a on common k; despite the name this is a power ratio, not amplitude ratio."""
    if spectrum_a.redshift != spectrum_b.redshift:
        raise ValueError("spectra must share redshift")
    lo = max(spectrum_a.k_mpc_inv[0], spectrum_b.k_mpc_inv[0])
    hi = min(spectrum_a.k_mpc_inv[-1], spectrum_b.k_mpc_inv[-1])
    k = spectrum_a.k_mpc_inv[(spectrum_a.k_mpc_inv >= lo) & (spectrum_a.k_mpc_inv <= hi)]
    if len(k) < 3:
        raise ValueError("insufficient common k support")
    return k, spectrum_b.evaluate(k) / spectrum_a.evaluate(k)


__all__ = ["matter_response_amplitude", "effective_log_slope", "fractional_transfer_ratio"]
