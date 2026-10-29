"""Exploratory BAO-wiggle diagnostics on tabulated spectra.

The smooth-spectrum split is intentionally a diagnostic filter, not a likelihood-ready no-wiggle
transfer function. For published BAO inference, use an agreed template, reconstruction model and
survey covariance instead of tuning the smoothing parameters to the desired answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class BAOWiggleDiagnostic:
    k_mpc_inv: np.ndarray
    smooth_pk_mpc3: np.ndarray
    fractional_wiggle: np.ndarray
    rms_fractional_wiggle: float
    k_range_mpc_inv: tuple[float, float]
    method: str
    caveat: str


def smooth_spectrum_logk(
    spectrum: MatterPowerSpectrum,
    *,
    window_length: int = 31,
    polyorder: int = 3,
    n_uniform: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth ln P against uniformly sampled ln k, then return P_smooth on the native k grid."""
    if window_length % 2 == 0 or window_length < 5 or polyorder < 1 or polyorder >= window_length:
        raise ValueError("window_length must be odd >=5 and polyorder in [1, window_length)")
    n = int(n_uniform or max(3 * len(spectrum.k_mpc_inv), window_length + 2))
    if n <= window_length:
        raise ValueError("n_uniform must exceed window_length")
    logk = np.log(spectrum.k_mpc_inv)
    uniform_logk = np.linspace(logk[0], logk[-1], n)
    uniform_logp = PchipInterpolator(logk, np.log(spectrum.pk_mpc3))(uniform_logk)
    smoothed_logp = savgol_filter(
        uniform_logp, window_length=window_length, polyorder=polyorder, mode="interp"
    )
    native_smoothed_logp = PchipInterpolator(uniform_logk, smoothed_logp)(logk)
    return spectrum.k_mpc_inv.copy(), np.exp(native_smoothed_logp)


def extract_bao_wiggles(
    spectrum: MatterPowerSpectrum,
    *,
    k_min_mpc_inv: float,
    k_max_mpc_inv: float,
    window_length: int = 31,
    polyorder: int = 3,
) -> BAOWiggleDiagnostic:
    """Compute P/P_smooth-1 and its in-range RMS as a descriptive diagnostic."""
    if k_min_mpc_inv <= 0 or k_max_mpc_inv <= k_min_mpc_inv:
        raise ValueError("require 0 < k_min < k_max")
    k, smooth = smooth_spectrum_logk(spectrum, window_length=window_length, polyorder=polyorder)
    wiggle = spectrum.pk_mpc3 / smooth - 1
    mask = (k >= k_min_mpc_inv) & (k <= k_max_mpc_inv)
    if mask.sum() < 3:
        raise ValueError("fewer than three samples in BAO diagnostic interval")
    rms = float(np.sqrt(np.mean(wiggle[mask] ** 2)))
    return BAOWiggleDiagnostic(
        k,
        smooth,
        wiggle,
        rms,
        (k_min_mpc_inv, k_max_mpc_inv),
        "Savitzky-Golay smoothing in log P vs log k",
        "Diagnostic only; not a no-wiggle physical transfer model.",
    )


__all__ = ["BAOWiggleDiagnostic", "smooth_spectrum_logk", "extract_bao_wiggles"]
