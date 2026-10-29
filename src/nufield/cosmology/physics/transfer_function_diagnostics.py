"""Transfer-function shape and spectral-support diagnostics.

The routines quantify properties of supplied spectra without inventing a transfer function. A power
spectrum ratio can be interpreted as a transfer-function-squared ratio only when primordial amplitudes,
units, redshift, gauge conventions, and component definitions are compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from nufield.cosmology.physics.sigma8_validation import dimensionless_power, spectral_slope
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class TransferRatioReport:
    k_mpc_inv: tuple[float, ...]
    power_ratio: tuple[float, ...]
    transfer_amplitude_ratio: tuple[float, ...]
    low_k_median: float
    high_k_median: float
    high_over_low: float
    k_range: tuple[float, float]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "k_mpc_inv": list(self.k_mpc_inv),
            "power_ratio": list(self.power_ratio),
            "transfer_amplitude_ratio": list(self.transfer_amplitude_ratio),
            "low_k_median": self.low_k_median,
            "high_k_median": self.high_k_median,
            "high_over_low": self.high_over_low,
            "k_range": list(self.k_range),
            "caveats": list(self.caveats),
        }


def compare_power_ratio(
    numerator: MatterPowerSpectrum,
    denominator: MatterPowerSpectrum,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
    n_k: int = 256,
    low_k_fraction: float = 0.2,
    high_k_fraction: float = 0.2,
    require_same_redshift: bool = True,
) -> TransferRatioReport:
    """Compare spectra on a common logarithmic support and report square-root ratios.

    The reported ``transfer_amplitude_ratio`` is merely sqrt(P1/P2). Interpreting it as a physical
    transfer-function ratio assumes common primordial spectrum and normalization and matching total/cb
    component definitions. This routine intentionally cannot verify all of those assumptions.
    """
    if require_same_redshift and not np.isclose(
        numerator.redshift, denominator.redshift, atol=1e-9, rtol=0
    ):
        raise ValueError("spectrum redshifts differ")
    lo = max(
        float(numerator.k_mpc_inv[0]),
        float(denominator.k_mpc_inv[0]),
        float(k_min_mpc_inv) if k_min_mpc_inv is not None else 0.0,
    )
    hi = min(
        float(numerator.k_mpc_inv[-1]),
        float(denominator.k_mpc_inv[-1]),
        float(k_max_mpc_inv) if k_max_mpc_inv is not None else float("inf"),
    )
    if not 0 < lo < hi or n_k < 8 or not 0 < low_k_fraction < 0.5 or not 0 < high_k_fraction < 0.5:
        raise ValueError("invalid shared k range or fractions")
    k = np.geomspace(lo, hi, n_k)
    p1 = PchipInterpolator(np.log(numerator.k_mpc_inv), numerator.pk_mpc3)(np.log(k))
    p0 = PchipInterpolator(np.log(denominator.k_mpc_inv), denominator.pk_mpc3)(np.log(k))
    if np.any(p1 <= 0) or np.any(p0 <= 0):
        raise ValueError("power spectra must be positive on comparison grid")
    ratio = p1 / p0
    nband = max(2, int(n_k * min(low_k_fraction, high_k_fraction)))
    low = float(np.median(ratio[:nband]))
    high = float(np.median(ratio[-nband:]))
    return TransferRatioReport(
        tuple(k),
        tuple(ratio),
        tuple(np.sqrt(ratio)),
        low,
        high,
        high / low,
        (lo, hi),
        (
            "Ratio is meaningful only for matching power units and compatible density components.",
            "sqrt(P1/P0) is not a transfer ratio if primordial normalizations differ.",
            "Low/high bands are descriptive summaries, not formal free-streaming-scale inference.",
        ),
    )


def spectral_support_report(
    spectrum: MatterPowerSpectrum,
    *,
    required_k: tuple[float, float],
) -> dict[str, object]:
    """Audit whether a spectrum covers a required interval and quantify logarithmic margins."""
    lo, hi = map(float, required_k)
    if not 0 < lo < hi:
        raise ValueError("required_k must have 0 < min < max")
    k0, k1 = float(spectrum.k_mpc_inv[0]), float(spectrum.k_mpc_inv[-1])
    lower_ok, upper_ok = k0 <= lo, k1 >= hi
    return {
        "available_k_range": [k0, k1],
        "required_k_range": [lo, hi],
        "lower_supported": lower_ok,
        "upper_supported": upper_ok,
        "fully_supported": bool(lower_ok and upper_ok),
        "lower_log_margin": float(np.log(lo / k0)) if lower_ok else float(np.log(k0 / lo)),
        "upper_log_margin": float(np.log(k1 / hi)) if upper_ok else float(np.log(hi / k1)),
        "linear_flag": bool(spectrum.linear),
        "source": spectrum.source,
        "warning": "Interpolation inside support does not validate the physical assumptions of the spectrum.",
    }


def summarize_spectrum_shape(spectrum: MatterPowerSpectrum) -> dict[str, object]:
    """Summarize Δ², log-slope and broad shape ranges for diagnostics and regression tests."""
    k, p = spectrum.k_mpc_inv, spectrum.pk_mpc3
    slopes = spectral_slope(k, p)
    delta = dimensionless_power(k, p)
    return {
        "k_range_mpc_inv": [float(k[0]), float(k[-1])],
        "n_k": int(k.size),
        "power_range_mpc3": [float(p.min()), float(p.max())],
        "dimensionless_power_range": [float(delta.min()), float(delta.max())],
        "log_slope_range": [float(slopes.min()), float(slopes.max())],
        "redshift": float(spectrum.redshift),
        "linear": bool(spectrum.linear),
        "source": spectrum.source,
    }


__all__ = [
    "TransferRatioReport",
    "compare_power_ratio",
    "spectral_support_report",
    "summarize_spectrum_shape",
]
