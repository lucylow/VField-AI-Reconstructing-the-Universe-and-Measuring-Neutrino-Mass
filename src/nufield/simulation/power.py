"""Phenomenological power spectrum for software smoke tests only.

This function is intentionally not a Boltzmann solution. Its scale-dependent `sum_mnu_ev`
response is an adjustable toy signal used to exercise data flow and inference code. It must not be
used to forecast or report physical neutrino constraints.
"""

from __future__ import annotations

import numpy as np

from nufield.parameters import CosmologyParameters


def toy_matter_power(k_h_mpc: np.ndarray, params: CosmologyParameters) -> np.ndarray:
    """Return a positive, toy matter-spectrum-like curve on `k` in h/Mpc.

    All constants are engineering choices for testability; no precision is claimed. The large-scale
    power-law-like shape and damping factor mimic broad qualitative behavior, not full cosmological
    transfer physics. In particular, the neutrino suppression amplitude is deliberately tunable.
    """
    k = np.asarray(k_h_mpc, dtype=np.float64)
    safe_k = np.maximum(k, 1e-5)
    turnover = 0.18 * (params.omega_m / 0.315) * (params.h / 0.674)
    shape = safe_k**params.n_s / (1.0 + (safe_k / turnover) ** 2.0) ** 2.0
    high_k_damping = np.exp(-0.015 * safe_k**2)
    # Arbitrary toy response. It is not derived from neutrino transfer functions.
    scale_response = safe_k**2 / (safe_k**2 + 0.22**2)
    toy_neutrino_suppression = np.exp(-1.3 * params.sum_mnu_ev * scale_response)
    amplitude = (params.sigma8 / 0.811) ** 2
    result = amplitude * shape * high_k_damping * toy_neutrino_suppression
    return np.where(k > 0, np.maximum(result, 1e-14), 0.0)


def wave_number_grid(grid_size: int, box_size_mpc_h: float) -> np.ndarray:
    """Construct the magnitude of the periodic FFT wavevector grid, units h/Mpc."""
    freq = np.fft.fftfreq(grid_size, d=box_size_mpc_h / grid_size) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    return np.sqrt(kx * kx + ky * ky + kz * kz)
