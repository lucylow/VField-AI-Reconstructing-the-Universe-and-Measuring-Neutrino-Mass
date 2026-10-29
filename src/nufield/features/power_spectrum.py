"""Simple FFT-based radial power spectrum for periodic toy fields."""

from __future__ import annotations

import numpy as np


def radial_power_spectrum(
    field: np.ndarray,
    box_size_mpc_h: float,
    n_bins: int = 10,
    k_min: float | None = None,
    k_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return geometric bin centers, mean |delta_k|^2, and mode counts.

    Normalization is chosen for stable feature extraction from this toy code; it is not yet matched to
    a survey analysis convention. Validate against a reference implementation before scientific use.
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 3 or len(set(arr.shape)) != 1:
        raise ValueError("field must be a cubic 3-D array")
    if not np.isfinite(arr).all() or box_size_mpc_h <= 0 or n_bins < 2:
        raise ValueError("invalid field, box size, or n_bins")
    n = arr.shape[0]
    arr = arr - arr.mean()
    delta_k = np.fft.fftn(arr)
    pk = (np.abs(delta_k) ** 2) / (n**6)
    freq = np.fft.fftfreq(n, d=box_size_mpc_h / n) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    fundamental = 2 * np.pi / box_size_mpc_h
    k_nyquist = np.pi * n / box_size_mpc_h
    lo = max(k_min or fundamental * 0.8, 1e-9)
    hi = min(k_max or k_nyquist, k_nyquist)
    if hi <= lo:
        raise ValueError("k_max must be larger than k_min")
    edges = np.geomspace(lo, hi, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    values = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask = (kmag >= edges[i]) & (kmag < edges[i + 1])
        counts[i] = int(mask.sum())
        if counts[i]:
            values[i] = float(pk[mask].mean())
    # Empty bins are kept explicit internally, but feature extraction will reject them.
    return centers, values, counts
