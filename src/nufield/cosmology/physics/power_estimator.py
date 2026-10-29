"""Periodic-box power-spectrum estimator with explicit Fourier-volume normalization."""

from __future__ import annotations

import numpy as np


def radial_power_spectrum_periodic(
    overdensity: np.ndarray,
    box_size_mpc: float,
    *,
    n_bins: int = 24,
    k_edges_mpc_inv: np.ndarray | None = None,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Estimate P(k) for a cubic periodic overdensity grid.

    Convention: delta_k = V/Ncell * FFT(delta), P_hat=|delta_k|²/V. Since NumPy's forward
    FFT is an unnormalized sum, the shell estimator below is P_hat=V |FFT(delta)|²/Ncell².
    Returned units are k [Mpc^-1], P [Mpc³], mode counts, and shell standard errors based on the
    Gaussian-mode approximation. The estimator does not correct mass-assignment windows or aliasing.
    """
    field = np.asarray(overdensity, dtype=float)
    if field.ndim != 3 or len(set(field.shape)) != 1:
        raise ValueError("overdensity must be a cubic 3-D grid")
    if field.shape[0] < 4 or np.any(~np.isfinite(field)):
        raise ValueError("grid must have at least four cells per dimension and finite values")
    if box_size_mpc <= 0 or not np.isfinite(box_size_mpc):
        raise ValueError("box_size_mpc must be finite and positive")
    if subtract_mean:
        field = field - field.mean()
    n = field.shape[0]
    ncell = n**3
    volume = box_size_mpc**3
    fk = np.fft.fftn(field)
    pgrid = volume * np.abs(fk) ** 2 / ncell**2
    freq = 2 * np.pi * np.fft.fftfreq(n, d=box_size_mpc / n)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    fundamental = 2 * np.pi / box_size_mpc
    knyquist = np.pi * n / box_size_mpc
    if k_edges_mpc_inv is None:
        if n_bins < 2:
            raise ValueError("n_bins must be >=2")
        edges = np.linspace(fundamental * 0.8, knyquist, n_bins + 1)
    else:
        edges = np.asarray(k_edges_mpc_inv, dtype=float)
        if edges.ndim != 1 or len(edges) < 3 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
            raise ValueError("k_edges_mpc_inv must be a strictly increasing 1-D array")
        if edges[-1] > np.sqrt(3) * knyquist * (1 + 1e-8):
            raise ValueError("k bin extends beyond the maximum corner mode")
    centers = 0.5 * (edges[:-1] + edges[1:])
    pk = np.full(centers.shape, np.nan, dtype=float)
    counts = np.zeros(centers.shape, dtype=np.int64)
    stderr = np.full(centers.shape, np.nan, dtype=float)
    for index in range(len(centers)):
        mask = (kmag >= edges[index]) & (kmag < edges[index + 1])
        # Exclude zero mode unconditionally; the mean density is not a finite-k fluctuation.
        mask &= kmag > 0
        vals = pgrid[mask]
        counts[index] = vals.size
        if vals.size:
            pk[index] = float(np.mean(vals))
            stderr[index] = pk[index] * np.sqrt(2.0 / vals.size)
    return centers, pk, counts, stderr


def cic_fourier_window(
    k_components: tuple[np.ndarray, np.ndarray, np.ndarray], cell_size_mpc: float
) -> np.ndarray:
    """Return CIC density-assignment amplitude window product_j sinc(k_j Δx/2)^2."""
    if cell_size_mpc <= 0:
        raise ValueError("cell_size_mpc must be positive")
    result = np.ones_like(np.asarray(k_components[0], dtype=float))
    for component in k_components:
        x = np.asarray(component, dtype=float) * cell_size_mpc / (2 * np.pi)
        result *= np.sinc(x) ** 2
    return result


def deconvolve_assignment_power(
    pk: np.ndarray,
    window_amplitude: np.ndarray,
    *,
    minimum_window: float = 0.2,
) -> np.ndarray:
    """Correct power for a known amplitude window where the window is not too small.

    Power scales as |W|², hence P_true=P_measured/|W|². Values near Nyquist are unstable and should
    normally be masked rather than aggressively deconvolved.
    """
    p, w = np.asarray(pk, dtype=float), np.asarray(window_amplitude, dtype=float)
    if p.shape != w.shape or np.any(p < 0) or np.any(~np.isfinite(p)):
        raise ValueError("pk and window must match; power must be finite and non-negative")
    if not 0 < minimum_window <= 1:
        raise ValueError("minimum_window must be in (0, 1]")
    if np.any(np.abs(w) > 1 + 1e-8) or np.any(~np.isfinite(w)):
        raise ValueError("window amplitude must be finite with |W|<=1")
    result = np.full_like(p, np.nan, dtype=float)
    valid = np.abs(w) >= minimum_window
    result[valid] = p[valid] / w[valid] ** 2
    return result


__all__ = ["radial_power_spectrum_periodic", "cic_fourier_window", "deconvolve_assignment_power"]
