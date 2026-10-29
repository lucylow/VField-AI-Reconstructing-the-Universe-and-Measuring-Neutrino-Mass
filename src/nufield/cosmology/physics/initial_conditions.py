"""Gaussian initial density-field realization from a physical-unit matter P(k).

This generator is a linear Gaussian-field utility. It does not evolve gravity or include neutrino
phase-space perturbations; it is suitable for validating Fourier normalization and controlled linear
inference benchmarks, not as a replacement for Quijote/CLASS/CAMB + N-body initial conditions.
"""

from __future__ import annotations

import numpy as np

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


def wave_number_magnitude_grid(grid_size: int, box_size_mpc: float) -> np.ndarray:
    if grid_size < 4 or box_size_mpc <= 0:
        raise ValueError("grid_size must be >=4 and box_size_mpc positive")
    freq = 2 * np.pi * np.fft.fftfreq(grid_size, d=box_size_mpc / grid_size)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    return np.sqrt(kx * kx + ky * ky + kz * kz)


def generate_gaussian_density_field(
    spectrum: MatterPowerSpectrum,
    grid_size: int,
    box_size_mpc: float,
    *,
    seed: int,
    strict_k_support: bool = True,
) -> np.ndarray:
    r"""Draw a real Gaussian field with E[P_hat(k)]≈P(k) under the package Fourier convention.

    For NumPy's unnormalized FFT, multiply FFT(white-noise) by sqrt(Ncell P(k)/V). The zero mode is
    set to zero. ``strict_k_support`` rejects spectra that do not cover the full lattice range.
    """
    if grid_size < 4 or box_size_mpc <= 0:
        raise ValueError("grid_size must be >=4 and box_size_mpc positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    kg = wave_number_magnitude_grid(grid_size, box_size_mpc)
    nonzero = kg > 0
    kmax, kmin = float(kg[nonzero].max()), float(kg[nonzero].min())
    if strict_k_support and (kmin < spectrum.k_mpc_inv[0] or kmax > spectrum.k_mpc_inv[-1]):
        raise ValueError(
            f"power-spectrum support [{spectrum.k_mpc_inv[0]:.4g}, {spectrum.k_mpc_inv[-1]:.4g}] "
            f"1/Mpc does not cover field lattice [{kmin:.4g}, {kmax:.4g}] 1/Mpc"
        )
    pgrid = np.zeros_like(kg)
    if strict_k_support:
        pgrid[nonzero] = spectrum.evaluate(kg[nonzero])
    else:
        inside = nonzero & (kg >= spectrum.k_mpc_inv[0]) & (kg <= spectrum.k_mpc_inv[-1])
        pgrid[inside] = spectrum.evaluate(kg[inside])
        # Unsupported modes are set to zero, not extrapolated. Record this limitation in provenance.
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(grid_size, grid_size, grid_size))
    fk = np.fft.fftn(white)
    ncell = grid_size**3
    volume = box_size_mpc**3
    fk *= np.sqrt(np.maximum(pgrid, 0.0) * ncell / volume)
    fk[0, 0, 0] = 0.0
    field = np.fft.ifftn(fk).real
    field -= field.mean()
    if not np.all(np.isfinite(field)):
        raise FloatingPointError("generated density field contains non-finite values")
    return field.astype(np.float32)


__all__ = ["wave_number_magnitude_grid", "generate_gaussian_density_field"]
