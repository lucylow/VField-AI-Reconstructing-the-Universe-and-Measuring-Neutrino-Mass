"""Explicit observational-noise templates used in synthetic survey tests."""

from __future__ import annotations

import numpy as np


def poisson_noise_power(number_density_mpc3: float) -> float:
    if number_density_mpc3 <= 0 or not np.isfinite(number_density_mpc3):
        raise ValueError("number_density_mpc3 must be finite and positive")
    return 1.0 / number_density_mpc3


def shear_noise_power(sigma_e: float, n_eff_arcmin2: float) -> float:
    """Shape-noise N_l=sigma_e²/(2n_eff), with n_eff given per square arcminute."""
    if sigma_e < 0 or n_eff_arcmin2 <= 0 or not np.isfinite(sigma_e + n_eff_arcmin2):
        raise ValueError("sigma_e must be non-negative and n_eff_arcmin2 positive")
    arcmin2_per_sr = (180 * 60 / np.pi) ** 2
    return float(sigma_e**2 / (2 * n_eff_arcmin2 * arcmin2_per_sr))


def add_white_noise(signal: np.ndarray, noise_std: float, *, seed: int) -> np.ndarray:
    """Add independent zero-mean Gaussian noise for controlled robustness experiments."""
    x = np.asarray(signal, float)
    if np.any(~np.isfinite(x)) or noise_std < 0 or not np.isfinite(noise_std) or seed < 0:
        raise ValueError("signal finite, noise_std>=0 and seed>=0 required")
    return x + np.random.default_rng(seed).normal(0, noise_std, size=x.shape)


def shape_noise_map(shape: tuple[int, int], sigma_e: float, *, seed: int) -> np.ndarray:
    """Generate a Gaussian shape-noise map; this is not a realistic ellipticity simulator."""
    if len(shape) != 2 or min(shape) < 1 or sigma_e < 0 or seed < 0:
        raise ValueError("shape must be 2-D positive, sigma_e>=0 and seed>=0")
    return (
        np.random.default_rng(seed).normal(0, sigma_e / np.sqrt(2), size=shape).astype(np.float32)
    )


__all__ = ["poisson_noise_power", "shear_noise_power", "add_white_noise", "shape_noise_map"]
