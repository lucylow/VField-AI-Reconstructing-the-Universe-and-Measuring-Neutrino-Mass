"""Periodic-box power spectrum estimator with documented Fourier normalization.

This estimator is for cubic, uniform, periodic simulation-box density contrast fields. It is not a
DESI estimator: survey masks, random catalogs, angular/radial selection, weights, fiber assignment,
window convolution and the relevant covariance are not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeriodicPowerSpectrum:
    k: np.ndarray
    power: np.ndarray
    mode_count: np.ndarray
    box_size: float
    grid_size: int
    shot_noise: float | None
    normalization: str
    mass_assignment: str

    def to_dict(self):
        return {
            "k": self.k.tolist(),
            "power": self.power.tolist(),
            "mode_count": self.mode_count.tolist(),
            "box_size": self.box_size,
            "grid_size": self.grid_size,
            "shot_noise": self.shot_noise,
            "normalization": self.normalization,
            "mass_assignment": self.mass_assignment,
        }


def estimate_periodic_power(
    field: np.ndarray,
    *,
    box_size: float,
    n_bins: int = 24,
    k_min: float | None = None,
    k_max: float | None = None,
    shot_noise: float | None = None,
    subtract_shot_noise: bool = False,
    assignment: str = "none",
    deconvolve_assignment: bool = False,
    max_window_correction: float = 5.0,
) -> PeriodicPowerSpectrum:
    """Estimate isotropic P(k) in volume units, k in inverse box-length units.

    If `assignment='cic'` and deconvolution is enabled, divide modes by the squared CIC window
    where the inverse correction is bounded by `max_window_correction`. The cutoff is a numerical
    safety, not a substitute for a resolution-convergence study. Shot noise is optionally subtracted
    as a declared constant; it should only be supplied when it matches the tracer/weight convention.
    """
    x = np.asarray(field, dtype=np.float64)
    if x.ndim != 3 or len(set(x.shape)) != 1:
        raise ValueError("field must be cubic [N,N,N]")
    if not np.isfinite(x).all():
        raise ValueError("field must be finite")
    if not np.isfinite(box_size) or box_size <= 0 or n_bins < 2:
        raise ValueError("invalid box_size or n_bins")
    if assignment not in {"none", "cic"}:
        raise ValueError("assignment must be none or cic")
    if max_window_correction < 1:
        raise ValueError("max_window_correction must be >=1")
    if shot_noise is not None and (not np.isfinite(shot_noise) or shot_noise < 0):
        raise ValueError("shot_noise must be nonnegative and finite")
    if subtract_shot_noise and shot_noise is None:
        raise ValueError("shot_noise value is required to subtract shot noise")
    n = x.shape[0]
    dx = box_size / n
    volume = box_size**3
    delta = x - x.mean()
    fk = np.fft.rfftn(delta)
    # With NumPy's unnormalized FFT, P(k)=V/N^6 |FFT(delta)|^2 for a cubic N^3 grid.
    power = (volume / n**6) * (fk.real**2 + fk.imag**2)
    freq = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    freq_z = 2 * np.pi * np.fft.rfftfreq(n, d=dx)
    kx, ky, kz = np.meshgrid(freq, freq, freq_z, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    if assignment == "cic" and deconvolve_assignment:
        # Window for CIC amplitude is prod(sinc(k_i dx/(2pi))^2); power window is its square.
        wx = np.sinc(kx * dx / (2 * np.pi)) ** 2
        wy = np.sinc(ky * dx / (2 * np.pi)) ** 2
        wz = np.sinc(kz * dx / (2 * np.pi)) ** 2
        window_power = (wx * wy * wz) ** 2
        inv = np.divide(
            1.0, window_power, out=np.full_like(window_power, np.inf), where=window_power > 0
        )
        allowed = inv <= max_window_correction
        power = np.where(allowed, power * inv, np.nan)
    fundamental = 2 * np.pi / box_size
    nyquist = np.pi * n / box_size
    low = max(float(k_min) if k_min is not None else fundamental * 0.8, 1e-12)
    high = min(float(k_max) if k_max is not None else nyquist, nyquist)
    if low >= high:
        raise ValueError("k_min must be less than k_max and below Nyquist")
    edges = np.geomspace(low, high, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pk = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask = (kmag >= edges[i]) & (kmag < edges[i + 1]) & np.isfinite(power)
        # rFFT stores only nonnegative kz; nonzero kz modes represent both signs and need multiplicity 2.
        multiplicity = np.full_like(kmag, 2.0, dtype=float)
        multiplicity[:, :, 0] = 1.0
        if n % 2 == 0:
            multiplicity[:, :, -1] = 1.0
        weights = multiplicity[mask]
        counts[i] = int(weights.sum())
        if counts[i]:
            pk[i] = float(np.sum(power[mask] * weights) / weights.sum())
    if subtract_shot_noise:
        pk = np.where(np.isfinite(pk), pk - float(shot_noise), np.nan)
    return PeriodicPowerSpectrum(
        centers,
        pk,
        counts,
        float(box_size),
        int(n),
        shot_noise,
        "P(k)=V/N^6 |FFT(delta)|^2; k in inverse box_size units",
        "cic_deconvolved" if assignment == "cic" and deconvolve_assignment else assignment,
    )
