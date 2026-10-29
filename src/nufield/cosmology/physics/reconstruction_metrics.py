"""Field-level reconstruction metrics based on Fourier-space cross statistics.

Correlation alone does not measure amplitude calibration. These diagnostics separately expose transfer,
propagator and stochastic residual power. They are meaningful only when fields are aligned, use the same
box/grid/units, and refer to the same realization or a justified matched ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    k: np.ndarray
    k_count: np.ndarray
    p_truth: np.ndarray
    p_reconstruction: np.ndarray
    p_cross: np.ndarray
    cross_correlation: np.ndarray
    propagator: np.ndarray
    stochastic_power: np.ndarray
    transfer_ratio: np.ndarray


def _validate_fields(truth: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(truth, float), np.asarray(reconstructed, float)
    if a.ndim != 3 or a.shape != b.shape or len(set(a.shape)) != 1 or a.shape[0] < 4:
        raise ValueError(
            "truth and reconstructed fields must be matching cubic 3-D arrays of size >=4"
        )
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
        raise ValueError("fields must be finite")
    return a - a.mean(), b - b.mean()


def reconstruction_metrics_periodic(
    truth: np.ndarray,
    reconstructed: np.ndarray,
    box_size_mpc: float,
    *,
    n_bins: int = 24,
    k_edges_mpc_inv: np.ndarray | None = None,
) -> ReconstructionMetrics:
    r"""Measure shell-averaged Ptt, Prr, Prt and their derived field-level diagnostics.

    Fourier convention is delta(k)=V/Ncell FFT(delta), with P=|delta(k)|²/V. Propagator G=Prt/Ptt,
    transfer ratio is Prr/Ptt, cross-correlation r=Prt/sqrt(Prr Ptt), and stochastic residual is
    Prr-Prt²/Ptt. Numerical negative residuals from round-off are retained rather than hidden.
    """
    t, r = _validate_fields(truth, reconstructed)
    if not np.isfinite(box_size_mpc) or box_size_mpc <= 0:
        raise ValueError("box_size_mpc must be finite and positive")
    n = t.shape[0]
    volume = box_size_mpc**3
    norm = volume / (n**3) ** 2
    ft, fr = np.fft.fftn(t), np.fft.fftn(r)
    ptt_grid = norm * np.abs(ft) ** 2
    prr_grid = norm * np.abs(fr) ** 2
    prt_grid = norm * np.real(fr * np.conjugate(ft))
    freq = 2 * np.pi * np.fft.fftfreq(n, d=box_size_mpc / n)
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    if k_edges_mpc_inv is None:
        if n_bins < 2:
            raise ValueError("n_bins must be >=2")
        edges = np.linspace(0.8 * 2 * np.pi / box_size_mpc, np.pi * n / box_size_mpc, n_bins + 1)
    else:
        edges = np.asarray(k_edges_mpc_inv, float)
        if edges.ndim != 1 or edges.size < 3 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
            raise ValueError("k_edges_mpc_inv must be increasing and non-negative")
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(centers.size, int)
    ptt = np.full(centers.size, np.nan)
    prr = np.full_like(ptt, np.nan)
    prt = np.full_like(ptt, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (kmag >= lo) & (kmag < hi) & (kmag > 0)
        counts[i] = int(mask.sum())
        if counts[i]:
            ptt[i] = float(ptt_grid[mask].mean())
            prr[i] = float(prr_grid[mask].mean())
            prt[i] = float(prt_grid[mask].mean())
    corr = np.full_like(ptt, np.nan)
    propagator = np.full_like(ptt, np.nan)
    residual = np.full_like(ptt, np.nan)
    ratio = np.full_like(ptt, np.nan)
    valid = (counts > 0) & (ptt > 0) & (prr >= 0) & np.isfinite(ptt + prr + prt)
    corr[valid] = prt[valid] / np.sqrt(prr[valid] * ptt[valid])
    propagator[valid] = prt[valid] / ptt[valid]
    residual[valid] = prr[valid] - prt[valid] ** 2 / ptt[valid]
    ratio[valid] = prr[valid] / ptt[valid]
    return ReconstructionMetrics(centers, counts, ptt, prr, prt, corr, propagator, residual, ratio)


def aggregate_reconstruction_score(
    metrics: ReconstructionMetrics, *, k_min: float, k_max: float, metric: str = "cross_correlation"
) -> float:
    """Return the finite-bin mean of a named diagnostic over a stated k interval."""
    if k_min < 0 or k_max <= k_min:
        raise ValueError("require 0 <= k_min < k_max")
    allowed = {"cross_correlation", "propagator", "transfer_ratio", "stochastic_power"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}")
    values = np.asarray(getattr(metrics, metric), float)
    mask = (metrics.k >= k_min) & (metrics.k <= k_max) & (metrics.k_count > 0) & np.isfinite(values)
    if not mask.any():
        raise ValueError("no valid modes in requested k interval")
    return float(np.mean(values[mask]))


__all__ = [
    "ReconstructionMetrics",
    "reconstruction_metrics_periodic",
    "aggregate_reconstruction_score",
]
