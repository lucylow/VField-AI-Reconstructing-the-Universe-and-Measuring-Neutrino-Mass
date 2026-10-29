"""Power-spectrum comparison metrics with explicit scale cuts."""

from __future__ import annotations

import numpy as np


def compare_spectra(
    k: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    k_min: float | None = None,
    k_max: float | None = None,
) -> dict[str, float | int]:
    kk, ref, pred = map(lambda x: np.asarray(x, float), (k, reference, prediction))
    if kk.ndim != 1 or ref.shape != kk.shape or pred.shape != kk.shape or not np.isfinite(kk).all():
        raise ValueError("k, reference, prediction must be same-shaped vectors")
    if np.any(ref <= 0) or np.any(pred <= 0):
        raise ValueError("spectra must be positive for relative/log metrics")
    mask = np.ones_like(kk, dtype=bool)
    if k_min is not None:
        mask &= kk >= k_min
    if k_max is not None:
        mask &= kk <= k_max
    if mask.sum() < 2:
        raise ValueError("scale cuts retain fewer than two bins")
    frac = pred[mask] / ref[mask] - 1
    logdiff = np.log(pred[mask]) - np.log(ref[mask])
    return {
        "n_bins": int(mask.sum()),
        "fractional_bias_mean": float(frac.mean()),
        "fractional_bias_rms": float(np.sqrt(np.mean(frac**2))),
        "fractional_bias_max_abs": float(np.max(np.abs(frac))),
        "log_spectrum_rmse": float(np.sqrt(np.mean(logdiff**2))),
        "k_min_used": float(kk[mask].min()),
        "k_max_used": float(kk[mask].max()),
    }


def fractional_response(
    reference: np.ndarray, varied: np.ndarray, floor: float = 1e-12
) -> np.ndarray:
    a, b = np.asarray(reference, float), np.asarray(varied, float)
    if a.shape != b.shape or np.any(a < 0) or np.any(b < 0) or floor <= 0:
        raise ValueError("spectra must be same-shaped and non-negative")
    return (b - a) / np.maximum(a, floor)
