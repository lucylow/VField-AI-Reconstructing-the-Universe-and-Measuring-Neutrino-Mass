"""Gaussian covariance approximations for periodic-box power-spectrum bins."""

from __future__ import annotations

import numpy as np


def gaussian_power_covariance(
    pk: np.ndarray,
    mode_counts: np.ndarray,
    *,
    shot_noise: np.ndarray | float = 0.0,
    cross_spectrum: bool = False,
) -> np.ndarray:
    r"""Return diagonal Gaussian covariance Var[P_i]≈2(P_i+N_i)²/N_modes,i.

    This is a periodic, Gaussian-mode approximation. It omits connected trispectrum covariance,
    super-sample covariance, survey-window coupling and estimated-covariance uncertainty.
    """
    p, n = np.asarray(pk, float), np.asarray(mode_counts, float)
    noise = np.asarray(shot_noise, float)
    if (
        p.ndim != 1
        or n.shape != p.shape
        or np.any(p < 0)
        or np.any(n <= 0)
        or np.any(~np.isfinite(p + n))
    ):
        raise ValueError("pk and positive mode_counts must be matching finite 1-D arrays")
    if np.any(noise < 0) or np.any(~np.isfinite(noise)):
        raise ValueError("shot_noise must be finite and non-negative")
    total = p + noise
    factor = 1.0 if cross_spectrum else 2.0
    return np.diag(factor * total**2 / n)


def gaussian_cross_power_variance(
    p_aa: np.ndarray,
    p_bb: np.ndarray,
    p_ab: np.ndarray,
    mode_counts: np.ndarray,
    *,
    noise_a: np.ndarray | float = 0.0,
    noise_b: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Gaussian variance of a cross-power estimator: [(Pab)^2+(Paa+Na)(Pbb+Nb)]/Nmodes."""
    aa, bb, ab, nn = map(lambda x: np.asarray(x, float), (p_aa, p_bb, p_ab, mode_counts))
    na, nb = np.asarray(noise_a, float), np.asarray(noise_b, float)
    if aa.shape != bb.shape or aa.shape != ab.shape or aa.shape != nn.shape:
        raise ValueError("all spectra and mode_counts must have matching shapes")
    if np.any(nn <= 0) or np.any(aa < 0) or np.any(bb < 0) or np.any(na < 0) or np.any(nb < 0):
        raise ValueError("auto powers and noises must be non-negative and mode counts positive")
    return (ab**2 + (aa + na) * (bb + nb)) / nn


def diagonal_covariance_condition(covariance: np.ndarray) -> dict[str, float | bool]:
    """Summarize eigenvalue conditioning and whether a covariance is positive definite."""
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("covariance must be square and symmetric")
    eig = np.linalg.eigvalsh(c)
    pos = bool(eig.min() > 0)
    cond = float(eig.max() / eig.min()) if pos else float("inf")
    return {
        "positive_definite": pos,
        "min_eigenvalue": float(eig.min()),
        "max_eigenvalue": float(eig.max()),
        "condition_number": cond,
    }


__all__ = [
    "gaussian_power_covariance",
    "gaussian_cross_power_variance",
    "diagonal_covariance_condition",
]
