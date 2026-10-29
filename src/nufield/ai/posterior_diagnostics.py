"""Sample-based diagnostics for approximate posterior samples and ensemble predictions."""

from __future__ import annotations

from typing import Any

import numpy as np


def effective_sample_size_1d(samples: Any, *, max_lag: int | None = None) -> float:
    """Estimate ESS with an initial-positive-sequence autocorrelation truncation.

    This is a simple diagnostic for approximately stationary Markov-chain draws. It is not a
    substitute for checking trace plots, multiple chains and parameter transformations.
    """
    x = np.asarray(samples, float).reshape(-1)
    n = len(x)
    if n < 4 or not np.isfinite(x).all():
        raise ValueError("ESS needs at least four finite samples")
    x = x - x.mean()
    variance = float(np.dot(x, x) / n)
    if variance <= 1e-15:
        return float(n)
    limit = min(n - 1, n // 2 if max_lag is None else int(max_lag))
    if limit < 1:
        raise ValueError("max_lag must be positive")
    correlations = []
    for lag in range(1, limit + 1):
        rho = float(np.dot(x[:-lag], x[lag:]) / ((n - lag) * variance))
        correlations.append(rho)
    # Geyer's initial positive sequence: stop at the first nonpositive pair sum.
    pair_sum = 0.0
    for i in range(0, len(correlations) - 1, 2):
        value = correlations[i] + correlations[i + 1]
        if value <= 0:
            break
        pair_sum += value
    tau = max(1.0, 1.0 + 2.0 * pair_sum)
    return float(min(n, n / tau))


def split_rhat(chains: Any) -> np.ndarray:
    """Compute split R-hat for array [n_chains,n_draws,n_parameters]."""
    x = np.asarray(chains, float)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError("chains must be finite [chains>=2, draws>=8, parameters]")
    n_half = x.shape[1] // 2
    split = np.concatenate([x[:, :n_half], x[:, -n_half:]], axis=0)
    n = split.shape[1]
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    within = variances.mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    var_hat = (n - 1) / n * within + between / n
    return np.sqrt(np.maximum(var_hat, 0) / np.maximum(within, 1e-15))


def chains_diagnostic_report(chains: Any) -> dict[str, Any]:
    """Report split-Rhat and per-chain/per-parameter ESS summaries."""
    x = np.asarray(chains, float)
    rhat = split_rhat(x)
    ess = np.empty((x.shape[0], x.shape[2]), float)
    for c in range(x.shape[0]):
        for p in range(x.shape[2]):
            ess[c, p] = effective_sample_size_1d(x[c, :, p])
    return {
        "split_rhat": rhat.tolist(),
        "ess_by_chain_parameter": ess.tolist(),
        "minimum_ess": float(ess.min()),
        "maximum_rhat": float(rhat.max()),
        "rhat_threshold_pass": bool(np.all(rhat < 1.01)),
        "warning": "Good R-hat/ESS does not establish posterior correctness or SBC calibration.",
    }


def covariance_condition_report(samples: Any, *, max_condition: float = 1e10) -> dict[str, Any]:
    """Conditioning of a sample covariance matrix, useful before Gaussian likelihood evaluation."""
    x = np.asarray(samples, float)
    if x.ndim != 2 or len(x) <= x.shape[1] or not np.isfinite(x).all():
        raise ValueError("samples must be finite [N,P] with N>P")
    cov = np.cov(x, rowvar=False, ddof=1)
    eig = np.linalg.eigvalsh(np.atleast_2d(cov))
    positive = eig[eig > 0]
    condition = float(
        np.inf
        if len(positive) != len(eig) or positive.size == 0
        else positive.max() / positive.min()
    )
    return {
        "eigenvalues": eig.tolist(),
        "condition_number": condition,
        "positive_definite": bool(np.all(eig > 0)),
        "condition_threshold_pass": bool(condition <= max_condition),
        "max_condition_threshold": float(max_condition),
    }
