"""Importance weights with explicit prior/proposal density correction."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def normalized_importance_weights(
    log_prior: np.ndarray, log_proposal: np.ndarray, log_likelihood: np.ndarray
) -> np.ndarray:
    """Compute normalized weights proportional to prior * likelihood / proposal."""
    lp, lq, ll = map(lambda x: np.asarray(x, float), (log_prior, log_proposal, log_likelihood))
    if lp.ndim != 1 or lp.shape != lq.shape or lp.shape != ll.shape or len(lp) == 0:
        raise ValueError("log densities must be same-shaped non-empty vectors")
    if np.isnan(lp).any() or np.isnan(lq).any() or np.isnan(ll).any():
        raise ValueError("log densities cannot contain NaN")
    logw = lp + ll - lq
    norm = logsumexp(logw)
    if not np.isfinite(norm):
        raise ValueError("importance weights cannot be normalized")
    return np.exp(logw - norm)


def importance_diagnostics(weights: np.ndarray) -> dict[str, float]:
    w = np.asarray(weights, float)
    if w.ndim != 1 or len(w) == 0 or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid weights")
    w = w / w.sum()
    positive = w[w > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return {
        "effective_sample_size": float(1 / np.sum(w * w)),
        "max_weight": float(w.max()),
        "weight_entropy": entropy,
        "perplexity": float(np.exp(entropy)),
        "n_particles": int(len(w)),
    }
