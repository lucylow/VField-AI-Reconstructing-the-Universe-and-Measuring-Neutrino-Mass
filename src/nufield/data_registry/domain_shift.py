"""Diagnostics for dataset shift between training and evaluation domains."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist


def standardized_mean_difference(
    train: np.ndarray, evaluation: np.ndarray, *, eps: float = 1e-12
) -> np.ndarray:
    """Per-feature mean shift in pooled standard-deviation units."""
    a, b = _matrices(train, evaluation)
    va, vb = np.var(a, axis=0, ddof=1), np.var(b, axis=0, ddof=1)
    pooled = np.sqrt((va + vb) / 2.0)
    return (b.mean(axis=0) - a.mean(axis=0)) / np.maximum(pooled, eps)


def _matrices(train, evaluation):
    a = np.asarray(train, dtype=float)
    b = np.asarray(evaluation, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("domain features must be aligned [N,F] matrices")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each domain needs at least two rows")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("domain features must be finite")
    return a, b


def maximum_mean_discrepancy(
    train: np.ndarray,
    evaluation: np.ndarray,
    *,
    kernel_bandwidth: float | None = None,
    max_rows: int = 1500,
    seed: int = 42,
) -> dict[str, Any]:
    """Biased RBF-kernel MMD² estimate with deterministic row subsampling for large arrays.

    Features must be transformed using training-only statistics before this function is called.
    This diagnostic is a shift indicator, not a formal acceptance test or p-value.
    """
    a, b = _matrices(train, evaluation)
    rng = np.random.default_rng(seed)
    if len(a) > max_rows:
        a = a[rng.choice(len(a), max_rows, replace=False)]
    if len(b) > max_rows:
        b = b[rng.choice(len(b), max_rows, replace=False)]
    aa = cdist(a, a, "sqeuclidean")
    bb = cdist(b, b, "sqeuclidean")
    ab = cdist(a, b, "sqeuclidean")
    if kernel_bandwidth is None:
        sample = np.concatenate([a, b], axis=0)
        if len(sample) > 500:
            sample = sample[rng.choice(len(sample), 500, replace=False)]
        d = cdist(sample, sample, "sqeuclidean")
        positive = d[d > 0]
        if not len(positive):
            bw = 1.0
        else:
            bw = float(np.sqrt(np.median(positive)))
    else:
        bw = float(kernel_bandwidth)
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError("kernel bandwidth must be finite and positive")
    den = 2 * bw * bw
    kaa = np.exp(-aa / den)
    kbb = np.exp(-bb / den)
    kab = np.exp(-ab / den)
    mmd2 = float(kaa.mean() + kbb.mean() - 2 * kab.mean())
    return {
        "biased_mmd2": max(0.0, mmd2),
        "raw_biased_mmd2": mmd2,
        "kernel": "RBF",
        "kernel_bandwidth": bw,
        "n_train_used": len(a),
        "n_evaluation_used": len(b),
        "seed": seed,
        "warning": "MMD is a descriptive diagnostic; select thresholds and null distributions before interpreting significance.",
    }


def domain_shift_report(
    train: np.ndarray,
    evaluation: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    max_rows: int = 1500,
    seed: int = 42,
) -> dict[str, Any]:
    a, b = _matrices(train, evaluation)
    smd = standardized_mean_difference(a, b)
    mmd = maximum_mean_discrepancy(a, b, max_rows=max_rows, seed=seed)
    names = feature_names or [f"feature_{i}" for i in range(a.shape[1])]
    if len(names) != a.shape[1]:
        raise ValueError("feature_names length mismatch")
    ranked = sorted(
        (
            {"feature": names[i], "smd": float(smd[i]), "abs_smd": float(abs(smd[i]))}
            for i in range(len(names))
        ),
        key=lambda r: r["abs_smd"],
        reverse=True,
    )
    return {
        "n_train": len(a),
        "n_evaluation": len(b),
        "n_features": a.shape[1],
        "mean_abs_smd": float(np.mean(np.abs(smd))),
        "max_abs_smd": float(np.max(np.abs(smd))),
        "features_sorted_by_abs_smd": ranked[: min(25, len(ranked))],
        "mmd": mmd,
        "warning": "A domain-shift metric does not by itself establish a model's performance degradation. Evaluate held-out predictive/posterior metrics in parallel.",
    }
