"""Input-drift monitoring for incoming observations and simulated evaluation domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import ks_2samp


def population_stability_index(
    reference: Any, current: Any, *, n_bins: int = 10, eps: float = 1e-6
) -> np.ndarray:
    """Compute PSI feature-wise using reference quantile bins.

    PSI thresholds are context-dependent. This function returns scores and does not label a dataset
    as physically invalid or acceptable.
    """
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    if (
        ref.ndim != 2
        or cur.ndim != 2
        or ref.shape[1] != cur.shape[1]
        or len(ref) < 10
        or len(cur) < 10
    ):
        raise ValueError("reference/current must be [N>=10,F] matrices with same width")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all() or n_bins < 2 or eps <= 0:
        raise ValueError("features must be finite and n_bins/eps valid")
    scores = []
    for j in range(ref.shape[1]):
        edges = np.unique(np.quantile(ref[:, j], np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            scores.append(0.0 if np.allclose(ref[:, j], cur[:, j]) else 1e6)
            continue
        p = np.histogram(ref[:, j], bins=edges)[0].astype(float) / len(ref)
        q = np.histogram(cur[:, j], bins=edges)[0].astype(float) / len(cur)
        p = np.maximum(p, eps)
        q = np.maximum(q, eps)
        p /= p.sum()
        q /= q.sum()
        scores.append(float(np.sum((q - p) * np.log(q / p))))
    return np.asarray(scores)


def feature_drift_report(
    reference: Any,
    current: Any,
    *,
    feature_names: list[str] | None = None,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.01,
) -> dict[str, Any]:
    """Report per-feature distribution differences with both PSI and KS statistics."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
        raise ValueError("reference/current must be [N,F] matrices with same feature width")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("features must be finite")
    names = feature_names or [f"feature_{i}" for i in range(ref.shape[1])]
    if len(names) != ref.shape[1]:
        raise ValueError("feature_names must match feature count")
    psi = population_stability_index(ref, cur)
    rows = []
    for j, name in enumerate(names):
        ks = ks_2samp(ref[:, j], cur[:, j])
        rows.append(
            {
                "feature": name,
                "psi": float(psi[j]),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "psi_flag": bool(psi[j] >= psi_threshold),
                "ks_flag": bool(ks.pvalue < ks_alpha),
            }
        )
    return {
        "features": rows,
        "psi_threshold": float(psi_threshold),
        "ks_alpha": float(ks_alpha),
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "warning": "Distribution drift is an alert for investigation, not proof that the model is unusable.",
    }


@dataclass
class DriftMonitor:
    """Reference-distribution monitor with user-declared alert thresholds."""

    reference: np.ndarray
    feature_names: tuple[str, ...]
    psi_threshold: float = 0.2
    ks_alpha: float = 0.01

    @classmethod
    def fit(
        cls,
        reference: Any,
        *,
        feature_names: list[str] | None = None,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.01,
    ) -> DriftMonitor:
        x = np.asarray(reference, float)
        if x.ndim != 2 or len(x) < 10 or not np.isfinite(x).all():
            raise ValueError("reference must be finite [N>=10,F]")
        names = tuple(feature_names or [f"feature_{i}" for i in range(x.shape[1])])
        if len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError("feature_names must be unique and match feature width")
        return cls(x.copy(), names, psi_threshold, ks_alpha)

    def evaluate(self, current: Any) -> dict[str, Any]:
        return feature_drift_report(
            self.reference,
            current,
            feature_names=list(self.feature_names),
            psi_threshold=self.psi_threshold,
            ks_alpha=self.ks_alpha,
        )
