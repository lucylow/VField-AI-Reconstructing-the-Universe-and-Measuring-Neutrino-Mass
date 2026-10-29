"""Finite-sample conformal intervals for held-out regression calibration sets.

Conformal prediction does not repair distribution shift. Its coverage guarantee requires exchangeable
calibration and future examples, and the calibration set must not be reused for model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def conformal_quantile(scores: Any, *, alpha: float = 0.1) -> float:
    """Finite-sample corrected empirical quantile using the ``higher`` order statistic."""
    x = np.asarray(scores, float).reshape(-1)
    if len(x) < 1 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("scores must be a nonempty finite nonnegative vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    rank = int(np.ceil((len(x) + 1) * (1 - alpha)))
    if rank > len(x):
        return float(np.max(x))
    return float(np.partition(x, rank - 1)[rank - 1])


@dataclass(frozen=True)
class ConformalRegressor:
    """Split-conformal calibrated prediction intervals for scalar or multi-target regression."""

    alpha: float
    quantile: float
    target_dim: int
    simultaneous: bool
    scale_floor: float
    calibration_n: int
    score_definition: str = "absolute standardized residual"

    @classmethod
    def fit(
        cls,
        y_calibration: Any,
        mean_calibration: Any,
        *,
        scale_calibration: Any | None = None,
        alpha: float = 0.1,
        simultaneous: bool = True,
        scale_floor: float = 1e-8,
    ) -> ConformalRegressor:
        y, mean = np.asarray(y_calibration, float), np.asarray(mean_calibration, float)
        if y.ndim == 1:
            y, mean = y[:, None], mean[:, None]
        if y.shape != mean.shape or y.ndim != 2 or len(y) < 1:
            raise ValueError("calibration targets and means must share shape [N,P]")
        if not np.isfinite(y).all() or not np.isfinite(mean).all():
            raise ValueError("calibration arrays must be finite")
        if scale_calibration is None:
            scale = np.ones_like(y)
        else:
            scale = np.asarray(scale_calibration, float)
            if scale.ndim == 1:
                scale = scale[:, None]
            if scale.shape != y.shape or not np.isfinite(scale).all() or np.any(scale < 0):
                raise ValueError("scale_calibration must be finite, nonnegative, and match targets")
            scale = np.maximum(scale, scale_floor)
        if scale_floor <= 0:
            raise ValueError("scale_floor must be positive")
        residual = np.abs(y - mean) / scale
        scores = residual.max(axis=1) if simultaneous else residual.reshape(-1)
        q = conformal_quantile(scores, alpha=alpha)
        return cls(
            float(alpha), q, int(y.shape[1]), bool(simultaneous), float(scale_floor), int(len(y))
        )

    def predict(self, mean: Any, *, scale: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(mean, float)
        if m.ndim == 1:
            m = m[:, None] if self.target_dim == 1 else m[None, :]
        if m.ndim != 2 or m.shape[1] != self.target_dim or not np.isfinite(m).all():
            raise ValueError("mean must be finite [N,P] with configured target dimension")
        if scale is None:
            s = np.ones_like(m)
        else:
            s = np.asarray(scale, float)
            if s.ndim == 1:
                if self.target_dim == 1 and len(s) == len(m):
                    s = s[:, None]
                elif len(s) == self.target_dim:
                    s = np.broadcast_to(s, m.shape)
            if s.shape != m.shape or not np.isfinite(s).all() or np.any(s < 0):
                raise ValueError("scale must be finite, nonnegative, and match means")
            s = np.maximum(s, self.scale_floor)
        radius = self.quantile * s
        return m - radius, m + radius

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "quantile": self.quantile,
            "target_dim": self.target_dim,
            "simultaneous": self.simultaneous,
            "scale_floor": self.scale_floor,
            "calibration_n": self.calibration_n,
            "score_definition": self.score_definition,
            "exchangeability_warning": "Coverage can fail under distribution shift or dependent samples.",
        }


def empirical_coverage(y: Any, lower: Any, upper: Any) -> dict[str, float]:
    """Compute per-coordinate and simultaneous empirical interval coverage."""
    target, lo, hi = map(lambda v: np.asarray(v, float), (y, lower, upper))
    if target.ndim == 1:
        target, lo, hi = target[:, None], lo[:, None], hi[:, None]
    if target.shape != lo.shape or target.shape != hi.shape:
        raise ValueError("target and interval arrays must have identical shape")
    if not all(np.isfinite(a).all() for a in (target, lo, hi)) or np.any(lo > hi):
        raise ValueError("interval arrays are invalid")
    covered = (target >= lo) & (target <= hi)
    return {
        "n": int(len(target)),
        "target_dim": int(target.shape[1]),
        "marginal_coverage_mean": float(covered.mean()),
        "marginal_coverage_by_target": covered.mean(axis=0).tolist(),
        "simultaneous_coverage": float(covered.all(axis=1).mean()),
        "mean_interval_width": float((hi - lo).mean()),
    }
