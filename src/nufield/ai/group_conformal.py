"""Group-level split conformal calibration for correlated observations within simulation groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nufield.ai.conformal import conformal_quantile


@dataclass(frozen=True)
class GroupConformalRegressor:
    """Conformal calibration where each parent simulation contributes one nonconformity score."""

    alpha: float
    quantile: float
    target_dim: int
    n_calibration_groups: int
    group_score: str = "maximum absolute standardized residual within each parent group"

    @classmethod
    def fit(
        cls,
        target: Any,
        mean: Any,
        group_ids: Any,
        *,
        alpha: float = 0.1,
        scale: Any | None = None,
        scale_floor: float = 1e-8,
    ) -> GroupConformalRegressor:
        y, m = np.asarray(target, float), np.asarray(mean, float)
        if y.ndim == 1:
            y, m = y[:, None], m[:, None]
        groups = np.asarray(group_ids).reshape(-1).astype(str)
        if (
            y.ndim != 2
            or y.shape != m.shape
            or len(y) != len(groups)
            or not np.isfinite(y).all()
            or not np.isfinite(m).all()
        ):
            raise ValueError("target/mean must be finite [N,P] aligned to group_ids")
        if scale is None:
            s = np.ones_like(y)
        else:
            s = np.asarray(scale, float)
            if s.ndim == 1:
                s = s[:, None] if y.shape[1] == 1 and len(s) == len(y) else s[None, :]
            if s.shape != y.shape or not np.isfinite(s).all() or np.any(s < 0):
                raise ValueError("scale must be finite, nonnegative and match target shape")
            s = np.broadcast_to(s, y.shape)
        if scale_floor <= 0:
            raise ValueError("scale_floor must be positive")
        s = np.maximum(s, scale_floor)
        unique = np.unique(groups)
        if len(unique) < 2:
            raise ValueError("at least two independent calibration groups are required")
        scores = []
        residual = np.abs(y - m) / s
        for g in unique:
            group_scores = residual[groups == g]
            scores.append(float(group_scores.max()))
        q = conformal_quantile(scores, alpha=alpha)
        return cls(float(alpha), q, int(y.shape[1]), int(len(unique)))

    def predict(self, mean: Any, *, scale: Any | None = None):
        m = np.asarray(mean, float)
        if m.ndim == 1:
            m = m[:, None]
        if m.ndim != 2 or m.shape[1] != self.target_dim or not np.isfinite(m).all():
            raise ValueError("mean must be finite [N,P]")
        if scale is None:
            s = np.ones_like(m)
        else:
            s = np.asarray(scale, float)
            if s.ndim == 1:
                s = s[:, None] if self.target_dim == 1 and len(s) == len(m) else s[None, :]
            if s.shape[1] != self.target_dim or not np.isfinite(s).all() or np.any(s < 0):
                raise ValueError("scale shape/values invalid")
            s = np.broadcast_to(s, m.shape)
        radius = self.quantile * np.maximum(s, 1e-8)
        return m - radius, m + radius

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "quantile": self.quantile,
            "target_dim": self.target_dim,
            "n_calibration_groups": self.n_calibration_groups,
            "group_score": self.group_score,
            "warning": "Coverage depends on exchangeability of parent groups and representative calibration groups.",
        }
