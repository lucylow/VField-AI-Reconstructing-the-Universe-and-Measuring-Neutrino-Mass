"""Invertible bounded-parameter transforms for neural density estimators.

A diagonal-Gaussian mixture in unconstrained coordinates can assign probability outside a physical
prior interval. The logit transform maps open bounded intervals to R so samples transform back into
the declared support. The endpoints are clipped by a documented epsilon; this introduces a tiny
numerical tail approximation and should be reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundedParameterTransform:
    low: np.ndarray
    high: np.ndarray
    eps: float = 1e-6

    def __post_init__(self):
        self.low = np.asarray(self.low, dtype=np.float64).reshape(-1)
        self.high = np.asarray(self.high, dtype=np.float64).reshape(-1)
        if self.low.shape != self.high.shape or self.low.size < 1:
            raise ValueError("low/high must be aligned nonempty vectors")
        if (
            not np.isfinite(self.low).all()
            or not np.isfinite(self.high).all()
            or np.any(self.high <= self.low)
        ):
            raise ValueError("bounds must be finite and strictly ordered")
        if not 0 < self.eps < 0.01:
            raise ValueError("eps must lie in (0, 0.01)")

    def _unit_interval(self, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(theta, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.low) or not np.isfinite(x).all():
            raise ValueError("theta must be finite [N,P] and match bounds")
        if np.any(x < self.low) or np.any(x > self.high):
            raise ValueError("theta lies outside declared prior bounds")
        u = (x - self.low) / (self.high - self.low)
        return np.clip(u, self.eps, 1 - self.eps)

    def forward(self, theta: np.ndarray) -> np.ndarray:
        u = self._unit_interval(theta)
        return np.log(u) - np.log1p(-u)

    def inverse(self, unconstrained: np.ndarray) -> np.ndarray:
        z = np.asarray(unconstrained, dtype=np.float64)
        if z.ndim < 1 or z.shape[-1] != len(self.low) or not np.isfinite(z).all():
            raise ValueError("unconstrained parameters must be finite and end in dimension P")
        # Stable logistic branch to avoid exp overflow for large-magnitude proposals.
        sig = np.empty_like(z)
        pos = z >= 0
        sig[pos] = 1 / (1 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        sig[~pos] = ez / (1 + ez)
        sig = np.clip(sig, self.eps, 1 - self.eps)
        return self.low + (self.high - self.low) * sig

    def state_dict(self) -> dict:
        return {
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "eps": self.eps,
            "transform": "logit of normalized finite interval",
            "endpoint_handling": "clip to [eps,1-eps]",
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> BoundedParameterTransform:
        return cls(
            np.asarray(state["low"]), np.asarray(state["high"]), float(state.get("eps", 1e-6))
        )
