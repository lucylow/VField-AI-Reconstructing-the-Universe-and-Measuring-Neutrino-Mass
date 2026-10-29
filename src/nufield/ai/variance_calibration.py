"""Scalar variance calibration for probabilistic regression outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nufield.ai.losses import gaussian_nll_numpy


@dataclass(frozen=True)
class VarianceTemperature:
    """Multiplicative calibration factor for Gaussian predictive variance."""

    temperature: float
    n_calibration: int
    baseline_nll: float
    calibrated_nll: float
    search_min: float
    search_max: float
    grid_size: int
    method: str = "held-out NLL grid search"

    @classmethod
    def fit(
        cls,
        mean: Any,
        variance: Any,
        target: Any,
        *,
        minimum: float = 0.1,
        maximum: float = 10.0,
        grid_size: int = 401,
    ) -> VarianceTemperature:
        m, v, y = map(lambda a: np.asarray(a, float), (mean, variance, target))
        if m.shape != v.shape or m.shape != y.shape or m.ndim not in {1, 2}:
            raise ValueError("mean, variance, target must share vector or matrix shape")
        if (
            not np.isfinite(m).all()
            or not np.isfinite(v).all()
            or np.any(v <= 0)
            or not np.isfinite(y).all()
        ):
            raise ValueError("mean/target must be finite and variance positive")
        if not 0 < minimum < maximum or grid_size < 3:
            raise ValueError("invalid temperature search range")
        temps = np.exp(np.linspace(np.log(minimum), np.log(maximum), grid_size))
        scores = np.asarray([gaussian_nll_numpy(m, v * t, y).mean() for t in temps])
        index = int(np.argmin(scores))
        baseline = float(gaussian_nll_numpy(m, v, y).mean())
        return cls(
            float(temps[index]),
            int(len(m)),
            baseline,
            float(scores[index]),
            float(minimum),
            float(maximum),
            int(grid_size),
        )

    def apply(self, variance: Any) -> np.ndarray:
        v = np.asarray(variance, float)
        if not np.isfinite(v).all() or np.any(v <= 0):
            raise ValueError("variance must be finite and positive")
        return v * self.temperature

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "n_calibration": self.n_calibration,
            "baseline_nll": self.baseline_nll,
            "calibrated_nll": self.calibrated_nll,
            "search_min": self.search_min,
            "search_max": self.search_max,
            "grid_size": self.grid_size,
            "method": self.method,
            "warning": "Fit only on a calibration split independent of training and model selection.",
        }


def reliability_by_variance_bin(
    target: Any, mean: Any, variance: Any, *, n_bins: int = 8
) -> dict[str, Any]:
    """Compare mean predicted standard deviation with empirical residual RMS by uncertainty bins."""
    y, m, v = (np.asarray(x, float) for x in (target, mean, variance))
    if (
        y.shape != m.shape
        or y.shape != v.shape
        or not np.isfinite(y).all()
        or not np.isfinite(m).all()
        or not np.isfinite(v).all()
        or np.any(v <= 0)
    ):
        raise ValueError("target, mean and positive variance must share finite shapes")
    if y.ndim == 1:
        y, m, v = y[:, None], m[:, None], v[:, None]
    if y.ndim != 2 or len(y) < n_bins or n_bins < 2:
        raise ValueError("arrays must be [N,P] with N>=n_bins and n_bins>=2")
    predicted = np.sqrt(v).mean(axis=1)
    edges = np.quantile(predicted, np.linspace(0, 1, n_bins + 1))
    records = []
    for i in range(n_bins):
        mask = (predicted >= edges[i]) & (
            predicted <= edges[i + 1] if i == n_bins - 1 else predicted < edges[i + 1]
        )
        if not mask.any():
            continue
        records.append(
            {
                "n": int(mask.sum()),
                "predicted_sigma_mean": float(predicted[mask].mean()),
                "empirical_rmse": float(np.sqrt(np.mean((y[mask] - m[mask]) ** 2))),
                "predicted_sigma_median": float(np.median(predicted[mask])),
            }
        )
    return {
        "bins": records,
        "n": int(len(y)),
        "warning": "Reliability is descriptive; evaluate under independent held-out simulations.",
    }
