"""A simple weighted-particle likelihood-free baseline for software benchmarking.

This is an approximate nearest-neighbour/kernel estimator, not a validated SBI algorithm. The training
simulation particles approximate the prior; local weights depend on distance in standardized feature
space. Use it as a baseline to test interfaces, splits, and diagnostics, not as final research inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.features.scaling import Standardizer


@dataclass(slots=True)
class PosteriorSummary:
    samples: np.ndarray
    point_estimate: np.ndarray
    effective_sample_size: float
    nearest_distance: float


class KernelPosterior:
    """Kernel-weighted posterior particle approximation in a learned/engineered summary space."""

    def __init__(self, bandwidth: float = 1.0, top_k: int = 128):
        if bandwidth <= 0 or top_k < 1:
            raise ValueError("bandwidth must be positive and top_k >= 1")
        self.bandwidth = float(bandwidth)
        self.top_k = int(top_k)
        self.scaler = Standardizer()
        self.features_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.training_ids_: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, theta: np.ndarray, simulation_ids: np.ndarray | None = None
    ) -> KernelPosterior:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(theta, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("features and theta must be row-aligned 2-D arrays")
        if len(x) < 5 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("at least five finite training rows are required")
        self.features_ = self.scaler.fit_transform(x)
        self.theta_ = y.astype(np.float32)
        self.training_ids_ = (
            np.asarray(simulation_ids).astype(str)
            if simulation_ids is not None
            else np.asarray([f"row-{i}" for i in range(len(x))])
        )
        if len(self.training_ids_) != len(x) or len(set(self.training_ids_.tolist())) != len(x):
            raise ValueError("training IDs must be unique and row-aligned")
        return self

    def _weights(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.features_ is None or self.theta_ is None:
            raise RuntimeError("fit must be called before inference")
        obs = self.scaler.transform(np.asarray(observation, dtype=np.float64).reshape(1, -1))[0]
        dist = np.linalg.norm(self.features_ - obs[None, :], axis=1)
        k = min(self.top_k, len(dist))
        idx = np.argpartition(dist, k - 1)[:k]
        idx = idx[np.argsort(dist[idx])]
        # Stabilize exponent by subtracting minimum distance; bandwidth is in standardized-feature units.
        delta = dist[idx] - dist[idx].min()
        weights = np.exp(-0.5 * (delta / self.bandwidth) ** 2)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            weights = np.full(len(idx), 1 / len(idx))
        else:
            weights /= total
        return idx, weights

    def summarize(
        self, observation: np.ndarray, n_samples: int = 2000, rng: np.random.Generator | None = None
    ) -> PosteriorSummary:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        if rng is None:
            rng = np.random.default_rng(0)
        idx, weights = self._weights(observation)
        assert self.theta_ is not None and self.features_ is not None
        chosen = rng.choice(idx, size=n_samples, replace=True, p=weights)
        samples = self.theta_[chosen]
        point = np.average(self.theta_[idx], axis=0, weights=weights)
        ess = float(1.0 / np.sum(weights**2))
        obs = self.scaler.transform(np.asarray(observation).reshape(1, -1))[0]
        nearest = float(np.min(np.linalg.norm(self.features_ - obs[None, :], axis=1)))
        return PosteriorSummary(
            samples=samples,
            point_estimate=point.astype(np.float32),
            effective_sample_size=ess,
            nearest_distance=nearest,
        )

    def sample(
        self, observation: np.ndarray, n_samples: int = 2000, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        return self.summarize(observation, n_samples, rng).samples

    def point_estimate(self, observation: np.ndarray) -> np.ndarray:
        return self.summarize(observation, 10, np.random.default_rng(0)).point_estimate

    def metadata(self) -> dict:
        return {
            "algorithm": "standardized-feature Gaussian kernel weighted particles",
            "bandwidth": self.bandwidth,
            "top_k": self.top_k,
            "physics_level": "toy",
            "posterior_status": "approximate; not calibrated by default",
            "n_training_particles": 0 if self.theta_ is None else int(len(self.theta_)),
        }
