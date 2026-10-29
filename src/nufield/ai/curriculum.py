"""Training-only curriculum and bounded hard-example weighting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CurriculumState:
    epoch: int
    total_epochs: int
    progress: float
    difficulty_threshold: float
    active_fraction: float
    mean_weight: float
    min_weight: float
    max_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "progress": self.progress,
            "difficulty_threshold": self.difficulty_threshold,
            "active_fraction": self.active_fraction,
            "mean_weight": self.mean_weight,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "warning": "Curriculum weights are fitted from training data only; do not infer difficulty from test labels.",
        }


def curriculum_weights(
    difficulty: Any,
    *,
    epoch: int,
    total_epochs: int,
    initial_fraction: float = 0.35,
    decay_strength: float = 3.0,
    minimum_weight: float = 0.15,
) -> tuple[np.ndarray, CurriculumState]:
    """Gradually expand training from easier to harder rows using a declared difficulty score.

    Difficulty must be constructed from predeclared simulation metadata or training-only residuals.
    No sample receives zero weight, preventing hard regions from being permanently excluded.
    """
    d = np.asarray(difficulty, float).reshape(-1)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("difficulty must be finite with at least two rows")
    if total_epochs < 1 or not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be in [1,total_epochs]")
    if not 0 < initial_fraction <= 1 or decay_strength < 0 or not 0 < minimum_weight <= 1:
        raise ValueError("invalid curriculum configuration")
    d_min, d_max = float(d.min()), float(d.max())
    scaled = np.zeros_like(d) if d_max == d_min else (d - d_min) / (d_max - d_min)
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    active_fraction = initial_fraction + (1 - initial_fraction) * progress
    threshold = float(np.quantile(scaled, active_fraction))
    excess = np.maximum(scaled - threshold, 0.0)
    weights = np.maximum(minimum_weight, np.exp(-decay_strength * excess))
    weights /= weights.mean()
    state = CurriculumState(
        epoch,
        total_epochs,
        float(progress),
        threshold,
        float(active_fraction),
        float(weights.mean()),
        float(weights.min()),
        float(weights.max()),
    )
    return weights, state


def training_residual_difficulty(residuals: Any, *, cap_quantile: float = 0.95) -> np.ndarray:
    """Summarize per-row training residuals into bounded difficulty scores.

    Only residuals from the train partition are permitted. This helper cannot determine provenance,
    so the caller must retain the split manifest that proves where residuals were calculated.
    """
    r = np.asarray(residuals, float)
    if r.ndim == 1:
        score = np.abs(r)
    elif r.ndim >= 2:
        score = np.sqrt(np.mean(r.reshape(len(r), -1) ** 2, axis=1))
    else:
        raise ValueError("residuals need a sample axis")
    if len(score) < 2 or not np.isfinite(score).all() or not 0.5 < cap_quantile <= 1:
        raise ValueError("invalid residuals or cap_quantile")
    cap = np.quantile(score, cap_quantile)
    clipped = np.minimum(score, cap)
    spread = clipped.max() - clipped.min()
    return (clipped - clipped.min()) / spread if spread > 0 else np.zeros_like(clipped)


def bounded_sample_weights(
    losses: Any,
    *,
    temperature: float = 1.0,
    minimum_weight: float = 0.25,
    maximum_weight: float = 4.0,
) -> np.ndarray:
    """Convert current training losses to stable mean-one hard-example weights."""
    x = np.asarray(losses, float).reshape(-1)
    if len(x) < 1 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("losses must be finite and nonnegative")
    if temperature <= 0 or minimum_weight <= 0 or maximum_weight < minimum_weight:
        raise ValueError("invalid weighting parameters")
    scaled = (x - np.median(x)) / max(float(np.std(x)), 1e-8)
    raw = np.exp(np.clip(scaled / temperature, -8, 8))
    raw = np.clip(raw, minimum_weight, maximum_weight)
    return raw / raw.mean()


def update_ema_weights(
    previous: Any,
    proposed: Any,
    *,
    momentum: float = 0.9,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> np.ndarray:
    """Smooth online training weights to reduce oscillation from noisy per-row losses."""
    old, new = np.asarray(previous, float), np.asarray(proposed, float)
    if (
        old.shape != new.shape
        or old.ndim != 1
        or not np.isfinite(old).all()
        or not np.isfinite(new).all()
    ):
        raise ValueError("previous and proposed must be finite vectors with identical shape")
    if not 0 <= momentum < 1 or min_weight <= 0 or max_weight < min_weight:
        raise ValueError("invalid EMA parameters")
    combined = momentum * old + (1 - momentum) * new
    combined = np.clip(combined, min_weight, max_weight)
    return combined / combined.mean()
