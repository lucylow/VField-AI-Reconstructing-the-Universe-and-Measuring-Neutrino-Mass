"""Prior-bounded proposal tools for allocating additional cosmological simulations.

This module suggests parameter locations, not simulation outcomes. Candidate selection should be
combined with the scientific simulation design and validated by an expert; it is not an autonomous
simulator or a guarantee of optimal Fisher information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DesignCandidate:
    index: int
    parameters: list[float]
    score: float
    nearest_existing_distance: float
    reason: str

    def to_dict(self):
        return asdict(self)


def _bounds(low, high):
    lo = np.asarray(low, dtype=float)
    hi = np.asarray(high, dtype=float)
    if (
        lo.ndim != 1
        or hi.shape != lo.shape
        or len(lo) == 0
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(hi <= lo)
    ):
        raise ValueError("prior bounds must be finite vectors with high > low")
    return lo, hi


def latin_hypercube_candidates(low, high, *, n_candidates: int = 500, seed: int = 42) -> np.ndarray:
    """Generate a NumPy-only Latin hypercube within a declared box prior."""
    lo, hi = _bounds(low, high)
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive")
    rng = np.random.default_rng(seed)
    n_dim = len(lo)
    base = np.empty((n_candidates, n_dim), dtype=float)
    for j in range(n_dim):
        base[:, j] = (rng.permutation(n_candidates) + rng.random(n_candidates)) / n_candidates
    return lo + base * (hi - lo)


def _scale(points, low, high):
    return (np.asarray(points, dtype=float) - low) / (high - low)


def maximin_design(
    existing, low, high, *, n_select: int, n_candidates: int = 2000, seed: int = 42
) -> list[DesignCandidate]:
    """Greedily choose candidates distant from existing and already-selected points in scaled prior space."""
    lo, hi = _bounds(low, high)
    cand = latin_hypercube_candidates(lo, hi, n_candidates=n_candidates, seed=seed)
    if n_select < 1 or n_select > len(cand):
        raise ValueError("invalid n_select")
    old = np.asarray(existing, dtype=float)
    if old.size:
        if old.ndim != 2 or old.shape[1] != len(lo) or not np.isfinite(old).all():
            raise ValueError("existing points must be [N,P] finite")
        if np.any(old < lo) or np.any(old > hi):
            raise ValueError("existing points lie outside prior bounds")
    scaled_cand = _scale(cand, lo, hi)
    scaled_old = _scale(old, lo, hi) if len(old) else np.empty((0, len(lo)))
    selected: list[int] = []
    current = scaled_old.copy()
    scores = np.zeros(len(cand))
    for _ in range(n_select):
        dist = (
            np.sqrt(((scaled_cand[:, None, :] - current[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            if len(current)
            else np.ones(len(cand))
        )
        if selected:
            dist[np.asarray(selected)] = -np.inf
        idx = int(np.argmax(dist))
        selected.append(idx)
        scores[idx] = float(dist[idx])
        current = np.vstack([current, scaled_cand[idx]])
    return [
        DesignCandidate(
            index=i,
            parameters=cand[i].tolist(),
            score=float(scores[i]),
            nearest_existing_distance=float(scores[i]),
            reason="maximin coverage in unit-scaled prior space",
        )
        for i in selected
    ]


def disagreement_design(
    candidates, ensemble_predictions, low, high, *, n_select: int
) -> list[DesignCandidate]:
    """Rank candidate points by model-ensemble disagreement with a distance-based diversity penalty.

    `ensemble_predictions` has shape [n_models,n_candidates,n_targets]. Its ensemble spread is a
    heuristic epistemic signal, not a calibrated uncertainty estimate. A selection penalty keeps
    proposed runs from clustering on nearly identical parameter points.
    """
    lo, hi = _bounds(low, high)
    x = np.asarray(candidates, dtype=float)
    pred = np.asarray(ensemble_predictions, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(lo) or np.any(x < lo) or np.any(x > hi):
        raise ValueError("candidate grid is invalid or outside prior bounds")
    if (
        pred.ndim != 3
        or pred.shape[1] != len(x)
        or pred.shape[0] < 2
        or not np.isfinite(pred).all()
    ):
        raise ValueError("predictions must be finite [ensemble>=2,candidates,targets]")
    if n_select < 1 or n_select > len(x):
        raise ValueError("invalid n_select")
    spread = pred.std(axis=0, ddof=1).mean(axis=1)
    spread = spread / max(float(spread.max()), 1e-12)
    xs = _scale(x, lo, hi)
    chosen: list[int] = []
    for _ in range(n_select):
        if not chosen:
            score = spread.copy()
        else:
            dist = np.sqrt(
                ((xs[:, None, :] - xs[np.asarray(chosen)][None, :, :]) ** 2).sum(axis=2)
            ).min(axis=1)
            dist /= max(float(dist.max()), 1e-12)
            score = spread * (0.5 + 0.5 * dist)
            score[np.asarray(chosen)] = -np.inf
        chosen.append(int(np.argmax(score)))
    return [
        DesignCandidate(
            i,
            x[i].tolist(),
            float(spread[i]),
            float(spread[i]),
            "ensemble disagreement heuristic; inspect coverage and prior density before scheduling",
        )
        for i in chosen
    ]
