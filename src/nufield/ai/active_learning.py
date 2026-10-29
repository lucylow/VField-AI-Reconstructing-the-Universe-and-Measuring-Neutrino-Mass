"""Active simulation design for improving AI training coverage efficiently.

Acquisition scores are heuristics unless a particular expected-information-gain estimator is
specified and validated. Candidate parameters are suggestions for an expert-approved simulator;
this module never fabricates simulation outputs or overrides physical parameter constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import cdist

from nufield.training.active_design import latin_hypercube_candidates

AcquisitionKind = Literal["maximin", "ensemble_disagreement", "predictive_entropy", "hybrid"]


@dataclass(frozen=True)
class AcquisitionRecord:
    candidate_index: int
    parameters: list[float]
    score: float
    disagreement: float
    diversity: float
    method: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcquisitionBatch:
    selected: list[AcquisitionRecord]
    candidate_count: int
    parameter_low: list[float]
    parameter_high: list[float]
    seed: int
    caveat: str = (
        "Verify simulator feasibility and design validity before scheduling new simulations."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [r.to_dict() for r in self.selected],
            "candidate_count": self.candidate_count,
            "parameter_low": self.parameter_low,
            "parameter_high": self.parameter_high,
            "seed": self.seed,
            "caveat": self.caveat,
        }


def _validate_bounds(low: Any, high: Any) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(low, float), np.asarray(high, float)
    if lo.ndim != 1 or hi.shape != lo.shape or len(lo) == 0:
        raise ValueError("low/high must be same-length vectors")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
        raise ValueError("bounds must be finite and high > low")
    return lo, hi


def scale_to_prior_box(parameters: Any, low: Any, high: Any) -> np.ndarray:
    """Scale parameter coordinates to [0,1] using prior widths, never train-set statistics."""
    lo, hi = _validate_bounds(low, high)
    x = np.asarray(parameters, float)
    if x.ndim != 2 or x.shape[1] != len(lo) or not np.isfinite(x).all():
        raise ValueError("parameters must be finite [N,P]")
    if np.any(x < lo) or np.any(x > hi):
        raise ValueError("parameters outside prior box")
    return (x - lo) / (hi - lo)


def candidate_pool(low: Any, high: Any, *, n_candidates: int = 2048, seed: int = 42) -> np.ndarray:
    """Generate a reproducible Latin-hypercube candidate pool."""
    lo, hi = _validate_bounds(low, high)
    return latin_hypercube_candidates(lo, hi, n_candidates=n_candidates, seed=seed)


def ensemble_disagreement(predictions: Any, prior_widths: Any, *, eps: float = 1e-12) -> np.ndarray:
    """Calculate dimensionless epistemic disagreement over a candidate batch.

    Predictions must have shape [n_models,n_candidates,n_parameters]. The score measures model
    spread, not calibrated epistemic variance. Use independently initialized/trained models and
    validate ensemble coverage; identical checkpoints produce an uninformative acquisition score.
    """
    pred = np.asarray(predictions, float)
    widths = np.asarray(prior_widths, float)
    if pred.ndim != 3 or pred.shape[0] < 2 or not np.isfinite(pred).all():
        raise ValueError("predictions must be finite [M>=2,N,P]")
    if widths.shape != (pred.shape[-1],) or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("prior_widths must be positive [P]")
    spread = pred.std(axis=0, ddof=1) / widths[None, :]
    score = np.sqrt(np.mean(spread**2, axis=-1) + eps)
    return score


def predictive_entropy_score(
    means: Any, variances: Any, prior_widths: Any, *, eps: float = 1e-12
) -> np.ndarray:
    """Estimate predictive entropy minus mean component entropy for Gaussian ensembles.

    The approximation fits one diagonal Gaussian to the ensemble's total predictive moments and
    subtracts average member aleatoric entropy. This is a rough mutual-information proxy, not exact
    Bayesian information gain. Outputs are dimensionless after scaling by prior widths.
    """
    mu, var = np.asarray(means, float), np.asarray(variances, float)
    widths = np.asarray(prior_widths, float)
    if mu.ndim != 3 or mu.shape[0] < 2 or var.shape != mu.shape:
        raise ValueError("means/variances must have matching [M,N,P], M>=2")
    if not np.isfinite(mu).all() or not np.isfinite(var).all() or np.any(var <= 0):
        raise ValueError("means must be finite and variances finite positive")
    if widths.shape != (mu.shape[-1],) or np.any(widths <= 0):
        raise ValueError("prior_widths must be positive [P]")
    scaled_var = var / widths[None, None, :] ** 2
    total_var = scaled_var.mean(axis=0) + (mu / widths[None, None, :]).var(axis=0, ddof=1)
    entropy_pred = 0.5 * np.log(2 * np.pi * np.e * np.maximum(total_var, eps)).sum(axis=-1)
    entropy_member = 0.5 * np.log(2 * np.pi * np.e * np.maximum(scaled_var, eps)).sum(axis=-1).mean(
        axis=0
    )
    return np.maximum(entropy_pred - entropy_member, 0.0)


def _distance_to_set(candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(reference) == 0:
        return np.ones(len(candidates), dtype=float)
    return cdist(candidates, reference).min(axis=1)


def select_candidates(
    candidates: Any,
    low: Any,
    high: Any,
    *,
    n_select: int,
    existing_parameters: Any | None = None,
    predictions: Any | None = None,
    variances: Any | None = None,
    method: AcquisitionKind = "hybrid",
    diversity_weight: float = 0.35,
    prediction_scales: Any | None = None,
    seed: int = 42,
) -> AcquisitionBatch:
    """Greedily select uncertain and/or diverse parameter points.

    Parameters are first scaled by the prior box so incomparable physical units do not dominate the
    distance. Diversity is recalculated after every selected candidate. Prediction shape must be
    [M,N,P_target], where its second axis indexes the candidate array.
    """
    lo, hi = _validate_bounds(low, high)
    cand = np.asarray(candidates, float)
    if cand.ndim != 2 or cand.shape[1] != len(lo) or not np.isfinite(cand).all():
        raise ValueError("candidate matrix must be finite [N,P]")
    if np.any(cand < lo) or np.any(cand > hi):
        raise ValueError("candidates lie outside the prior box")
    if not 1 <= n_select <= len(cand):
        raise ValueError("n_select must be within candidate count")
    if not 0 <= diversity_weight <= 1:
        raise ValueError("diversity_weight must be in [0,1]")
    if method not in {"maximin", "ensemble_disagreement", "predictive_entropy", "hybrid"}:
        raise ValueError("unsupported acquisition method")
    x_scaled = scale_to_prior_box(cand, lo, hi)
    existing = (
        np.empty((0, len(lo)))
        if existing_parameters is None
        else np.asarray(existing_parameters, float)
    )
    if existing.size:
        if existing.ndim != 2 or existing.shape[1] != len(lo) or not np.isfinite(existing).all():
            raise ValueError("existing_parameters must be finite [N,P]")
        if np.any(existing < lo) or np.any(existing > hi):
            raise ValueError("existing parameters are outside prior bounds")
        reference = scale_to_prior_box(existing, lo, hi)
    else:
        reference = np.empty((0, len(lo)))

    disagree = np.zeros(len(cand))
    if method in {"ensemble_disagreement", "hybrid"}:
        if predictions is None:
            raise ValueError(f"{method} requires ensemble predictions")
        pred = np.asarray(predictions, float)
        if pred.shape[1] != len(cand):
            raise ValueError("prediction candidate axis does not match candidates")
        target_scales = (
            np.ones(pred.shape[-1])
            if prediction_scales is None
            else np.asarray(prediction_scales, float)
        )
        disagree = ensemble_disagreement(pred, target_scales)
    elif method == "predictive_entropy":
        if predictions is None or variances is None:
            raise ValueError("predictive_entropy requires member means and variances")
        pred = np.asarray(predictions, float)
        if pred.shape[1] != len(cand):
            raise ValueError("prediction candidate axis does not match candidates")
        target_scales = (
            np.ones(pred.shape[-1])
            if prediction_scales is None
            else np.asarray(prediction_scales, float)
        )
        disagree = predictive_entropy_score(pred, variances, target_scales)
    dmax = float(disagree.max())
    if dmax > 0:
        disagree /= dmax
    selected_indices: list[int] = []
    records: list[AcquisitionRecord] = []
    current = reference.copy()
    for _ in range(n_select):
        distances = _distance_to_set(x_scaled, current)
        dscale = max(float(distances.max()), 1e-12)
        diversity = distances / dscale
        if method == "maximin":
            score = diversity.copy()
        elif method in {"ensemble_disagreement", "predictive_entropy"}:
            score = disagree.copy()
        else:
            score = (1 - diversity_weight) * disagree + diversity_weight * diversity
        if selected_indices:
            score[np.asarray(selected_indices)] = -np.inf
        idx = int(np.argmax(score))
        selected_indices.append(idx)
        if method == "maximin":
            explanation = "maximum minimum distance in prior-scaled parameter space"
        elif method == "ensemble_disagreement":
            explanation = (
                "high normalized model-ensemble disagreement; not a calibrated uncertainty"
            )
        elif method == "predictive_entropy":
            explanation = "approximate Gaussian-ensemble mutual information; validate before use"
        else:
            explanation = "weighted combination of ensemble disagreement and diversity"
        records.append(
            AcquisitionRecord(
                idx,
                cand[idx].tolist(),
                float(score[idx]),
                float(disagree[idx]),
                float(diversity[idx]),
                method,
                explanation,
            )
        )
        current = np.vstack([current, x_scaled[idx]])
    return AcquisitionBatch(records, len(cand), lo.tolist(), hi.tolist(), int(seed))


def acquisition_ablation(
    candidates: Any,
    low: Any,
    high: Any,
    *,
    existing_parameters: Any | None,
    predictions: Any,
    n_select: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a deterministic maximin/disagreement/hybrid selection comparison."""
    outputs = {}
    for method in ("maximin", "ensemble_disagreement", "hybrid"):
        kwargs = {"predictions": predictions} if method != "maximin" else {}
        outputs[method] = select_candidates(
            candidates,
            low,
            high,
            n_select=n_select,
            existing_parameters=existing_parameters,
            method=method,
            seed=seed,
            **kwargs,
        ).to_dict()
    return {
        "comparison": outputs,
        "interpretation": "Acquisition rank is a design proposal. Compare downstream simulator labels and posterior calibration.",
    }
