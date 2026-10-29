"""Evidence-gated candidate selection using validation metrics only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateEvidence:
    name: str
    validation_score: float
    validation_calibration_error: float | None
    robustness_score: float | None
    measured_runtime_seconds: float | None
    measured_peak_memory_mb: float | None
    train_validation_test_disjoint: bool
    trained_on_verified_target_variation: bool
    simulator_status: str
    validation_score_name: str = "nll"
    notes: tuple[str, ...] = ()

    def gates(
        self, *, max_calibration_error: float = 0.1, min_robustness_score: float = 0.0
    ) -> dict[str, bool]:
        return {
            "group_split": bool(self.train_validation_test_disjoint),
            "target_variation": bool(self.trained_on_verified_target_variation),
            "simulator_verified": self.simulator_status == "validated",
            "calibration_available": self.validation_calibration_error is not None
            and np.isfinite(self.validation_calibration_error),
            "calibration_threshold": self.validation_calibration_error is not None
            and np.isfinite(self.validation_calibration_error)
            and self.validation_calibration_error <= max_calibration_error,
            "robustness_available": self.robustness_score is not None
            and np.isfinite(self.robustness_score),
            "robustness_threshold": self.robustness_score is not None
            and np.isfinite(self.robustness_score)
            and self.robustness_score >= min_robustness_score,
        }

    def to_dict(self, **kwargs):
        result = asdict(self)
        result["gates"] = self.gates(**kwargs)
        result["eligible_for_scientific_claim"] = all(result["gates"].values())
        return result


def rank_candidates(
    candidates: list[CandidateEvidence],
    *,
    direction: str = "minimize",
    max_calibration_error: float = 0.1,
    min_robustness_score: float = 0.0,
    require_validated_simulator: bool = False,
) -> dict[str, Any]:
    """Rank models by validation score only after mandatory predeclared gates.

    Never pass test metrics here. Candidate records deliberately have no test-score field to reduce
    the temptation to select a model based on final test performance.
    """
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("candidate names must be unique")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be minimize or maximize")
    evaluated = []
    for c in candidates:
        if not np.isfinite(c.validation_score):
            gate_errors = {"finite_validation_score": False}
        else:
            gate_errors = {"finite_validation_score": True}
        gates = c.gates(
            max_calibration_error=max_calibration_error, min_robustness_score=min_robustness_score
        )
        if require_validated_simulator:
            gates["simulator_verified"] = c.simulator_status == "validated"
        eligible = all(gates.values()) and gate_errors["finite_validation_score"]
        evaluated.append(
            {
                "candidate": c.name,
                "validation_score": float(c.validation_score),
                "gates": {**gates, **gate_errors},
                "eligible": bool(eligible),
                "runtime_seconds": c.measured_runtime_seconds,
                "peak_memory_mb": c.measured_peak_memory_mb,
            }
        )
    eligible = [x for x in evaluated if x["eligible"]]
    eligible.sort(key=lambda row: row["validation_score"], reverse=direction == "maximize")
    return {
        "ranking": eligible,
        "rejected_or_gated": [x for x in evaluated if not x["eligible"]],
        "winner": eligible[0]["candidate"] if eligible else None,
        "selection_metric": "validation_score",
        "direction": direction,
        "warning": "Selection does not use or authorize test-set metrics. Validate the selected checkpoint once on a locked test set.",
    }


def validation_one_standard_error(
    scores_by_fold: Any, *, complexity: Any | None = None, direction: str = "minimize"
) -> dict[str, Any]:
    """One-standard-error rule favoring a simpler model statistically indistinguishable from best.

    ``scores_by_fold`` has shape [candidate,fold], where the same group-aware folds are used across
    candidates. Complexity may be parameter count/runtime proxy; the smallest value among candidates
    within one standard error of the best mean score is selected.
    """
    scores = np.asarray(scores_by_fold, float)
    if (
        scores.ndim != 2
        or scores.shape[0] < 2
        or scores.shape[1] < 2
        or not np.isfinite(scores).all()
    ):
        raise ValueError("scores_by_fold must be finite [candidates>=2,folds>=2]")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be minimize or maximize")
    means = scores.mean(axis=1)
    ses = scores.std(axis=1, ddof=1) / np.sqrt(scores.shape[1])
    best = int(np.argmin(means) if direction == "minimize" else np.argmax(means))
    threshold = means[best] + ses[best] if direction == "minimize" else means[best] - ses[best]
    acceptable = means <= threshold if direction == "minimize" else means >= threshold
    comp = (
        np.arange(len(means), dtype=float) if complexity is None else np.asarray(complexity, float)
    )
    if comp.shape != means.shape or not np.isfinite(comp).all():
        raise ValueError("complexity must be finite with one value per candidate")
    ids = np.flatnonzero(acceptable)
    chosen = int(ids[np.argmin(comp[ids])])
    return {
        "best_mean_candidate_index": best,
        "one_se_candidate_index": chosen,
        "means": means.tolist(),
        "standard_errors": ses.tolist(),
        "acceptable_candidates": ids.tolist(),
        "complexity": comp.tolist(),
        "threshold": float(threshold),
    }
