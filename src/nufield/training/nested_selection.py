"""Strict validation-only model selection helpers that preserve a locked final test set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

MetricDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class CandidateMetric:
    candidate_id: str
    metric_name: str
    value: float
    direction: MetricDirection
    compute_seconds: float
    peak_memory_mb: float | None = None
    calibrated: bool = False
    validation_group_hash: str = ""

    def to_dict(self):
        return asdict(self)


def hash_ids(ids: np.ndarray | list[str]) -> str:
    values = sorted(str(x) for x in np.asarray(ids).reshape(-1))
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def select_candidate(
    metrics: list[CandidateMetric],
    *,
    metric_name: str,
    expected_validation_hash: str,
    require_calibration: bool = True,
    max_compute_seconds: float | None = None,
) -> dict[str, Any]:
    """Choose a model based solely on validation scores and a frozen validation-group hash.

    This function intentionally receives no test labels. The winner's performance must be measured
    once on the locked test set after selection and hyperparameters are frozen.
    """
    eligible = []
    excluded = []
    for m in metrics:
        reason = []
        if m.metric_name != metric_name:
            reason.append("wrong metric")
        if not np.isfinite(m.value):
            reason.append("nonfinite validation metric")
        if m.compute_seconds < 0:
            reason.append("invalid compute time")
        if max_compute_seconds is not None and m.compute_seconds > max_compute_seconds:
            reason.append("compute budget exceeded")
        if m.validation_group_hash != expected_validation_hash:
            reason.append("validation group hash mismatch")
        if require_calibration and not m.calibrated:
            reason.append("calibration gate not passed")
        if reason:
            excluded.append({"candidate_id": m.candidate_id, "reasons": reason})
        else:
            eligible.append(m)
    if not eligible:
        return {
            "selected": None,
            "metric_name": metric_name,
            "eligible_count": 0,
            "excluded": excluded,
            "status": "NO_ELIGIBLE_MODEL",
            "warning": "Do not select from the locked test set or waive calibration without documenting a scientific reason.",
        }
    directions = {m.direction for m in eligible}
    if len(directions) != 1:
        raise ValueError("all candidate metrics must use the same direction")
    direction = next(iter(directions))
    ordered = sorted(eligible, key=lambda m: m.value, reverse=direction == "maximize")
    winner = ordered[0]
    return {
        "selected": winner.to_dict(),
        "metric_name": metric_name,
        "direction": direction,
        "eligible_count": len(eligible),
        "excluded": excluded,
        "ranked_validation_only": [m.to_dict() for m in ordered],
        "status": "WINNER_SELECTED_ON_VALIDATION_ONLY",
        "next_step": "Freeze candidate and config, then evaluate once on locked test groups; include paired uncertainty and SBC diagnostics.",
    }


def write_selection_decision(
    metrics: list[CandidateMetric],
    *,
    output: str,
    metric_name: str,
    expected_validation_hash: str,
    require_calibration: bool = True,
    max_compute_seconds: float | None = None,
) -> dict[str, Any]:
    decision = select_candidate(
        metrics,
        metric_name=metric_name,
        expected_validation_hash=expected_validation_hash,
        require_calibration=require_calibration,
        max_compute_seconds=max_compute_seconds,
    )
    from pathlib import Path

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    decision["output_path"] = str(path.resolve())
    return decision
