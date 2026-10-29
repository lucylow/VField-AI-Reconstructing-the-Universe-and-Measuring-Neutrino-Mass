"""Validation-only selection for predeclared model candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    name: str
    validation_loss: float
    complexity: float
    calibration_penalty: float
    admissible: bool


def choose_by_validation(
    results: list[ModelSelectionResult], *, max_calibration_penalty: float
) -> ModelSelectionResult:
    """Choose a candidate using validation metrics only, with a calibration gate."""
    if not results or max_calibration_penalty < 0:
        raise ValueError("results must be non-empty and penalty threshold non-negative")
    eligible = []
    for item in results:
        vals = [item.validation_loss, item.complexity, item.calibration_penalty]
        if any(not np.isfinite(v) or v < 0 for v in vals):
            raise ValueError("selection metrics must be finite and non-negative")
        if item.admissible and item.calibration_penalty <= max_calibration_penalty:
            eligible.append(item)
    if not eligible:
        raise ValueError("no validation candidate passed the calibration gate")
    return min(eligible, key=lambda r: (r.validation_loss, r.complexity))


def assert_test_set_unused(stage: str) -> None:
    if stage.lower() in {"tune", "select", "calibrate", "feature_select", "early_stop"}:
        raise ValueError("the frozen test set must not be used for model selection or calibration")
