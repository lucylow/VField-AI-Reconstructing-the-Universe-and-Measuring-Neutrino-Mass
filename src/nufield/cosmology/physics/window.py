"""Survey-window mixing contracts for binned power-spectrum predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PowerWindow:
    """Linear binned-window operator mapping theory-bandpowers to observed-bandpowers."""

    matrix: np.ndarray
    input_labels: tuple[str, ...] = ()
    output_labels: tuple[str, ...] = ()
    provenance: str = "unspecified"

    def __post_init__(self) -> None:
        w = np.asarray(self.matrix, float)
        if w.ndim != 2 or min(w.shape) < 1 or np.any(~np.isfinite(w)):
            raise ValueError("window matrix must be a finite 2-D array")
        if self.input_labels and len(self.input_labels) != w.shape[1]:
            raise ValueError("input_labels length does not match window columns")
        if self.output_labels and len(self.output_labels) != w.shape[0]:
            raise ValueError("output_labels length does not match window rows")
        object.__setattr__(self, "matrix", w.copy())

    def apply(self, theory_bandpowers: np.ndarray) -> np.ndarray:
        p = np.asarray(theory_bandpowers, float)
        if p.ndim != 1 or p.size != self.matrix.shape[1] or np.any(~np.isfinite(p)):
            raise ValueError(
                "theory bandpowers must be a finite vector matching the window columns"
            )
        return self.matrix @ p

    def effective_support(self, threshold: float = 1e-3) -> np.ndarray:
        """Return boolean output-by-input support mask based on row-relative maximum weight."""
        if threshold < 0 or threshold >= 1:
            raise ValueError("threshold must lie in [0,1)")
        rowmax = np.max(np.abs(self.matrix), axis=1, keepdims=True)
        return np.abs(self.matrix) >= threshold * rowmax

    def row_normalization(self) -> np.ndarray:
        """Return row sums; do not assume the window is normalized automatically."""
        return np.sum(self.matrix, axis=1)


def identity_window(n_bins: int, labels: tuple[str, ...] = ()) -> PowerWindow:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if labels and len(labels) != n_bins:
        raise ValueError("labels length must match n_bins")
    return PowerWindow(np.eye(n_bins), labels, labels, "identity-test-window")


def validate_window_for_inference(
    window: PowerWindow,
    *,
    require_nonnegative: bool = False,
    require_row_sum_one: bool = False,
    tolerance: float = 1e-3,
) -> list[str]:
    """Return review findings for a provided survey-window matrix."""
    issues: list[str] = []
    w = window.matrix
    if require_nonnegative and np.any(w < -tolerance):
        issues.append(
            "window has negative entries; confirm they are expected for the chosen estimator"
        )
    row_sums = w.sum(axis=1)
    if require_row_sum_one and not np.allclose(row_sums, 1.0, atol=tolerance, rtol=0):
        issues.append("window rows are not normalized to unity")
    if np.any(np.max(np.abs(w), axis=1) == 0):
        issues.append("one or more output bins receive no model support")
    if window.provenance == "unspecified":
        issues.append(
            "window provenance is unspecified; record mask, estimator, and mock provenance"
        )
    return issues


__all__ = ["PowerWindow", "identity_window", "validate_window_for_inference"]
