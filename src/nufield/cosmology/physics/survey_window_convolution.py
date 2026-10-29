"""Forward-folding theory bandpowers through a linear survey/estimator window.

A survey window is an operator defined by a particular mask, estimator, binning, selection and convention.
This module provides math and validation utilities; it cannot infer a real survey window from a catalog.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.window import PowerWindow


@dataclass(frozen=True, slots=True)
class WindowedTheory:
    theory_native: np.ndarray
    theory_windowed: np.ndarray
    covariance_windowed: np.ndarray | None
    model_label: str
    warnings: tuple[str, ...]


def fold_theory_through_window(
    theory_native: np.ndarray,
    window: PowerWindow,
    *,
    covariance_native: np.ndarray | None = None,
    model_label: str = "unspecified",
) -> WindowedTheory:
    """Calculate y=W t and optionally C_y=W C_t W^T with strict dimension checks."""
    t = np.asarray(theory_native, float)
    if t.ndim != 1 or t.size != window.matrix.shape[1] or np.any(~np.isfinite(t)):
        raise ValueError("theory_native must be finite and match the window's input dimension")
    y = window.matrix @ t
    cy = None
    if covariance_native is not None:
        c = np.asarray(covariance_native, float)
        if (
            c.shape != (t.size, t.size)
            or np.any(~np.isfinite(c))
            or not np.allclose(c, c.T, atol=1e-10)
        ):
            raise ValueError(
                "covariance_native must be finite symmetric with native-theory dimensions"
            )
        cy = window.matrix @ c @ window.matrix.T
        cy = 0.5 * (cy + cy.T)
    warnings = []
    if window.provenance == "unspecified":
        warnings.append(
            "window provenance is unspecified; result is a software test, not a survey prediction"
        )
    if np.any(~np.isfinite(y)):
        raise ArithmeticError("window application produced non-finite bandpowers")
    return WindowedTheory(t.copy(), y, cy, str(model_label), tuple(warnings))


def propagate_windowed_covariance(window: PowerWindow, covariance_native: np.ndarray) -> np.ndarray:
    """Propagate covariance through the exact linear operator W C W^T."""
    c = np.asarray(covariance_native, float)
    if c.shape != (window.matrix.shape[1], window.matrix.shape[1]) or np.any(~np.isfinite(c)):
        raise ValueError("native covariance shape mismatch or non-finite entries")
    if not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("native covariance must be symmetric")
    out = window.matrix @ c @ window.matrix.T
    return 0.5 * (out + out.T)


def validate_window_operator(window: PowerWindow, *, tolerance: float = 1e-8) -> dict[str, object]:
    """Report dimensions, rank and response to a constant input; does not assert normalization."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    w = window.matrix
    singular = np.linalg.svd(w, compute_uv=False)
    row_sums = w.sum(axis=1)
    return {
        "input_bins": int(w.shape[1]),
        "output_bins": int(w.shape[0]),
        "rank": int(np.linalg.matrix_rank(w, tol=tolerance)),
        "condition_number": float(singular[0] / singular[-1])
        if singular.size and singular[-1] > tolerance
        else float("inf"),
        "constant_input_response": row_sums.tolist(),
        "rows_sum_to_one": bool(np.allclose(row_sums, 1.0, atol=tolerance, rtol=0)),
        "has_empty_rows": bool(np.any(np.max(np.abs(w), axis=1) <= tolerance)),
        "provenance": window.provenance,
    }


__all__ = [
    "WindowedTheory",
    "fold_theory_through_window",
    "propagate_windowed_covariance",
    "validate_window_operator",
]
