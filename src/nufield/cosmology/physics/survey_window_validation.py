"""Sensitivity and invertibility diagnostics for binned survey-window operators.

Window matrices are analysis products, not generic smoothing filters. Their rows/columns encode a
specific estimator, survey mask, binning and normalization. This module can identify numerical issues,
but cannot reconstruct a scientifically correct window from a galaxy catalog alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.window import PowerWindow


@dataclass(frozen=True, slots=True)
class SurveyWindowReport:
    shape: tuple[int, int]
    rank: int
    singular_values: tuple[float, ...]
    condition_number: float
    row_sum_min: float
    row_sum_max: float
    empty_rows: tuple[int, ...]
    dominant_input_for_output: tuple[int, ...]
    max_offdiagonal_fraction: float
    invertibility_recommendation: str
    provenance: str

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "rank": self.rank,
            "singular_values": list(self.singular_values),
            "condition_number": self.condition_number,
            "row_sum_min": self.row_sum_min,
            "row_sum_max": self.row_sum_max,
            "empty_rows": list(self.empty_rows),
            "dominant_input_for_output": list(self.dominant_input_for_output),
            "max_offdiagonal_fraction": self.max_offdiagonal_fraction,
            "invertibility_recommendation": self.invertibility_recommendation,
            "provenance": self.provenance,
            "warning": "A numerically invertible window need not be statistically or physically safe to deconvolve.",
        }


def audit_survey_window(
    window: PowerWindow, *, rank_rtol: float = 1e-10, acceptable_condition: float = 1e6
) -> SurveyWindowReport:
    """Report SVD rank, conditioning, row support and cross-bin mixing."""
    w = np.asarray(window.matrix, float)
    if (
        not np.isfinite(rank_rtol)
        or rank_rtol <= 0
        or not np.isfinite(acceptable_condition)
        or acceptable_condition <= 1
    ):
        raise ValueError("rank_rtol must be positive and acceptable_condition >1")
    singular = np.linalg.svd(w, compute_uv=False)
    rank = (
        int(np.count_nonzero(singular > rank_rtol * singular[0]))
        if singular.size and singular[0] > 0
        else 0
    )
    condition = (
        float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    )
    sums = w.sum(axis=1)
    empty = tuple(int(i) for i in np.flatnonzero(np.max(np.abs(w), axis=1) == 0))
    dominant = tuple(int(i) for i in np.argmax(np.abs(w), axis=1))
    fractions = []
    for row in w:
        total = np.sum(np.abs(row))
        if total > 0:
            # fraction of absolute response in all entries except the dominant input bin
            fractions.append(float((total - np.max(np.abs(row))) / total))
        else:
            fractions.append(float("nan"))
    mix = (
        float(np.nanmax(fractions))
        if fractions and np.any(np.isfinite(fractions))
        else float("nan")
    )
    if empty:
        advice = "blocked: at least one output bin has no theory support"
    elif rank < min(w.shape):
        advice = "do not directly invert: operator is rank deficient; use forward folding or a documented regularized model"
    elif condition > acceptable_condition:
        advice = "avoid naive deconvolution: inversion amplifies small perturbations; forward-fold theory or justify regularization"
    else:
        advice = "full-rank numerically; forward-folding remains preferred unless deconvolution is independently justified"
    return SurveyWindowReport(
        w.shape,
        rank,
        tuple(float(x) for x in singular),
        condition,
        float(sums.min()),
        float(sums.max()),
        empty,
        dominant,
        mix,
        advice,
        window.provenance,
    )


def impulse_responses(window: PowerWindow) -> np.ndarray:
    """Return output response to each unit input basis vector (the window columns)."""
    return window.matrix.copy()


def perturbation_amplification(
    window: PowerWindow,
    theory: np.ndarray,
    perturbation: np.ndarray,
) -> dict[str, float | list[float]]:
    """Quantify how an input-theory perturbation propagates through a fixed window."""
    t, dt = np.asarray(theory, float), np.asarray(perturbation, float)
    if t.shape != (window.matrix.shape[1],) or dt.shape != t.shape or np.any(~np.isfinite(t + dt)):
        raise ValueError("theory and perturbation must match window input dimension and be finite")
    y, dy = window.apply(t), window.apply(dt)
    yin, din = float(np.linalg.norm(t)), float(np.linalg.norm(dt))
    yout, dout = float(np.linalg.norm(y)), float(np.linalg.norm(dy))
    return {
        "input_theory_norm": yin,
        "output_theory_norm": yout,
        "input_perturbation_norm": din,
        "output_perturbation_norm": dout,
        "perturbation_amplification": float(dout / din) if din > 0 else 0.0,
        "relative_output_perturbation": float(dout / max(yout, np.finfo(float).tiny)),
        "windowed_theory": y.tolist(),
        "windowed_perturbation": dy.tolist(),
    }


def regularized_window_inverse(window: PowerWindow, *, tikhonov_lambda: float) -> np.ndarray:
    """Return the Tikhonov inverse W⁺=(WᵀW+λI)⁻¹Wᵀ for explicitly chosen λ>0.

    This is an inverse operator for diagnostic experiments only. A scientifically valid deconvolution
    needs a regularizer/prior and uncertainty propagation justified for the specific estimator.
    """
    w = np.asarray(window.matrix, float)
    if not np.isfinite(tikhonov_lambda) or tikhonov_lambda <= 0:
        raise ValueError("tikhonov_lambda must be finite and positive")
    lhs = w.T @ w + tikhonov_lambda * np.eye(w.shape[1])
    return np.linalg.solve(lhs, w.T)


def window_inverse_resolution_report(
    window: PowerWindow, *, lambdas: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2)
) -> list[dict[str, float]]:
    """Report reconstruction bias and operator norm across a declared Tikhonov lambda grid."""
    if (
        len(lambdas) < 2
        or any(not np.isfinite(v) or v <= 0 for v in lambdas)
        or any(b <= a for a, b in zip(lambdas, lambdas[1:]))
    ):
        raise ValueError("lambdas must be a sorted vector of at least two positive values")
    w = window.matrix
    rows = []
    for lam in lambdas:
        inv = regularized_window_inverse(window, tikhonov_lambda=lam)
        resolution = inv @ w
        rows.append(
            {
                "lambda": float(lam),
                "inverse_operator_norm": float(np.linalg.norm(inv, ord=2)),
                "resolution_trace": float(np.trace(resolution)),
                "resolution_frobenius_error_from_identity": float(
                    np.linalg.norm(resolution - np.eye(w.shape[1]), ord="fro")
                ),
                "forward_reconstruction_residual": float(
                    np.linalg.norm(w @ inv @ w - w, ord="fro")
                ),
            }
        )
    return rows


__all__ = [
    "SurveyWindowReport",
    "audit_survey_window",
    "impulse_responses",
    "perturbation_amplification",
    "regularized_window_inverse",
    "window_inverse_resolution_report",
]
