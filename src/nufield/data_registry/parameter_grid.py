"""Parameter-grid and identifiability diagnostics for simulation datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridColumnReport:
    name: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    n_unique_exact: int
    n_unique_tolerance: int
    normalized_entropy: float
    duplicate_fraction: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ParameterGridReport:
    n_rows: int
    parameter_names: tuple[str, ...]
    columns: tuple[GridColumnReport, ...]
    pairwise_correlation: list[list[float]]
    matrix_condition_number: float | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "columns": [x.to_dict() for x in self.columns]}


def _normalized_entropy(x: np.ndarray, bins: int = 12) -> float:
    if np.ptp(x) == 0:
        return 0.0
    counts, _ = np.histogram(x, bins=min(bins, max(2, np.unique(x).size)))
    p = counts[counts > 0] / max(1, counts.sum())
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(max(2, len(counts))))


def inspect_parameter_grid(
    theta: np.ndarray,
    parameter_names: list[str] | tuple[str, ...],
    *,
    tolerance: float = 1e-8,
    target_names: tuple[str, ...] = ("sum_mnu_ev", "Mnu"),
    minimum_unique_target_values: int = 3,
) -> ParameterGridReport:
    """Summarize target support and parameter degeneracies without approving scientific use.

    `n_unique_tolerance` is based on rounding after scale-normalization. It is only a diagnostic;
    never replace the source's actual values with the rounded values.
    """
    x = np.asarray(theta, dtype=float)
    names = tuple(str(n) for n in parameter_names)
    errors: list[str] = []
    warnings: list[str] = []
    if x.ndim != 2:
        raise ValueError("theta must have shape [N,P]")
    if x.shape[1] != len(names):
        raise ValueError("parameter_names length mismatch")
    if not np.isfinite(x).all():
        raise ValueError("theta contains NaN/Inf")
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    if len(x) < 3:
        warnings.append("Fewer than three parameter rows; grid diagnostics are preliminary.")
    columns: list[GridColumnReport] = []
    for j, name in enumerate(names):
        col = x[:, j]
        if np.ptp(col) == 0:
            errors.append(f"{name} is constant; cannot infer sensitivity to it.")
        scale = max(float(np.std(col)), abs(float(np.mean(col))), 1.0)
        rounded = np.round(col / scale / max(tolerance, 1e-15), 0)
        n_tol = int(np.unique(rounded).size)
        counts = np.unique(col, return_counts=True)[1]
        columns.append(
            GridColumnReport(
                name,
                float(col.min()),
                float(col.max()),
                float(col.mean()),
                float(col.std()),
                int(np.unique(col).size),
                n_tol,
                _normalized_entropy(col),
                float(1 - len(counts) / len(col)),
            )
        )
        if np.unique(col).size <= 2:
            warnings.append(
                f"{name} has only {np.unique(col).size} exact values; continuous posterior interpolation may be unsupported."
            )
    for target in target_names:
        if target in names:
            vals = x[:, names.index(target)]
            if np.unique(vals).size < minimum_unique_target_values:
                errors.append(
                    f"Target {target} has fewer than {minimum_unique_target_values} distinct values."
                )
            if len(np.unique(vals)) < 8:
                warnings.append(
                    f"Target {target} has a coarse grid; prefer discrete sensitivity tests or validated interpolation evidence."
                )
    if x.shape[1] > 1 and len(x) > x.shape[1]:
        std = x.std(axis=0)
        nonconstant = std > 0
        safe_std = std.copy()
        safe_std[~nonconstant] = 1.0
        z = (x - x.mean(axis=0)) / safe_std
        # Compute correlations only over nonconstant columns to avoid NaN warnings; constant
        # columns receive zero off-diagonal correlation and unit diagonal for stable JSON output.
        corr = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        np.fill_diagonal(corr, 1.0)
        ix = np.flatnonzero(nonconstant)
        if len(ix) > 1:
            corr_sub = np.corrcoef(z[:, ix], rowvar=False)
            corr[np.ix_(ix, ix)] = np.nan_to_num(corr_sub, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.linalg.svd(z / np.sqrt(max(1, len(x) - 1)), compute_uv=False)
        cond = float(s.max() / max(s.min(), 1e-15)) if s.size else None
        if cond is not None and cond > 1e8:
            warnings.append(
                "Standardized parameter design is ill-conditioned; inspect degeneracies and grid design."
            )
        if np.max(np.abs(corr - np.eye(x.shape[1]))) > 0.98:
            warnings.append("At least one parameter pair has near-perfect linear correlation.")
    else:
        corr = np.eye(x.shape[1])
        cond = None
    return ParameterGridReport(
        len(x),
        names,
        tuple(columns),
        np.asarray(corr).tolist(),
        cond,
        tuple(dict.fromkeys(warnings)),
        tuple(dict.fromkeys(errors)),
    )


def assert_target_support(
    theta: np.ndarray,
    names: list[str],
    *,
    target_name: str,
    prior_bounds: tuple[float, float],
    min_unique: int = 3,
    require_interpolation: bool = False,
) -> dict[str, Any]:
    """Hard preflight for target range/variation; this checks observed support, not physics fidelity."""
    arr = np.asarray(theta, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError("invalid theta/name shapes")
    if target_name not in names:
        raise ValueError(f"target {target_name!r} not in {names}")
    lo, hi = map(float, prior_bounds)
    if not lo < hi:
        raise ValueError("prior bounds must be ordered")
    vals = arr[:, names.index(target_name)]
    n_unique = int(np.unique(vals).size)
    outside = bool(np.any(vals < lo) or np.any(vals > hi))
    holes = np.array([])
    uniq = np.unique(vals)
    if len(uniq) > 1:
        holes = np.diff(uniq)
    uniformity = (
        float(np.std(holes) / np.mean(holes)) if len(holes) and np.mean(holes) > 0 else None
    )
    errors = []
    warnings = []
    if not np.isfinite(arr).all():
        errors.append("parameter matrix contains non-finite values")
    if n_unique < min_unique:
        errors.append(f"target has {n_unique} distinct values; needs at least {min_unique}")
    if outside:
        errors.append("target values lie outside declared prior bounds")
    if vals.min() > lo or vals.max() < hi:
        warnings.append("simulation labels do not span the declared prior bounds")
    if require_interpolation and (n_unique < 8 or uniformity is None or uniformity > 1.0):
        warnings.append(
            "support is sparse/irregular for continuous interpolation; increase targeted simulation coverage"
        )
    return {
        "target_name": target_name,
        "prior_bounds": [lo, hi],
        "data_range": [float(vals.min()), float(vals.max())],
        "n_unique": n_unique,
        "spacing_cv": uniformity,
        "errors": errors,
        "warnings": warnings,
        "eligible_for_preliminary_target_training": not errors,
        "scientific_approval": False,
    }
