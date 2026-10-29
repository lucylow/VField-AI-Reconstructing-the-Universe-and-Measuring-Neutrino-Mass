"""Diagnostics for whether observations contain distinguishable parameter information.

These are exploratory diagnostics, not a substitute for posterior calibration or a physical
identifiability analysis. In particular, linear feature correlations can miss nonlinear signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IdentifiabilityReport:
    parameter_names: list[str]
    target_name: str
    n_simulations: int
    target_unique_values: int
    target_range: list[float]
    target_std: float
    feature_count: int
    feature_correlations: list[float]
    strongest_feature_indices: list[int]
    linear_r2_in_sample: float | None
    standardized_design_condition_number: float | None
    parameter_correlation_matrix: list[list[float]]
    findings: list[str]
    disclaimer: str = (
        "Exploratory linear diagnostics only; not physical or posterior identifiability approval."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or np.std(a) <= 1e-14 or np.std(b) <= 1e-14:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def analyze_identifiability(
    observations: np.ndarray,
    theta: np.ndarray,
    *,
    parameter_names: list[str],
    target_name: str = "sum_mnu_ev",
    max_features: int = 5000,
) -> IdentifiabilityReport:
    """Estimate target/feature associations and linear design degeneracy.

    Raw fields should first be represented by a declared summary or dimensionality reduction fitted
    only on training simulations. If the observation tensor is multidimensional, flattening happens
    here solely for a coarse exploratory diagnostic, and a deterministic feature cap is applied.
    """
    x = np.asarray(observations, dtype=np.float64)
    y = np.asarray(theta, dtype=np.float64)
    if x.ndim < 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected aligned observations [N,...] and theta [N,P]")
    if y.shape[1] != len(parameter_names):
        raise ValueError("parameter_names must match theta columns")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Inputs must be finite")
    if target_name not in parameter_names:
        raise ValueError(f"target_name {target_name!r} missing from parameter_names")
    if max_features < 1:
        raise ValueError("max_features must be positive")
    x = x.reshape(len(x), -1)
    if x.shape[1] > max_features:
        # Stable evenly-spaced selection avoids an arbitrary random subset.
        keep = np.linspace(0, x.shape[1] - 1, max_features, dtype=int)
        x = x[:, keep]
    target_index = parameter_names.index(target_name)
    target = y[:, target_index]
    std = x.std(axis=0)
    useful = std > 1e-12
    x_use = x[:, useful]
    corr = [_corr(x_use[:, j], target) for j in range(x_use.shape[1])]
    order = np.argsort(np.abs(corr))[::-1]
    r2: float | None = None
    condition: float | None = None
    findings: list[str] = []
    if np.unique(target).size <= 1:
        findings.append(
            "Target is constant: a conditional estimator cannot learn a target response from this dataset."
        )
    if np.unique(target).size < 5:
        findings.append(
            "Target has fewer than five unique values; continuous-target inference may be poorly identified."
        )
    if not x_use.shape[1]:
        findings.append(
            "All sampled features are constant; no linear target signal can be estimated."
        )
    elif len(target) >= 3:
        xz = (x_use - x_use.mean(axis=0)) / np.maximum(x_use.std(axis=0), 1e-12)
        # Avoid a huge P x P normal matrix. SVD condition on standardized design is stable up to cap.
        design = np.column_stack([np.ones(len(xz)), xz])
        if design.shape[1] <= min(1500, max(10, len(target) * 2)):
            singular = np.linalg.svd(design, compute_uv=False)
            condition = float(singular.max() / max(singular.min(), 1e-14))
        else:
            condition = None
        beta, *_ = np.linalg.lstsq(design, target, rcond=1e-8)
        pred = design @ beta
        denom = float(np.sum((target - target.mean()) ** 2))
        r2 = float(1 - np.sum((target - pred) ** 2) / denom) if denom > 1e-14 else 0.0
        if condition is not None and condition > 1e8:
            findings.append(
                "Standardized design is severely ill-conditioned; feature degeneracies may destabilize linear summaries."
            )
        if r2 < 0.01:
            findings.append(
                "Linear feature model explains less than 1% of in-sample target variance; test nonlinear signal and simulator design before scaling model complexity."
            )
        if r2 > 0.98 and len(target) < 100:
            findings.append(
                "Near-perfect small-sample linear predictability can signal target leakage or a synthetic shortcut; audit provenance and splits."
            )
    pcorr = np.corrcoef(y.T) if y.shape[1] > 1 else np.ones((1, 1))
    pcorr = np.nan_to_num(pcorr, nan=0.0, posinf=0.0, neginf=0.0)
    return IdentifiabilityReport(
        parameter_names=list(parameter_names),
        target_name=target_name,
        n_simulations=len(y),
        target_unique_values=int(np.unique(target).size),
        target_range=[float(target.min()), float(target.max())],
        target_std=float(target.std()),
        feature_count=int(x.shape[1]),
        feature_correlations=[float(v) for v in corr],
        strongest_feature_indices=[
            int(np.flatnonzero(useful)[i]) for i in order[: min(20, len(order))]
        ],
        linear_r2_in_sample=r2,
        standardized_design_condition_number=condition,
        parameter_correlation_matrix=pcorr.tolist(),
        findings=findings,
    )
