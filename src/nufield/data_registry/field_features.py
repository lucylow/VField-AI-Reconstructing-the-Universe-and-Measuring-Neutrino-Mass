"""Per-realization summary feature generation for periodic mesh fields.

This module is intended to construct an interpretable baseline view of simulation cubes. All
transforms are explicit and can be recorded in the experiment lineage. No feature is automatically
considered a sufficient statistic for cosmological parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nufield.features.periodic_power import estimate_periodic_power


def field_summary(
    field: np.ndarray,
    *,
    box_size: float,
    power_bins: int = 24,
    quantiles: tuple[float, ...] = (0.01, 0.1, 0.5, 0.9, 0.99),
    include_gradient: bool = True,
    shot_noise: float | None = None,
    assignment: str = "none",
    deconvolve_assignment: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Return fixed-width summary vector + exact ordered names for a cubic overdensity field."""
    x = np.asarray(field, dtype=float)
    if x.ndim != 3 or len(set(x.shape)) != 1:
        raise ValueError("field must be cubic [N,N,N]")
    if not np.isfinite(x).all():
        raise ValueError("field contains non-finite values")
    if box_size <= 0 or power_bins < 2:
        raise ValueError("invalid box_size or power_bins")
    names = []
    vals = []

    def add(name, value):
        names.append(name)
        vals.append(float(value))

    add("mean", x.mean())
    add("std", x.std())
    add("variance", x.var())
    add("skewness", np.mean(((x - x.mean()) / (x.std() + 1e-12)) ** 3))
    add("excess_kurtosis", np.mean(((x - x.mean()) / (x.std() + 1e-12)) ** 4) - 3.0)
    for q in quantiles:
        if not 0 <= q <= 1:
            raise ValueError("quantile levels must be in [0,1]")
        add(f"quantile_{q:g}", np.quantile(x, q))
    ps = estimate_periodic_power(
        x,
        box_size=box_size,
        n_bins=power_bins,
        shot_noise=shot_noise,
        subtract_shot_noise=False,
        assignment=assignment,
        deconvolve_assignment=deconvolve_assignment,
    )
    for i, (k, p, n) in enumerate(zip(ps.k, ps.power, ps.mode_count)):
        # Empty modes remain explicitly represented as NaN, rather than silently zero filled.
        add(f"pk_bin_{i:03d}_k_{k:.7g}", p)
        add(f"pk_modes_{i:03d}", n)
    if include_gradient:
        grad = [np.gradient(x, axis=ax) for ax in range(3)]
        gmag = np.sqrt(sum(g * g for g in grad))
        add("gradient_rms", np.sqrt(np.mean(gmag * gmag)))
        add("gradient_median", np.median(gmag))
        add("gradient_p95", np.quantile(gmag, 0.95))
    vec = np.asarray(vals, dtype=float)
    if not np.isfinite(vec).all():
        raise ValueError(
            "summary contains non-finite values, likely because one or more P(k) bins are empty; change bins/range explicitly"
        )
    return vec, names


def summarize_field_batch(
    fields: np.ndarray, *, box_size: float, power_bins: int = 24, include_gradient: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Summarize a batch [N,D,H,W]; all fields must share cubic resolution."""
    x = np.asarray(fields)
    if x.ndim != 4 or x.shape[1] != x.shape[2] or x.shape[2] != x.shape[3]:
        raise ValueError("fields must be [N,Ngrid,Ngrid,Ngrid]")
    if len(x) < 1:
        raise ValueError("empty field batch")
    rows = []
    expected_names = None
    for i, field in enumerate(x):
        values, names = field_summary(
            field, box_size=box_size, power_bins=power_bins, include_gradient=include_gradient
        )
        if expected_names is None:
            expected_names = names
        elif expected_names != names:
            raise RuntimeError("internal error: summary feature schema changed")
        rows.append(values)
    return np.stack(rows), expected_names or []


def feature_schema(
    names: list[str], *, source_key: str, box_size: float, preprocessing: dict[str, Any]
) -> dict[str, Any]:
    if not source_key or box_size <= 0:
        raise ValueError("source_key and box_size are required")
    return {
        "schema_version": "1.0",
        "feature_names": list(names),
        "feature_count": len(names),
        "source_key": source_key,
        "box_size": float(box_size),
        "preprocessing": preprocessing,
        "feature_status": "candidate_summary_not_approved_sufficient_statistic",
        "warning": "P(k) normalization and field units must be validated against a trusted implementation.",
    }
