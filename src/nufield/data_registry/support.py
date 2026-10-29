"""Parameter-support diagnostics for dataset choice and inference-prior coverage."""

from __future__ import annotations

import numpy as np


def parameter_support_report(
    theta: np.ndarray,
    parameter_names: list[str],
    *,
    prior_low: list[float] | None = None,
    prior_high: list[float] | None = None,
    mass_parameter_name: str = "Mnu",
    n_mass_bins: int = 8,
) -> dict:
    arr = np.asarray(theta, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(parameter_names):
        raise ValueError("theta/name dimensions mismatch")
    if not np.isfinite(arr).all():
        raise ValueError("theta must be finite")
    if n_mass_bins < 2:
        raise ValueError("n_mass_bins must be >=2")
    summary = {}
    for i, name in enumerate(parameter_names):
        vals = arr[:, i]
        summary[name] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "n_unique": int(np.unique(vals).size),
            "constant": bool(np.ptp(vals) <= 1e-12),
        }
    issues = []
    if mass_parameter_name not in parameter_names:
        issues.append(
            {
                "severity": "error",
                "code": "MASS_LABEL_MISSING",
                "message": f"{mass_parameter_name} not in parameter names",
            }
        )
        mass = None
        edges = np.array([])
        counts = np.array([], dtype=int)
    else:
        mass = arr[:, parameter_names.index(mass_parameter_name)]
        unique = np.unique(mass)
        if len(unique) < 3:
            issues.append(
                {
                    "severity": "error",
                    "code": "MASS_NOT_VARYING",
                    "message": f"Only {len(unique)} unique mass value(s); not enough for a continuous posterior benchmark.",
                }
            )
        if len(unique) < n_mass_bins:
            issues.append(
                {
                    "severity": "warning",
                    "code": "MASS_COARSE_GRID",
                    "message": f"Only {len(unique)} unique mass values for {n_mass_bins} requested bins.",
                }
            )
        edges = np.quantile(mass, np.linspace(0, 1, n_mass_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            counts = np.array([len(mass)], dtype=int)
        else:
            counts, _ = np.histogram(mass, bins=edges)
            if (counts == 0).any():
                issues.append(
                    {
                        "severity": "warning",
                        "code": "MASS_EMPTY_BINS",
                        "message": "Some quantile bins are empty after duplicate edges were removed.",
                    }
                )
    if prior_low is not None or prior_high is not None:
        if (
            prior_low is None
            or prior_high is None
            or len(prior_low) != arr.shape[1]
            or len(prior_high) != arr.shape[1]
        ):
            raise ValueError("prior_low/high must both match parameter dimension")
        low = np.asarray(prior_low, dtype=float)
        high = np.asarray(prior_high, dtype=float)
        if not (low < high).all():
            raise ValueError("invalid prior bounds")
        for i, name in enumerate(parameter_names):
            if arr[:, i].min() > low[i] or arr[:, i].max() < high[i]:
                issues.append(
                    {
                        "severity": "warning",
                        "code": f"PRIOR_UNCOVERED_{name}",
                        "message": f"training support [{arr[:, i].min():.6g}, {arr[:, i].max():.6g}] does not cover prior [{low[i]:.6g}, {high[i]:.6g}]",
                    }
                )
    return {
        "n_rows": int(len(arr)),
        "parameters": summary,
        "mass_parameter": mass_parameter_name,
        "mass_bin_edges": edges.tolist(),
        "mass_bin_counts": counts.tolist(),
        "issues": issues,
        "warning": "This checks empirical support only; density near boundaries, simulator validity, priors, and dependence require separate review.",
    }
