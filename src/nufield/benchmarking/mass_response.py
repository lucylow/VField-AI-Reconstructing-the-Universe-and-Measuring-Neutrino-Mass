"""Matched-realization diagnostics for neutrino-mass-sensitive summary statistics.

The module answers a necessary early question: does the chosen dataset encode a measurable response
in the feature space? It does not establish a cosmological constraint. Use matched phases/initial
conditions and fixed nuisance conventions whenever possible; unmatched noise can swamp the response.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MassResponseReport:
    low_mass_ev: float
    high_mass_ev: float
    n_matched_pairs: int
    feature_count: int
    mean_difference: list[float]
    standard_error: list[float]
    relative_change_fraction: list[float]
    paired_z_score: list[float]
    strongest_abs_z_feature_indices: list[int]
    paired_design: bool
    findings: list[str]
    disclaimer: str = (
        "Matched-response diagnostic only; not a posterior constraint or significance claim."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_mass_response(
    features: np.ndarray,
    sum_mnu_ev: np.ndarray,
    phase_ids: np.ndarray,
    *,
    low_mass_ev: float,
    high_mass_ev: float,
    strata_ids: np.ndarray | None = None,
    relative_floor: float = 1e-12,
    min_pairs: int = 3,
    atol: float = 1e-8,
) -> MassResponseReport:
    """Estimate high-minus-low feature response with paired phases and optional nuisance strata.

    Each matching key `(stratum, phase_id)` must have exactly one row at each requested mass. Extra
    or duplicated rows are ignored/rejected rather than averaged silently. For physics, `strata_ids`
    should encode other matched cosmological/nuisance choices, and the simulation provenance must
    establish that the phase IDs really are paired across the mass variants.
    """
    x = np.asarray(features, dtype=np.float64)
    m = np.asarray(sum_mnu_ev, dtype=np.float64).reshape(-1)
    phase = np.asarray(phase_ids).astype(str).reshape(-1)
    strata = (
        np.asarray(strata_ids).astype(str).reshape(-1)
        if strata_ids is not None
        else np.full(len(m), "all")
    )
    if x.ndim != 2 or len(x) != len(m) or len(phase) != len(m) or len(strata) != len(m):
        raise ValueError("features [N,K], mass, phase_ids and strata_ids must have aligned rows")
    if not np.isfinite(x).all() or not np.isfinite(m).all():
        raise ValueError("features and masses must be finite")
    if low_mass_ev >= high_mass_ev or min_pairs < 2 or relative_floor <= 0:
        raise ValueError("invalid mass bounds/min_pairs/floor")
    low_rows = np.flatnonzero(np.isclose(m, low_mass_ev, atol=atol, rtol=0))
    high_rows = np.flatnonzero(np.isclose(m, high_mass_ev, atol=atol, rtol=0))
    if not len(low_rows) or not len(high_rows):
        raise ValueError("requested low/high mass level missing from dataset")

    def index_rows(indices):
        result = {}
        for i in indices:
            key = (str(strata[i]), str(phase[i]))
            if key in result:
                raise ValueError(f"duplicate row for matched key {key!r} at a target mass")
            result[key] = i
        return result

    lo = index_rows(low_rows)
    hi = index_rows(high_rows)
    common = sorted(set(lo) & set(hi))
    findings = []
    if len(common) < min_pairs:
        findings.append(
            f"Only {len(common)} matched phase/stratum pairs; at least {min_pairs} were requested."
        )
    if len(common) < 2:
        raise ValueError(
            "at least two matched pairs are required to estimate a response uncertainty"
        )
    diffs = np.stack([x[hi[k]] - x[lo[k]] for k in common])
    mean = diffs.mean(axis=0)
    sd = diffs.std(axis=0, ddof=1) if len(diffs) >= 2 else np.full(x.shape[1], np.nan)
    se = sd / np.sqrt(len(diffs))
    z = mean / np.maximum(se, relative_floor)
    base = np.stack([x[lo[k]] for k in common]).mean(axis=0)
    rel = mean / np.maximum(np.abs(base), relative_floor)
    if not np.isfinite(z).all():
        findings.append(
            "Some feature response z-scores are nonfinite due to insufficient pair count or numerical degeneracy."
        )
    top = np.argsort(np.nan_to_num(np.abs(z), nan=-np.inf))[::-1][: min(20, x.shape[1])]
    if np.nanmax(np.abs(z)) < 1.0:
        findings.append(
            "No feature has |paired mean / standard error| >= 1; first inspect sample count and matched-design validity before increasing model complexity."
        )
    if len(common) < 10:
        findings.append(
            "Fewer than 10 matched pairs: uncertainty on feature response is likely unstable; report pair bootstrap intervals."
        )
    return MassResponseReport(
        low_mass_ev=float(low_mass_ev),
        high_mass_ev=float(high_mass_ev),
        n_matched_pairs=len(common),
        feature_count=int(x.shape[1]),
        mean_difference=mean.tolist(),
        standard_error=se.tolist(),
        relative_change_fraction=rel.tolist(),
        paired_z_score=z.tolist(),
        strongest_abs_z_feature_indices=top.astype(int).tolist(),
        paired_design=True,
        findings=findings,
    )


def response_table_by_mass(
    features: np.ndarray,
    mass: np.ndarray,
    *,
    reference_mass_ev: float,
    feature_names: list[str] | None = None,
    atol: float = 1e-8,
) -> list[dict[str, Any]]:
    """Return mean/std and fractional shift of each feature relative to a reference mass bin.

    This unpaired descriptive table is more vulnerable to realization noise than
    `paired_mass_response`; use it for an initial view only.
    """
    x = np.asarray(features, dtype=float)
    m = np.asarray(mass, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) != len(m) or not np.isfinite(x).all() or not np.isfinite(m).all():
        raise ValueError("features [N,K] and finite mass labels must align")
    names = feature_names or [f"feature_{i}" for i in range(x.shape[1])]
    if len(names) != x.shape[1]:
        raise ValueError("feature_names length mismatch")
    ref = np.flatnonzero(np.isclose(m, reference_mass_ev, atol=atol, rtol=0))
    if not len(ref):
        raise ValueError("reference mass not present")
    ref_mean = x[ref].mean(axis=0)
    table = []
    for value in np.unique(m):
        idx = np.flatnonzero(np.isclose(m, value, atol=atol, rtol=0))
        mean = x[idx].mean(axis=0)
        std = x[idx].std(axis=0, ddof=1) if len(idx) > 1 else np.full(x.shape[1], np.nan)
        for j, name in enumerate(names):
            table.append(
                {
                    "sum_mnu_ev": float(value),
                    "feature": name,
                    "n_rows": int(len(idx)),
                    "mean": float(mean[j]),
                    "std": float(std[j]),
                    "fractional_shift_vs_reference": float(
                        (mean[j] - ref_mean[j]) / max(abs(ref_mean[j]), 1e-12)
                    ),
                }
            )
    return table
