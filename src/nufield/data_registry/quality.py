"""Robust, machine-readable dataset quality audits before AI training.

This module inspects numeric arrays and metadata contracts. It is deliberately not a scientific
approval engine: a clean dataset can still have invalid physics, wrong units, or simulator bias.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QualityFinding:
    severity: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetQualityReport:
    dataset_id: str
    n_rows: int
    observation_shape: list[int]
    theta_shape: list[int]
    parameter_names: list[str]
    finite_observation_fraction: float
    finite_theta_fraction: float
    n_unique_groups: int | None
    duplicate_observation_rows: int | None
    findings: list[QualityFinding]
    warnings: list[str]
    science_approval: bool = False

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    @property
    def can_attempt_training(self) -> bool:
        return not self.has_errors

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["has_errors"] = self.has_errors
        out["can_attempt_training"] = self.can_attempt_training
        out["science_approval"] = False
        return out


def _hash_numeric_row(row: np.ndarray) -> bytes:
    """Hash a contiguous row; only used on small/medium summary matrices by default."""
    return hashlib.blake2b(np.ascontiguousarray(row).view(np.uint8), digest_size=12).digest()


def audit_numeric_dataset(
    observations: np.ndarray,
    theta: np.ndarray,
    *,
    parameter_names: list[str] | tuple[str, ...],
    dataset_id: str = "unspecified",
    simulation_ids: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    duplicate_row_scan_limit: int = 20_000,
    constant_feature_tolerance: float = 1e-12,
) -> DatasetQualityReport:
    """Audit shapes, labels, finiteness, grouping and obvious degeneracy.

    Inputs may be feature vectors or raw fields. Duplicate scans are skipped for very large sample
    counts and are an exact-row check, not a claim about physical duplicate simulations.
    """
    x = np.asarray(observations)
    y = np.asarray(theta)
    names = [str(n) for n in parameter_names]
    findings: list[QualityFinding] = []
    warnings: list[str] = [
        "This audit checks data hygiene and coarse support only; it does not validate the simulator.",
        "A good report does not establish neutrino-mass identifiability or posterior calibration.",
    ]
    if x.ndim < 2:
        findings.append(
            QualityFinding(
                "error", "OBSERVATION_RANK", "Observations must be [N,...].", {"ndim": x.ndim}
            )
        )
    if y.ndim != 2:
        findings.append(
            QualityFinding("error", "THETA_RANK", "theta must have shape [N,P].", {"ndim": y.ndim})
        )
    n = int(x.shape[0]) if x.ndim >= 1 else 0
    if y.ndim == 2 and y.shape[0] != n:
        findings.append(
            QualityFinding(
                "error",
                "ROW_MISMATCH",
                "Observation and theta row counts differ.",
                {"n_x": n, "n_theta": int(y.shape[0])},
            )
        )
    if y.ndim == 2 and y.shape[1] != len(names):
        findings.append(
            QualityFinding(
                "error",
                "PARAMETER_NAME_MISMATCH",
                "Parameter names do not match theta columns.",
                {"theta_columns": int(y.shape[1]), "names": len(names)},
            )
        )
    if len(set(names)) != len(names):
        findings.append(
            QualityFinding("error", "DUPLICATE_PARAMETER_NAMES", "Parameter names must be unique.")
        )
    x_finite = (
        np.isfinite(x) if np.issubdtype(x.dtype, np.number) else np.zeros(x.shape, dtype=bool)
    )
    y_finite = (
        np.isfinite(y) if np.issubdtype(y.dtype, np.number) else np.zeros(y.shape, dtype=bool)
    )
    x_frac = float(x_finite.mean()) if x.size else 0.0
    y_frac = float(y_finite.mean()) if y.size else 0.0
    if x_frac < 1:
        findings.append(
            QualityFinding(
                "error",
                "NONFINITE_OBSERVATIONS",
                "Observation data contain NaN/Inf or nonnumeric values.",
                {"finite_fraction": x_frac},
            )
        )
    if y_frac < 1:
        findings.append(
            QualityFinding(
                "error",
                "NONFINITE_PARAMETERS",
                "Parameter labels contain NaN/Inf or nonnumeric values.",
                {"finite_fraction": y_frac},
            )
        )

    groups = None if group_ids is None else np.asarray(group_ids).astype(str).reshape(-1)
    ids = None if simulation_ids is None else np.asarray(simulation_ids).astype(str).reshape(-1)
    if groups is not None and len(groups) != n:
        findings.append(
            QualityFinding("error", "GROUP_ROW_MISMATCH", "group_ids do not align with rows.")
        )
        groups = None
    if ids is not None and len(ids) != n:
        findings.append(
            QualityFinding("error", "ID_ROW_MISMATCH", "simulation_ids do not align with rows.")
        )
        ids = None
    n_groups = int(np.unique(groups).size) if groups is not None else None
    if groups is None:
        warnings.append(
            "No parent group IDs supplied; split leakage cannot be audited at parent-realization level."
        )
    if ids is None:
        warnings.append(
            "No simulation IDs supplied; duplicate realization IDs cannot be inspected."
        )
    elif len(np.unique(ids)) != len(ids):
        findings.append(
            QualityFinding(
                "warning",
                "REPEATED_SIMULATION_ID",
                "Simulation IDs repeat; verify whether rows are multiple views of the same realization.",
                {"n_rows": n, "n_unique_ids": int(np.unique(ids).size)},
            )
        )

    duplicate_count: int | None = None
    if x.ndim == 2 and n <= duplicate_row_scan_limit and x_frac == 1:
        hashes = [_hash_numeric_row(row) for row in x]
        duplicate_count = int(n - len(set(hashes)))
        if duplicate_count:
            findings.append(
                QualityFinding(
                    "warning",
                    "DUPLICATE_OBSERVATION_ROWS",
                    "Exact duplicate feature rows found; determine whether expected or a generation bug.",
                    {"extra_duplicate_rows": duplicate_count},
                )
            )
    elif x.ndim != 2:
        warnings.append(
            "Exact duplicate-row scan was skipped for field tensors; use provenance IDs and physical checks instead."
        )
    else:
        warnings.append(
            "Exact duplicate-row scan skipped because the sample count exceeds the configured limit."
        )

    if x.ndim == 2 and x.shape[0] and x_frac == 1:
        std = np.std(x.astype(np.float64), axis=0)
        constant = np.flatnonzero(std <= constant_feature_tolerance)
        if constant.size:
            findings.append(
                QualityFinding(
                    "warning",
                    "CONSTANT_FEATURES",
                    "Some observation features are constant and may be uninformative.",
                    {"count": int(constant.size), "indices": constant[:100].astype(int).tolist()},
                )
            )
    if y.ndim == 2 and y.shape[0] and y_frac == 1:
        y64 = y.astype(np.float64)
        for j, name in enumerate(names[: y.shape[1]]):
            uniq = int(np.unique(y64[:, j]).size)
            std = float(np.std(y64[:, j]))
            if uniq <= 1 or std <= 1e-12:
                findings.append(
                    QualityFinding(
                        "error",
                        "CONSTANT_TARGET",
                        f"Target {name!r} has no observed variation.",
                        {"parameter": name, "unique": uniq, "std": std},
                    )
                )
            elif uniq < 5:
                findings.append(
                    QualityFinding(
                        "warning",
                        "SPARSE_TARGET_GRID",
                        f"Target {name!r} has only {uniq} distinct values; continuous posterior generalization may be weak.",
                        {"parameter": name, "unique": uniq},
                    )
                )

    meta = dict(metadata or {})
    if not meta.get("source_release"):
        warnings.append("Metadata lacks source_release; write an immutable release/version label.")
    if not meta.get("units"):
        warnings.append(
            "Metadata lacks units; document physical units and h-convention before fitting."
        )
    if not meta.get("physics_level"):
        warnings.append(
            "Metadata lacks physics_level (e.g. toy/prototype/validated); status must be explicit."
        )
    physics_level = str(meta.get("physics_level", "unknown")).lower()
    if physics_level in {"toy", "synthetic", "prototype", "unknown"}:
        warnings.append(
            f"physics_level={physics_level!r} is not evidence of a validated physical forward model."
        )

    return DatasetQualityReport(
        dataset_id=str(dataset_id),
        n_rows=n,
        observation_shape=list(x.shape),
        theta_shape=list(y.shape),
        parameter_names=names,
        finite_observation_fraction=x_frac,
        finite_theta_fraction=y_frac,
        n_unique_groups=n_groups,
        duplicate_observation_rows=duplicate_count,
        findings=findings,
        warnings=warnings,
        science_approval=False,
    )
