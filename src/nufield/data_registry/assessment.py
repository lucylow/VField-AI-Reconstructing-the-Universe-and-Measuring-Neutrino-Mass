"""Array-level assessment for dataset candidates before model fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class AssessmentFinding:
    severity: str
    code: str
    message: str
    recommendation: str


@dataclass
class DatasetAssessment:
    n_rows: int
    observation_shape: list[int]
    parameter_shape: list[int]
    parameter_names: list[str]
    findings: list[AssessmentFinding] = field(default_factory=list)
    parameter_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    can_train_mnu_posterior: bool = False
    scientific_approval: bool = False

    @property
    def errors(self) -> list[AssessmentFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[AssessmentFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_errors"] = len(self.errors)
        d["n_warnings"] = len(self.warnings)
        d["scientific_approval"] = False  # this utility never grants scientific approval
        return d


def assess_arrays(
    observations: np.ndarray,
    theta: np.ndarray,
    *,
    parameter_names: list[str],
    simulation_ids: np.ndarray | None = None,
    mass_parameter_name: str = "Mnu",
    minimum_mass_values: int = 3,
    expected_parameter_ranges: dict[str, tuple[float, float]] | None = None,
) -> DatasetAssessment:
    """Check shapes, numeric validity, label variation, and plausible parameter coverage.

    Passing means only that arrays meet preliminary data-contract requirements. It does not certify
    the simulator, independence, prior completeness, physical units, or posterior calibration.
    """
    x = np.asarray(observations)
    y = np.asarray(theta)
    findings: list[AssessmentFinding] = []
    if x.ndim < 2:
        findings.append(
            AssessmentFinding(
                "error",
                "OBS_NDIM",
                "observations must have sample axis plus feature/field axes",
                "Use shape [N,F] or [N,C,...].",
            )
        )
    if y.ndim != 2:
        findings.append(
            AssessmentFinding(
                "error",
                "THETA_NDIM",
                "theta must be a 2-D array [N,P]",
                "Store one parameter vector per simulation.",
            )
        )
    if x.ndim and y.ndim and len(x) != len(y):
        findings.append(
            AssessmentFinding(
                "error",
                "ROW_MISMATCH",
                "observation and parameter row counts differ",
                "Align by simulation ID and verify joins.",
            )
        )
    if y.ndim == 2 and len(parameter_names) != y.shape[1]:
        findings.append(
            AssessmentFinding(
                "error",
                "PARAMETER_NAME_MISMATCH",
                "parameter_names length does not match theta columns",
                "Document column order and units in the dataset card.",
            )
        )
    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        findings.append(
            AssessmentFinding(
                "error",
                "NON_NUMERIC",
                "arrays must be numeric",
                "Convert only documented physical columns; never coerce IDs into numeric features.",
            )
        )
    else:
        if not np.isfinite(x).all():
            findings.append(
                AssessmentFinding(
                    "error",
                    "NONFINITE_X",
                    "observations contain NaN or infinity",
                    "Resolve missing/corrupt values explicitly; do not silently replace them.",
                )
            )
        if not np.isfinite(y).all():
            findings.append(
                AssessmentFinding(
                    "error",
                    "NONFINITE_THETA",
                    "parameters contain NaN or infinity",
                    "Drop or repair simulations only with a documented reason.",
                )
            )
    if simulation_ids is not None:
        ids = np.asarray(simulation_ids).astype(str)
        if x.ndim and len(ids) != len(x):
            findings.append(
                AssessmentFinding(
                    "error",
                    "ID_LENGTH",
                    "simulation ID count differs from rows",
                    "Join IDs with the field arrays and parameter table.",
                )
            )
        elif len(set(ids.tolist())) != len(ids):
            findings.append(
                AssessmentFinding(
                    "warning",
                    "DUPLICATE_IDS",
                    "simulation IDs are not unique",
                    "Check whether rows are correlated crops/views from the same simulation; split by parent ID.",
                )
            )
    summary: dict[str, dict[str, float]] = {}
    if (
        y.ndim == 2
        and len(parameter_names) == y.shape[1]
        and np.issubdtype(y.dtype, np.number)
        and np.isfinite(y).all()
    ):
        for j, name in enumerate(parameter_names):
            col = y[:, j].astype(float)
            unique = int(np.unique(col).size)
            summary[name] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std()),
                "n_unique": float(unique),
            }
            if unique < 2:
                findings.append(
                    AssessmentFinding(
                        "error",
                        f"CONSTANT_PARAMETER_{name}",
                        f"parameter {name!r} is constant",
                        "Cannot learn sensitivity to a constant target; choose a subset varying this parameter.",
                    )
                )
            if unique < 10 and len(col) >= 50:
                findings.append(
                    AssessmentFinding(
                        "warning",
                        f"SPARSE_PARAMETER_{name}",
                        f"parameter {name!r} has only {unique} distinct values",
                        "Check whether the task needs a dense continuous posterior or a discrete model comparison.",
                    )
                )
            if expected_parameter_ranges and name in expected_parameter_ranges:
                lo, hi = expected_parameter_ranges[name]
                if col.min() < lo or col.max() > hi:
                    findings.append(
                        AssessmentFinding(
                            "warning",
                            f"RANGE_{name}",
                            f"observed range [{col.min()}, {col.max()}] exceeds declared [{lo}, {hi}]",
                            "Reconcile units, priors, and dataset-card range.",
                        )
                    )
    mnu_ok = False
    if mass_parameter_name in parameter_names and y.ndim == 2 and np.isfinite(y).all():
        j = parameter_names.index(mass_parameter_name)
        mnu_vals = np.unique(y[:, j])
        mnu_ok = len(mnu_vals) >= minimum_mass_values and float(mnu_vals.max() - mnu_vals.min()) > 0
        if not mnu_ok:
            findings.append(
                AssessmentFinding(
                    "error",
                    "MNU_NOT_IDENTIFIABLE",
                    f"{mass_parameter_name} has {len(mnu_vals)} unique value(s)",
                    "Choose a documented neutrino-varying suite; do not infer Mnu from a fixed-mass catalog.",
                )
            )
        elif len(mnu_vals) < 8:
            findings.append(
                AssessmentFinding(
                    "warning",
                    "MNU_COARSE_GRID",
                    f"{mass_parameter_name} has only {len(mnu_vals)} unique values",
                    "A coarse grid may support sensitivity tests but not a smooth amortized posterior without additional validated simulations.",
                )
            )
    else:
        findings.append(
            AssessmentFinding(
                "error",
                "MNU_LABEL_MISSING",
                f"parameter {mass_parameter_name!r} not found",
                "Map the exact mass label and units from the source metadata.",
            )
        )
    if len(x) < 200:
        findings.append(
            AssessmentFinding(
                "warning",
                "SMALL_N",
                f"only {len(x)} rows are available",
                "Treat as a pipeline smoke test; use simulation-based calibration with many independent simulations for research conclusions.",
            )
        )
    if not any(f.severity == "error" for f in findings):
        findings.append(
            AssessmentFinding(
                "info",
                "ARRAYS_PASS",
                "basic array-contract checks passed",
                "Next verify independent splits, forward-model validity, simulator coverage, and posterior calibration.",
            )
        )
    return DatasetAssessment(
        n_rows=int(len(x)) if x.ndim else 0,
        observation_shape=list(x.shape[1:]) if x.ndim > 1 else [],
        parameter_shape=list(y.shape[1:]) if y.ndim > 1 else [],
        parameter_names=list(parameter_names),
        findings=findings,
        parameter_summary=summary,
        can_train_mnu_posterior=mnu_ok and not any(f.severity == "error" for f in findings),
        scientific_approval=False,
    )
