"""Audit whether requested Fourier scales lie inside declared solver, box and model domains.

The audit operates on limits supplied by the user or a validated backend. It deliberately does not guess
where the matter field becomes nonlinear or what k range a survey supports. Those limits depend on
redshift, cosmology, tracer, estimator, simulation resolution, and the target precision.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class KDomainAudit:
    requested_k_min_mpc_inv: float
    requested_k_max_mpc_inv: float
    fundamental_mode_mpc_inv: float
    axis_nyquist_mpc_inv: float
    solver_k_max_mpc_inv: float | None
    declared_linear_k_max_mpc_inv: float | None
    declared_survey_k_max_mpc_inv: float | None
    requested_k_inside_fundamental_to_nyquist: bool
    requested_k_inside_solver: bool | None
    requested_k_inside_declared_linear_range: bool | None
    requested_k_inside_survey_range: bool | None
    grid_size: int
    box_size_mpc: float
    warnings: tuple[str, ...]
    valid_for_claimed_use: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["units"] = {"k": "Mpc^-1", "box_size": "Mpc"}
        out["interpretation"] = (
            "A range check, not proof that the estimator or forward model is unbiased."
        )
        return out


def audit_k_domain(
    k_mpc_inv: np.ndarray,
    *,
    box_size_mpc: float,
    grid_size: int,
    solver_k_max_mpc_inv: float | None = None,
    declared_linear_k_max_mpc_inv: float | None = None,
    declared_survey_k_max_mpc_inv: float | None = None,
    require_linear: bool = False,
    require_survey_support: bool = False,
    nyquist_fraction: float = 0.8,
) -> KDomainAudit:
    """Check requested k against a periodic grid and any explicitly supplied domain limits."""
    k = np.asarray(k_mpc_inv, float)
    if (
        k.ndim != 1
        or len(k) < 2
        or np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError(
            "k_mpc_inv must be a positive finite increasing vector with at least two values"
        )
    if not math.isfinite(box_size_mpc) or box_size_mpc <= 0 or grid_size < 4:
        raise ValueError("box_size_mpc must be positive finite and grid_size>=4")
    if not (0 < nyquist_fraction <= 1):
        raise ValueError("nyquist_fraction must lie in (0,1]")
    limits = (solver_k_max_mpc_inv, declared_linear_k_max_mpc_inv, declared_survey_k_max_mpc_inv)
    if any(v is not None and (not math.isfinite(v) or v <= 0) for v in limits):
        raise ValueError("declared k limits must be positive and finite")
    fundamental = 2 * math.pi / box_size_mpc
    nyquist = nyquist_fraction * math.pi * grid_size / box_size_mpc
    inside_grid = bool(k[0] >= fundamental * (1 - 1e-8) and k[-1] <= nyquist * (1 + 1e-8))
    inside_solver = (
        None if solver_k_max_mpc_inv is None else bool(k[-1] <= solver_k_max_mpc_inv * (1 + 1e-10))
    )
    inside_linear = (
        None
        if declared_linear_k_max_mpc_inv is None
        else bool(k[-1] <= declared_linear_k_max_mpc_inv * (1 + 1e-10))
    )
    inside_survey = (
        None
        if declared_survey_k_max_mpc_inv is None
        else bool(k[-1] <= declared_survey_k_max_mpc_inv * (1 + 1e-10))
    )
    warnings = []
    if k[0] < fundamental:
        warnings.append(
            "Requested minimum k is below the box fundamental mode; those modes cannot be represented by this periodic box."
        )
    if k[-1] > nyquist:
        warnings.append(
            "Requested maximum k exceeds the conservative grid Nyquist limit; aliasing/discretization can dominate."
        )
    if inside_solver is False:
        warnings.append(
            "Requested maximum k exceeds the declared solver support; do not extrapolate silently."
        )
    if inside_linear is False:
        warnings.append("Requested maximum k exceeds the declared linear-theory validity range.")
    if inside_survey is False:
        warnings.append("Requested maximum k exceeds the declared survey/estimator support.")
    if require_linear and declared_linear_k_max_mpc_inv is None:
        warnings.append("Linear validity was required but no linear k limit was supplied.")
    if require_survey_support and declared_survey_k_max_mpc_inv is None:
        warnings.append("Survey support was required but no survey k limit was supplied.")
    valid = inside_grid and inside_solver is not False
    if require_linear:
        valid = valid and inside_linear is True
    if require_survey_support:
        valid = valid and inside_survey is True
    return KDomainAudit(
        float(k[0]),
        float(k[-1]),
        fundamental,
        nyquist,
        solver_k_max_mpc_inv,
        declared_linear_k_max_mpc_inv,
        declared_survey_k_max_mpc_inv,
        inside_grid,
        inside_solver,
        inside_linear,
        inside_survey,
        int(grid_size),
        float(box_size_mpc),
        tuple(warnings),
        bool(valid),
    )


def suggest_log_k_grid(
    k_min_mpc_inv: float,
    k_max_mpc_inv: float,
    n_samples: int,
    *,
    box_size_mpc: float | None = None,
    grid_size: int | None = None,
    nyquist_fraction: float = 0.8,
) -> np.ndarray:
    """Construct a log-k grid, optionally rejecting box/grid-incompatible support."""
    if (
        not (
            math.isfinite(k_min_mpc_inv)
            and math.isfinite(k_max_mpc_inv)
            and 0 < k_min_mpc_inv < k_max_mpc_inv
        )
        or n_samples < 3
    ):
        raise ValueError("require finite 0<k_min<k_max and n_samples>=3")
    grid = np.geomspace(k_min_mpc_inv, k_max_mpc_inv, int(n_samples))
    if box_size_mpc is not None or grid_size is not None:
        if box_size_mpc is None or grid_size is None:
            raise ValueError("box_size_mpc and grid_size must be supplied together")
        audit = audit_k_domain(
            grid, box_size_mpc=box_size_mpc, grid_size=grid_size, nyquist_fraction=nyquist_fraction
        )
        if not audit.requested_k_inside_fundamental_to_nyquist:
            raise ValueError(
                "requested grid falls outside the periodic-box fundamental-to-Nyquist domain"
            )
    return grid


__all__ = ["KDomainAudit", "audit_k_domain", "suggest_log_k_grid"]
