"""Finite-difference response experiments for cosmological observables.

This module treats a theory prediction as a black box and records how responses change with finite-
difference step. It is intended for building local Jacobians of P(k), BAO observables or joint data vectors
from CAMB/CLASS or another backend. It does not assume that any particular step is converged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

ArrayModel = Callable[[Mapping[str, float]], np.ndarray]


@dataclass(frozen=True, slots=True)
class ResponseColumnDiagnostics:
    parameter: str
    fiducial_value: float
    step: float
    method: str
    derivative_norm: float
    coarse_fine_relative_difference: float | None
    max_abs_relative_difference: float | None
    near_bound: bool


@dataclass(frozen=True, slots=True)
class ParameterResponseCampaign:
    parameter_names: tuple[str, ...]
    observable_shape: tuple[int, ...]
    fiducial_observable: np.ndarray
    jacobian: np.ndarray
    columns: tuple[ResponseColumnDiagnostics, ...]
    parameter_bounds: dict[str, tuple[float | None, float | None]]
    caveats: tuple[str, ...]

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, object]:
        out = {
            "parameter_names": list(self.parameter_names),
            "observable_shape": list(self.observable_shape),
            "columns": [
                d.__dict__
                if hasattr(d, "__dict__")
                else {
                    "parameter": d.parameter,
                    "fiducial_value": d.fiducial_value,
                    "step": d.step,
                    "method": d.method,
                    "derivative_norm": d.derivative_norm,
                    "coarse_fine_relative_difference": d.coarse_fine_relative_difference,
                    "max_abs_relative_difference": d.max_abs_relative_difference,
                    "near_bound": d.near_bound,
                }
                for d in self.columns
            ],
            "parameter_bounds": {k: list(v) for k, v in self.parameter_bounds.items()},
            "caveats": list(self.caveats),
        }
        if include_arrays:
            out["fiducial_observable"] = self.fiducial_observable.tolist()
            out["jacobian"] = self.jacobian.tolist()
        return out


def _model_eval(
    model: ArrayModel, p: dict[str, float], expected_shape: tuple[int, ...] | None = None
) -> np.ndarray:
    value = np.asarray(model(p), float)
    if value.ndim < 1 or np.any(~np.isfinite(value)):
        raise ValueError("model must return a non-scalar finite array")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(f"model output shape changed from {expected_shape} to {value.shape}")
    return value


def finite_difference_responses(
    model: ArrayModel,
    fiducial_parameters: Mapping[str, float],
    parameter_names: Sequence[str],
    *,
    relative_steps: Mapping[str, float] | None = None,
    absolute_steps: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    central_if_possible: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[ResponseColumnDiagnostics, ...]]:
    """Return fiducial observable, flattened Jacobian and diagnostics for each parameter.

    Central differences are used when they fit inside bounds; otherwise a one-sided difference is
    taken toward the supported side. Relative steps scale with ``max(abs(theta), 1)`` unless an
    absolute step is supplied. Keep parameter units explicit: e.g. Σmν steps are in eV and H0 steps
    are in km s⁻¹ Mpc⁻¹.
    """
    p0 = {str(k): float(v) for k, v in fiducial_parameters.items()}
    names = tuple(str(x) for x in parameter_names)
    if not names or len(set(names)) != len(names) or any(x not in p0 for x in names):
        raise ValueError("parameter_names must be unique and present in fiducial_parameters")
    if any(not np.isfinite(v) for v in p0.values()):
        raise ValueError("fiducial parameters must be finite")
    rel = dict(relative_steps or {})
    abs_steps = dict(absolute_steps or {})
    bds = dict(bounds or {})
    y0 = _model_eval(model, p0)
    flat0 = y0.ravel()
    jac = np.empty((flat0.size, len(names)), float)
    diagnostics: list[ResponseColumnDiagnostics] = []
    for j, name in enumerate(names):
        x = p0[name]
        if name in abs_steps:
            h = float(abs_steps[name])
        else:
            h = float(rel.get(name, 1e-3)) * max(abs(x), 1.0)
        if not np.isfinite(h) or h <= 0:
            raise ValueError(f"step for {name!r} must be finite and positive")
        lo, hi = bds.get(name, (None, None))
        if lo is not None and x < lo or hi is not None and x > hi:
            raise ValueError(f"fiducial {name!r} lies outside its declared bounds")
        can_minus = lo is None or x - h >= lo
        can_plus = hi is None or x + h <= hi
        if central_if_possible and can_minus and can_plus:
            pm, pp = dict(p0), dict(p0)
            pm[name], pp[name] = x - h, x + h
            ym, yp = _model_eval(model, pm, y0.shape), _model_eval(model, pp, y0.shape)
            derivative = (yp - ym) / (2 * h)
            method = "central"
        elif can_plus:
            pp = dict(p0)
            pp[name] = x + h
            yp = _model_eval(model, pp, y0.shape)
            derivative = (yp - y0) / h
            method = "forward"
        elif can_minus:
            pm = dict(p0)
            pm[name] = x - h
            ym = _model_eval(model, pm, y0.shape)
            derivative = (y0 - ym) / h
            method = "backward"
        else:
            raise ValueError(f"no valid finite-difference direction for {name!r} at its bounds")
        jac[:, j] = derivative.ravel()
        near = not (can_minus and can_plus)
        diagnostics.append(
            ResponseColumnDiagnostics(
                name, x, h, method, float(np.linalg.norm(derivative)), None, None, near
            )
        )
    return y0, jac, tuple(diagnostics)


def response_step_stability(
    model: ArrayModel,
    fiducial_parameters: Mapping[str, float],
    parameter_names: Sequence[str],
    *,
    fine_steps: Mapping[str, float],
    coarse_steps: Mapping[str, float],
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    relative: bool = False,
    relative_floor: float = 1e-12,
) -> list[dict[str, float | str | bool]]:
    """Compare response columns at two step sizes; report relative differences observable by observable."""
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    steps_f: dict[str, float] = {}
    steps_c: dict[str, float] = {}
    for name in parameter_names:
        if name not in fine_steps or name not in coarse_steps:
            raise ValueError(f"fine_steps and coarse_steps must specify {name!r}")
        fine, coarse = float(fine_steps[name]), float(coarse_steps[name])
        if fine <= 0 or coarse <= fine:
            raise ValueError(f"require 0 < fine step < coarse step for {name!r}")
        steps_f[name], steps_c[name] = fine, coarse
    _, jf, _ = finite_difference_responses(
        model, fiducial_parameters, parameter_names, absolute_steps=steps_f, bounds=bounds
    )
    _, jc, _ = finite_difference_responses(
        model, fiducial_parameters, parameter_names, absolute_steps=steps_c, bounds=bounds
    )
    rows = []
    for i, name in enumerate(parameter_names):
        a, b = jf[:, i], jc[:, i]
        denom = np.maximum(np.abs(a), np.median(np.abs(a)) * relative_floor + relative_floor)
        diff = (a - b) / denom if relative else a - b
        rows.append(
            {
                "parameter": str(name),
                "fine_step": steps_f[name],
                "coarse_step": steps_c[name],
                "rms_difference": float(np.sqrt(np.mean(diff**2))),
                "max_abs_difference": float(np.max(np.abs(diff))),
                "relative_metric": bool(relative),
            }
        )
    return rows


__all__ = [
    "ResponseColumnDiagnostics",
    "ParameterResponseCampaign",
    "finite_difference_responses",
    "response_step_stability",
]
