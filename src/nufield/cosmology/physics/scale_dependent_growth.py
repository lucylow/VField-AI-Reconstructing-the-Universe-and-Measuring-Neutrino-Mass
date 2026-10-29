"""Measure scale-dependent growth directly from linear spectra returned by a theory backend.

For a linear density contrast, P(k,z) ∝ D(k,z)^2. With massive neutrinos, growth can depend on scale,
so the ratio sqrt[P(k,z)/P(k,0)] is not generally a single scalar D(z). This diagnostic estimates D(k,z)
and f(k,z)=d ln D/d ln a from a backend-provided component spectrum. It is not valid for a nonlinear
spectrum and inherits all density-variable/gauge conventions of the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from nufield.cosmology.physics.parameters import PhysicalCosmology


@dataclass(frozen=True, slots=True)
class ScaleDependentGrowth:
    redshifts: np.ndarray
    k_mpc_inv: np.ndarray
    growth_factor: np.ndarray
    growth_rate: np.ndarray
    component: str
    backend: str
    backend_version: str
    parameters: dict[str, object]
    scale_dependence_rms: np.ndarray
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        z, k = np.asarray(self.redshifts, float), np.asarray(self.k_mpc_inv, float)
        d, f = np.asarray(self.growth_factor, float), np.asarray(self.growth_rate, float)
        if z.ndim != 1 or k.ndim != 1 or len(z) < 3 or len(k) < 3:
            raise ValueError("redshifts and k must be 1-D with >=3 points")
        if d.shape != (len(z), len(k)) or f.shape != d.shape:
            raise ValueError("growth arrays must have shape (n_z,n_k)")
        if (
            np.any(~np.isfinite(z))
            or np.any(~np.isfinite(k))
            or np.any(~np.isfinite(d))
            or np.any(~np.isfinite(f))
        ):
            raise ValueError("growth outputs must be finite")
        if np.any(np.diff(z) <= 0) or np.any(k <= 0) or np.any(np.diff(k) <= 0) or np.any(d <= 0):
            raise ValueError("redshifts and k must be increasing, and growth positive")
        object.__setattr__(self, "redshifts", z.copy())
        object.__setattr__(self, "k_mpc_inv", k.copy())
        object.__setattr__(self, "growth_factor", d.copy())
        object.__setattr__(self, "growth_rate", f.copy())

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, object]:
        out = {
            "component": self.component,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "parameters": self.parameters,
            "scale_dependence_rms": self.scale_dependence_rms.tolist(),
            "caveats": list(self.caveats),
            "units": {"k": "Mpc^-1", "D": "dimensionless", "f": "dimensionless"},
        }
        if include_arrays:
            out.update(
                {
                    "redshifts": self.redshifts.tolist(),
                    "k_mpc_inv": self.k_mpc_inv.tolist(),
                    "growth_factor": self.growth_factor.tolist(),
                    "growth_rate": self.growth_rate.tolist(),
                }
            )
        return out


def compute_scale_dependent_growth(
    backend: Any,
    params: PhysicalCosmology,
    redshifts: np.ndarray,
    k_mpc_inv: np.ndarray,
    *,
    component: str = "cb",
    interpolation_points: int | None = None,
) -> ScaleDependentGrowth:
    """Compute D(k,z) and f(k,z) from linear P(k,z) at a redshift grid including z=0."""
    z = np.asarray(redshifts, float)
    k = np.asarray(k_mpc_inv, float)
    if (
        z.ndim != 1
        or len(z) < 3
        or np.any(~np.isfinite(z))
        or np.any(z < 0)
        or np.any(np.diff(z) <= 0)
    ):
        raise ValueError("redshifts must be strictly increasing, finite, non-negative, >=3 points")
    if not np.isclose(z[0], 0.0, atol=1e-12, rtol=0):
        raise ValueError("redshift grid must start at z=0 for D(k,0)=1 normalization")
    if (
        k.ndim != 1
        or len(k) < 3
        or np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError("k must be positive finite increasing with >=3 values")
    method = getattr(backend, "matter_power_component", None)
    if not callable(method):
        if component != "total":
            raise TypeError(
                "backend must expose matter_power_component for non-total density components"
            )
        method = getattr(backend, "matter_power", None)
    if not callable(method):
        raise TypeError("backend must expose a matter power method")
    pk = np.empty((len(z), len(k)), float)
    for i, zi in enumerate(z):
        spectrum = (
            method(params, float(zi), k, component=component)
            if component != "total" or hasattr(backend, "matter_power_component")
            else method(params, float(zi), k)
        )
        if not spectrum.linear:
            raise ValueError("scale-dependent growth must be computed from linear spectra")
        pk[i] = spectrum.evaluate(k)
    growth = np.sqrt(pk / pk[0][None, :])
    # x=ln(a) decreases as z increases; np.gradient supports strictly monotonic coordinate arrays.
    ln_a = -np.log1p(z)
    growth_rate = np.gradient(np.log(growth), ln_a, axis=0, edge_order=2)
    # RMS spread across k in log growth at each z, excluding the normalized z=0 row.
    if growth.shape[1] > 1:
        logd = np.log(growth)
        scale_rms = np.sqrt(np.mean((logd - logd.mean(axis=1, keepdims=True)) ** 2, axis=1))
    else:
        scale_rms = np.zeros(len(z))
    return ScaleDependentGrowth(
        z,
        k,
        growth,
        growth_rate,
        component,
        type(backend).__name__,
        str(getattr(backend, "version", "unknown")),
        params.to_dict(),
        scale_rms,
        (
            "Derived from linear auto-power spectra; nonlinear P(k,z) is rejected.",
            "Growth inherits the solver's component and gauge conventions; cb and total-matter growth differ with massive neutrinos.",
            "f(k,z) is numerically differentiated in ln(a); rerun at denser redshift grids to test stability.",
            "This diagnostic is not an independent Boltzmann solver or neutrino-mass constraint.",
        ),
    )


def compare_growth_scale_dependence(
    a: ScaleDependentGrowth, b: ScaleDependentGrowth
) -> dict[str, object]:
    """Compare two growth calculations on a common overlapping (z,k) support."""
    if a.component != b.component:
        raise ValueError("growth components must match")
    zlo, zhi = max(a.redshifts[0], b.redshifts[0]), min(a.redshifts[-1], b.redshifts[-1])
    klo, khi = max(a.k_mpc_inv[0], b.k_mpc_inv[0]), min(a.k_mpc_inv[-1], b.k_mpc_inv[-1])
    if zhi <= zlo or khi <= klo:
        raise ValueError("growth grids do not overlap")
    z = np.asarray([x for x in a.redshifts if zlo <= x <= zhi])
    k = np.asarray([x for x in a.k_mpc_inv if klo <= x <= khi])
    if len(z) < 2 or len(k) < 2:
        raise ValueError("not enough common grid points")

    def interp_grid(obj: ScaleDependentGrowth, values: np.ndarray) -> np.ndarray:
        rows = []
        for i in range(values.shape[0]):
            rows.append(
                PchipInterpolator(np.log(obj.k_mpc_inv), values[i], extrapolate=False)(np.log(k))
            )
        return PchipInterpolator(obj.redshifts, np.asarray(rows), axis=0, extrapolate=False)(z)

    da, db = interp_grid(a, a.growth_factor), interp_grid(b, b.growth_factor)
    fa, fb = interp_grid(a, a.growth_rate), interp_grid(b, b.growth_rate)
    dlog = np.log(db) - np.log(da)
    return {
        "z_common": z.tolist(),
        "k_common_mpc_inv": k.tolist(),
        "growth_log_ratio_rms": float(np.sqrt(np.mean(dlog**2))),
        "growth_log_ratio_max_abs": float(np.max(np.abs(dlog))),
        "growth_rate_difference_rms": float(np.sqrt(np.mean((fb - fa) ** 2))),
        "growth_rate_difference_max_abs": float(np.max(np.abs(fb - fa))),
        "backend_a": a.backend,
        "backend_b": b.backend,
        "note": "Compare only matched cosmology, component definition, gauge, and linear solver conventions.",
    }


__all__ = [
    "ScaleDependentGrowth",
    "compute_scale_dependent_growth",
    "compare_growth_scale_dependence",
]
