"""Connect oscillation-scale neutrino masses to cosmological mass-sum parameters.

The module is bookkeeping for the relation between the cosmological sum Σmν and selected laboratory
observables. It is not a global neutrino-oscillation fit and it does not include likelihoods from
oscillation, beta-decay, or neutrinoless-double-beta-decay experiments. The default mixing parameters
are representative values; production analyses should version and cite their chosen global fit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from nufield.cosmology.physics.neutrinos import (
    DEFAULT_DM2_ATM_INVERTED_EV2,
    DEFAULT_DM2_ATM_NORMAL_EV2,
    DEFAULT_DM2_SOLAR_EV2,
)

Hierarchy = Literal["normal", "inverted"]


@dataclass(frozen=True, slots=True)
class OscillationReference:
    """Versioned oscillation inputs for hierarchy bookkeeping.

    Values are representative rather than claimed to be the latest global-fit central values. Store
    the reference string and values in result manifests whenever masses are converted between bases.
    """

    dm2_solar_ev2: float = DEFAULT_DM2_SOLAR_EV2
    dm2_atm_normal_ev2: float = DEFAULT_DM2_ATM_NORMAL_EV2
    dm2_atm_inverted_ev2: float = DEFAULT_DM2_ATM_INVERTED_EV2
    sin2_theta12: float = 0.304
    sin2_theta13: float = 0.0222
    reference: str = "representative inputs; replace with pinned oscillation global fit"

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(x) or x <= 0
            for x in (self.dm2_solar_ev2, self.dm2_atm_normal_ev2, self.dm2_atm_inverted_ev2)
        ):
            raise ValueError("mass-squared splittings must be finite and positive")
        if not (0 < self.sin2_theta12 < 1 and 0 < self.sin2_theta13 < 1):
            raise ValueError("mixing-angle squared sines must lie strictly between zero and one")
        if not self.reference.strip():
            raise ValueError("reference must be a non-empty provenance label")

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeutrinoMassObservables:
    """Mass eigenvalues and selected laboratory-sensitive combinations, all masses in eV."""

    hierarchy: str
    sum_mnu_ev: float
    masses_ev: tuple[float, float, float]
    minimum_sum_ev: float
    effective_beta_mass_ev: float
    effective_double_beta_min_ev: float
    effective_double_beta_max_ev: float
    oscillation_reference: dict[str, float | str]

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.sum_mnu_ev,
                self.minimum_sum_ev,
                self.effective_beta_mass_ev,
                self.effective_double_beta_min_ev,
                self.effective_double_beta_max_ev,
                *self.masses_ev,
            ),
            float,
        )
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("mass observables must be finite and non-negative")
        if (
            len(self.masses_ev) != 3
            or self.effective_double_beta_min_ev > self.effective_double_beta_max_ev
        ):
            raise ValueError("invalid mass tuple or double-beta interval")

    def to_dict(self) -> dict[str, object]:
        return {
            "hierarchy": self.hierarchy,
            "sum_mnu_ev": self.sum_mnu_ev,
            "masses_ev": list(self.masses_ev),
            "minimum_sum_ev": self.minimum_sum_ev,
            "effective_beta_mass_ev": self.effective_beta_mass_ev,
            "effective_double_beta_min_ev": self.effective_double_beta_min_ev,
            "effective_double_beta_max_ev": self.effective_double_beta_max_ev,
            "oscillation_reference": self.oscillation_reference,
            "caveat": "Derived combinations only; no experimental likelihood or phase prior is included.",
        }


def _eigenmasses_from_lightest(
    lightest: float, hierarchy: Hierarchy, inputs: OscillationReference
) -> tuple[float, float, float]:
    m0 = float(lightest)
    if hierarchy == "normal":
        return (
            m0,
            math.sqrt(m0 * m0 + inputs.dm2_solar_ev2),
            math.sqrt(m0 * m0 + inputs.dm2_atm_normal_ev2),
        )
    if hierarchy == "inverted":
        m3 = m0
        m1 = math.sqrt(m3 * m3 + inputs.dm2_atm_inverted_ev2)
        m2 = math.sqrt(m1 * m1 + inputs.dm2_solar_ev2)
        return m1, m2, m3
    raise ValueError("hierarchy must be 'normal' or 'inverted'")


def minimum_mass_sum_ev(
    hierarchy: Hierarchy, *, inputs: OscillationReference | None = None
) -> float:
    """Return the minimum mass sum permitted by the selected three-state hierarchy."""
    ref = inputs or OscillationReference()
    return float(sum(_eigenmasses_from_lightest(0.0, hierarchy, ref)))


def solve_mass_eigenvalues(
    sum_mnu_ev: float, hierarchy: Hierarchy, *, inputs: OscillationReference | None = None
) -> tuple[float, float, float]:
    """Solve the three eigenmasses for Σmν without assuming a degenerate hierarchy."""
    total = float(sum_mnu_ev)
    if not math.isfinite(total) or total < 0:
        raise ValueError("sum_mnu_ev must be finite and non-negative")
    ref = inputs or OscillationReference()
    minimum = minimum_mass_sum_ev(hierarchy, inputs=ref)
    if total < minimum - 1e-12:
        raise ValueError(f"Σmν={total:g} eV is below {hierarchy} hierarchy minimum {minimum:g} eV")
    if abs(total - minimum) <= 1e-12:
        return _eigenmasses_from_lightest(0.0, hierarchy, ref)
    hi = max(total, 1.0)
    root = brentq(
        lambda m: sum(_eigenmasses_from_lightest(m, hierarchy, ref)) - total, 0.0, hi, xtol=1e-14
    )
    masses = _eigenmasses_from_lightest(root, hierarchy, ref)
    if not np.isclose(sum(masses), total, rtol=0, atol=1e-10):
        raise ArithmeticError("mass eigenvalues do not reproduce the requested total")
    return masses


def electron_flavor_weights(
    inputs: OscillationReference | None = None,
) -> tuple[float, float, float]:
    """Return |U_ei|² in the standard three-angle parametrization, ignoring irrelevant phases."""
    ref = inputs or OscillationReference()
    s12, s13 = ref.sin2_theta12, ref.sin2_theta13
    c12, c13 = 1.0 - s12, 1.0 - s13
    weights = (c13 * c12, c13 * s12, s13)
    if not np.isclose(sum(weights), 1.0, atol=1e-12):
        raise ArithmeticError("electron-flavor mixing weights failed normalization")
    return tuple(float(x) for x in weights)


def effective_beta_mass_ev(
    masses_ev: tuple[float, float, float] | list[float] | np.ndarray,
    *,
    inputs: OscillationReference | None = None,
) -> float:
    r"""Compute m_beta = sqrt(Σ_i |U_ei|² m_i²), in eV."""
    m = np.asarray(masses_ev, float)
    if m.shape != (3,) or np.any(~np.isfinite(m)) or np.any(m < 0):
        raise ValueError("masses_ev must contain three finite non-negative eigenmasses")
    weights = np.asarray(electron_flavor_weights(inputs))
    return float(np.sqrt(np.dot(weights, m * m)))


def effective_double_beta_interval_ev(
    masses_ev: tuple[float, float, float] | list[float] | np.ndarray,
    *,
    inputs: OscillationReference | None = None,
) -> tuple[float, float]:
    r"""Return phase-agnostic extrema for m_ββ over two unknown Majorana phases.

    With amplitudes a_i=|U_ei|²m_i, the maximum is Σa_i and the minimum is
    max(2 max(a_i) - Σa_i, 0). This is a geometric envelope, not a probability interval: no phase
    prior, nuclear matrix element, or experimental likelihood is included.
    """
    m = np.asarray(masses_ev, float)
    if m.shape != (3,) or np.any(~np.isfinite(m)) or np.any(m < 0):
        raise ValueError("masses_ev must contain three finite non-negative eigenmasses")
    amplitudes = np.asarray(electron_flavor_weights(inputs)) * m
    total = float(amplitudes.sum())
    lower = float(max(2 * float(amplitudes.max()) - total, 0.0))
    return lower, total


def mass_observables(
    sum_mnu_ev: float, hierarchy: Hierarchy, *, inputs: OscillationReference | None = None
) -> NeutrinoMassObservables:
    """Compute a complete, serializable set of hierarchy-dependent mass combinations."""
    ref = inputs or OscillationReference()
    masses = solve_mass_eigenvalues(sum_mnu_ev, hierarchy, inputs=ref)
    m_beta = effective_beta_mass_ev(masses, inputs=ref)
    m_bb_lo, m_bb_hi = effective_double_beta_interval_ev(masses, inputs=ref)
    return NeutrinoMassObservables(
        hierarchy=hierarchy,
        sum_mnu_ev=float(sum_mnu_ev),
        masses_ev=masses,
        minimum_sum_ev=minimum_mass_sum_ev(hierarchy, inputs=ref),
        effective_beta_mass_ev=m_beta,
        effective_double_beta_min_ev=m_bb_lo,
        effective_double_beta_max_ev=m_bb_hi,
        oscillation_reference=ref.to_dict(),
    )


def compare_hierarchy_observables(
    sum_mnu_values_ev: list[float] | np.ndarray, *, inputs: OscillationReference | None = None
) -> list[dict[str, object]]:
    """Create paired normal/inverted hierarchy mass-observable rows for plotting or reports."""
    values = np.asarray(sum_mnu_values_ev, float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("sum_mnu_values_ev must be a non-empty finite non-negative vector")
    rows: list[dict[str, object]] = []
    for total in values:
        for hierarchy in ("normal", "inverted"):
            try:
                row = mass_observables(float(total), hierarchy, inputs=inputs).to_dict()
            except ValueError as exc:
                row = {
                    "hierarchy": hierarchy,
                    "sum_mnu_ev": float(total),
                    "valid": False,
                    "reason": str(exc),
                }
            else:
                row["valid"] = True
            rows.append(row)
    return rows


__all__ = [
    "OscillationReference",
    "NeutrinoMassObservables",
    "minimum_mass_sum_ev",
    "solve_mass_eigenvalues",
    "electron_flavor_weights",
    "effective_beta_mass_ev",
    "effective_double_beta_interval_ev",
    "mass_observables",
    "compare_hierarchy_observables",
]
