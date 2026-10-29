"""Explicit unit and convention audits for cosmological spectra and distances.

Many severe numerical mistakes in cosmology are convention mismatches rather than bugs in a formula:
P(k) in (Mpc/h)^3 paired with k in 1/Mpc, H0 confused with h, or a total-matter spectrum treated as
CDM+baryon power in a massive-neutrino cosmology. This module records and validates conversion intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

LengthUnit = Literal["Mpc", "Mpc/h"]
PowerUnit = Literal["Mpc^3", "(Mpc/h)^3"]


@dataclass(frozen=True, slots=True)
class SpectrumConvention:
    k_unit: str = "Mpc^-1"
    power_unit: str = "Mpc^3"
    component: str = "total"
    gauge: str = "unspecified"
    redshift: float = 0.0
    solver: str = "unspecified"

    def __post_init__(self) -> None:
        if self.k_unit not in {"Mpc^-1", "h/Mpc"}:
            raise ValueError("k_unit must be 'Mpc^-1' or 'h/Mpc'")
        if self.power_unit not in {"Mpc^3", "(Mpc/h)^3"}:
            raise ValueError("power_unit must be 'Mpc^3' or '(Mpc/h)^3'")
        if self.component not in {"total", "cb", "cdm", "baryon", "nu", "unknown"}:
            raise ValueError("unsupported density component label")
        if not np.isfinite(self.redshift) or self.redshift < 0:
            raise ValueError("redshift must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "k_unit": self.k_unit,
            "power_unit": self.power_unit,
            "component": self.component,
            "gauge": self.gauge,
            "redshift": self.redshift,
            "solver": self.solver,
        }


def convert_spectrum_units(
    k: np.ndarray,
    power: np.ndarray,
    *,
    h: float,
    input_k_unit: str,
    input_power_unit: str,
    output_k_unit: str = "Mpc^-1",
    output_power_unit: str = "Mpc^3",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert paired k/P values between physical Mpc and h-scaled Mpc conventions.

    ``k[h/Mpc] = k[1/Mpc]/h`` and ``P[(Mpc/h)^3] = P[Mpc^3] * h^3``. The transformations
    are paired; never convert k without also converting P according to the corresponding convention.
    """
    kk, pp = np.asarray(k, float), np.asarray(power, float)
    if kk.ndim != 1 or pp.shape != kk.shape or kk.size == 0:
        raise ValueError("k and power must be matching non-empty vectors")
    if np.any(~np.isfinite(kk + pp)) or np.any(kk <= 0) or np.any(pp < 0):
        raise ValueError("k must be positive and finite; power finite and non-negative")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    valid_k = {"Mpc^-1", "h/Mpc"}
    valid_p = {"Mpc^3", "(Mpc/h)^3"}
    if input_k_unit not in valid_k or output_k_unit not in valid_k:
        raise ValueError("unsupported k unit")
    if input_power_unit not in valid_p or output_power_unit not in valid_p:
        raise ValueError("unsupported power unit")
    kout, pout = kk.copy(), pp.copy()
    if input_k_unit != output_k_unit:
        kout = kout / h if input_k_unit == "Mpc^-1" else kout * h
    if input_power_unit != output_power_unit:
        pout = pout * h**3 if input_power_unit == "Mpc^3" else pout / h**3
    return kout, pout


def audit_spectrum_pair(
    k_a: np.ndarray,
    p_a: np.ndarray,
    convention_a: SpectrumConvention,
    k_b: np.ndarray,
    p_b: np.ndarray,
    convention_b: SpectrumConvention,
    *,
    h: float,
    allow_component_mismatch: bool = False,
    allow_redshift_mismatch: bool = False,
    atol: float = 1e-10,
) -> dict[str, object]:
    """Convert two spectra to physical units and audit whether their ratio is meaningful."""
    ka, pa = convert_spectrum_units(
        k_a, p_a, h=h, input_k_unit=convention_a.k_unit, input_power_unit=convention_a.power_unit
    )
    kb, pb = convert_spectrum_units(
        k_b, p_b, h=h, input_k_unit=convention_b.k_unit, input_power_unit=convention_b.power_unit
    )
    errors: list[str] = []
    if convention_a.component != convention_b.component and not allow_component_mismatch:
        errors.append(
            f"density components differ: {convention_a.component} vs {convention_b.component}"
        )
    if (
        not np.isclose(convention_a.redshift, convention_b.redshift, atol=atol, rtol=0)
        and not allow_redshift_mismatch
    ):
        errors.append(f"redshifts differ: {convention_a.redshift} vs {convention_b.redshift}")
    if not np.allclose(ka, kb, rtol=1e-8, atol=atol):
        errors.append(
            "k grids differ after unit conversion; interpolate explicitly before taking a ratio"
        )
    ratio = None
    if not errors:
        if np.any(pb <= 0):
            errors.append("denominator spectrum contains non-positive power")
        else:
            ratio = pa / pb
    return {
        "valid_comparison": not errors,
        "errors": errors,
        "k_mpc_inv": ka.tolist() if ka.shape == kb.shape else None,
        "ratio_a_over_b": None if ratio is None else ratio.tolist(),
        "convention_a": convention_a.to_dict(),
        "convention_b": convention_b.to_dict(),
        "h": float(h),
        "warning": "Unit compatibility is necessary but not sufficient for a physically matched spectrum ratio.",
    }


__all__ = ["SpectrumConvention", "convert_spectrum_units", "audit_spectrum_pair"]
