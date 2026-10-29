"""Explicit target-name and unit contracts for cosmological inference.

No aliases are silently converted to a physical convention: the user must specify both the
source name and source unit when a conversion is required. In particular, Mnu in an arbitrary
NPZ file is not assumed to mean eV without metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

CanonicalTarget = Literal["sum_mnu_ev", "omega_m", "sigma8", "h", "n_s"]


@dataclass(frozen=True)
class TargetColumn:
    canonical_name: str
    source_name: str
    source_unit: str
    canonical_unit: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    description: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.canonical_name not in {"sum_mnu_ev", "omega_m", "sigma8", "h", "n_s"}:
            errors.append(f"unsupported canonical target {self.canonical_name!r}")
        if not self.source_name.strip():
            errors.append("source_name must be non-empty")
        if not self.source_unit.strip():
            errors.append("source_unit must be explicit")
        if not self.canonical_unit.strip():
            errors.append("canonical_unit must be explicit")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            errors.append("lower_bound must be less than upper_bound")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


# Only dimensionless identity mappings and explicitly named mass conversions are supported.
_UNIT_TO_EV = {"eV": 1.0, "meV": 1e-3, "keV": 1e3}
_DIMENSIONLESS = {"1", "dimensionless"}


def convert_target(values: np.ndarray, contract: TargetColumn) -> np.ndarray:
    """Convert a target column to the declared canonical unit, refusing ambiguous inputs."""
    errors = contract.validate()
    if errors:
        raise ValueError("invalid target contract: " + "; ".join(errors))
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise ValueError("target values must be a finite 1-D vector")
    if contract.canonical_name == "sum_mnu_ev":
        if contract.canonical_unit != "eV":
            raise ValueError("sum_mnu_ev canonical unit must be eV")
        if contract.source_unit not in _UNIT_TO_EV:
            raise ValueError(f"unsupported or ambiguous mass unit {contract.source_unit!r}")
        out = arr * _UNIT_TO_EV[contract.source_unit]
    else:
        if (
            contract.source_unit not in _DIMENSIONLESS
            or contract.canonical_unit not in _DIMENSIONLESS
        ):
            raise ValueError(
                f"{contract.canonical_name} requires explicit dimensionless source/canonical units"
            )
        out = arr.copy()
    if contract.lower_bound is not None and np.min(out) < contract.lower_bound:
        raise ValueError("converted target values fall below declared lower bound")
    if contract.upper_bound is not None and np.max(out) > contract.upper_bound:
        raise ValueError("converted target values exceed declared upper bound")
    return out


def canonical_parameter_order(columns: list[TargetColumn]) -> list[str]:
    names = [c.canonical_name for c in columns]
    if len(set(names)) != len(names):
        raise ValueError("duplicate canonical target names")
    for c in columns:
        errors = c.validate()
        if errors:
            raise ValueError("invalid target contract: " + "; ".join(errors))
    return names
