"""Parameter definitions and strict validation.

These ranges are only starter defaults. They are not recommendations for scientific priors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from nufield.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class CosmologyParameters:
    """Small cosmology-like parameter vector used by the prototype.

    `sum_mnu_ev` is a toy parameter until paired with a validated massive-neutrino simulator.
    """

    sum_mnu_ev: float = 0.12
    omega_m: float = 0.315
    sigma8: float = 0.811
    h: float = 0.674
    n_s: float = 0.965
    bias: float = 1.2

    def __post_init__(self) -> None:
        bounds = {
            "sum_mnu_ev": (0.0, 1.5),
            "omega_m": (0.1, 0.6),
            "sigma8": (0.4, 1.4),
            "h": (0.4, 1.0),
            "n_s": (0.5, 1.5),
            "bias": (0.1, 4.0),
        }
        for name, (lo, hi) in bounds.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ConfigurationError(f"{name} must be finite and in [{lo}, {hi}], got {value}")

    def as_array(self) -> list[float]:
        """Return parameters in the package's canonical order."""
        return [self.sum_mnu_ev, self.omega_m, self.sigma8, self.h, self.n_s, self.bias]

    def to_dict(self) -> dict[str, float]:
        """Serialize parameters with canonical names."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> CosmologyParameters:
        known = set(cls.__dataclass_fields__)
        unknown = set(mapping) - known
        if unknown:
            raise ConfigurationError(f"Unknown cosmology parameter(s): {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in mapping.items()})


PARAMETER_NAMES = ["sum_mnu_ev", "omega_m", "sigma8", "h", "n_s", "bias"]
PARAMETER_UNITS = {
    "sum_mnu_ev": "eV (toy parameter; physical interpretation disabled in toy mode)",
    "omega_m": "dimensionless",
    "sigma8": "dimensionless",
    "h": "dimensionless H0/100 km s^-1 Mpc^-1",
    "n_s": "dimensionless",
    "bias": "dimensionless tracer bias coefficient",
}
