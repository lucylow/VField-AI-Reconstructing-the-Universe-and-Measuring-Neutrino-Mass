"""Protocol and metadata contract for future physical backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from nufield.parameters import CosmologyParameters


@dataclass(frozen=True, slots=True)
class BackendMetadata:
    name: str
    version: str
    physics_level: str
    k_unit: str
    power_unit: str
    neutrino_model: str
    valid_scales: str
    caveats: tuple[str, ...] = ()


@runtime_checkable
class LinearPowerBackend(Protocol):
    """Protocol for linear power-spectrum backends.

    Implementations must document units. The starter package does not claim that a linear spectrum
    is sufficient for nonlinear field-level inference.
    """

    def metadata(self) -> BackendMetadata: ...
    def linear_power(
        self, params: CosmologyParameters, z: float, k_h_mpc: np.ndarray
    ) -> np.ndarray: ...


def validate_power_output(k: np.ndarray, power: np.ndarray) -> None:
    """Shared numeric validation for backend outputs."""
    kk, pp = np.asarray(k, float), np.asarray(power, float)
    if kk.ndim != 1 or pp.shape != kk.shape:
        raise ValueError("k and power must be one-dimensional arrays of equal shape")
    if not np.isfinite(kk).all() or not np.isfinite(pp).all():
        raise ValueError("k and power must be finite")
    if np.any(kk <= 0) or np.any(pp <= 0):
        raise ValueError("k and power values must be strictly positive")
    if np.any(np.diff(kk) <= 0):
        raise ValueError("k must be strictly increasing")
