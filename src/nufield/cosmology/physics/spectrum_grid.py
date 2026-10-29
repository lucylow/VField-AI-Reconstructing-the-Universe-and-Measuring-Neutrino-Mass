"""Interpolation on a rectilinear cosmological parameter grid.

This class is intended for tabulated Boltzmann-code grids. It refuses out-of-bounds requests and
interpolates log power, but it does not add an emulator error model or guarantee that coarse grids
resolve sharp parameter responses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class GridSupport:
    parameter_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    grid_sizes: tuple[int, ...]


class RectilinearSpectrumGrid:
    """Regular-grid interpolator for log P(k) over parameter dimensions."""

    def __init__(
        self,
        axes: tuple[np.ndarray, ...],
        k_mpc_inv: np.ndarray,
        log_pk: np.ndarray,
        parameter_names: tuple[str, ...] | list[str],
        *,
        redshift: float,
        source: str,
    ):
        self.axes = tuple(np.asarray(axis, float) for axis in axes)
        self.k = np.asarray(k_mpc_inv, float)
        self.log_pk = np.asarray(log_pk, float)
        self.names = tuple(parameter_names)
        self.redshift, self.source = float(redshift), source
        if len(self.axes) != len(self.names) or len(self.axes) < 1:
            raise ValueError("one parameter axis is required per parameter name")
        for axis in self.axes:
            if (
                axis.ndim != 1
                or len(axis) < 2
                or np.any(np.diff(axis) <= 0)
                or np.any(~np.isfinite(axis))
            ):
                raise ValueError("each parameter axis must be finite and strictly increasing")
        if (
            self.k.ndim != 1
            or len(self.k) < 3
            or np.any(self.k <= 0)
            or np.any(np.diff(self.k) <= 0)
        ):
            raise ValueError("k must be positive and strictly increasing")
        expected = tuple(len(axis) for axis in self.axes) + (len(self.k),)
        if self.log_pk.shape != expected or np.any(~np.isfinite(self.log_pk)):
            raise ValueError(f"log_pk shape must equal {expected} and be finite")
        if not np.isfinite(self.redshift) or self.redshift < 0:
            raise ValueError("redshift must be finite and non-negative")
        self._interp = RegularGridInterpolator(
            self.axes, self.log_pk, method="linear", bounds_error=True
        )
        self.support = GridSupport(
            self.names,
            tuple(float(a[0]) for a in self.axes),
            tuple(float(a[-1]) for a in self.axes),
            tuple(len(a) for a in self.axes),
        )

    def predict_log_pk(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, float)
        if values.shape != (len(self.axes),) or np.any(~np.isfinite(values)):
            raise ValueError("theta must be a finite vector matching parameter dimension")
        # RegularGridInterpolator returns one vector for each query point, where trailing axis is k.
        return np.asarray(self._interp(values[None, :])[0], float)

    def predict(self, theta: np.ndarray) -> MatterPowerSpectrum:
        return MatterPowerSpectrum(
            self.k,
            np.exp(self.predict_log_pk(theta)),
            self.redshift,
            True,
            self.source,
            (("interpolation", "rectilinear-log-linear"), ("uncertainty_model", "none")),
        )

    def contains(self, theta: np.ndarray) -> bool:
        v = np.asarray(theta, float)
        if v.shape != (len(self.axes),) or np.any(~np.isfinite(v)):
            return False
        return all(axis[0] <= val <= axis[-1] for axis, val in zip(self.axes, v))

    def fractional_step_resolution(self) -> dict[str, np.ndarray]:
        """Return interval arrays for each parameter axis to assist grid-design inspection."""
        return {name: np.diff(axis) for name, axis in zip(self.names, self.axes)}


__all__ = ["GridSupport", "RectilinearSpectrumGrid"]
