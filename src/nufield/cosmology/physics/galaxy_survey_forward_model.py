"""Transparent toy-to-research forward operators for galaxy field experiments.

This module provides local-bias, redshift-space coordinate, selection, and Poisson sampling operators
for controlled periodic-box studies. These are intentionally modular operators, not an end-to-end DESI
mock generator: realistic survey inference also requires a trusted nonlinear matter simulation, tracer
HOD/selection model, angular completeness, fiber assignment, radial selection, weights, window functions,
and a matching likelihood/covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ForwardModelManifest:
    box_size_mpc: float
    grid_size: int
    scale_factor: float
    H_km_s_mpc: float
    bias_model: str
    bias_parameters: dict[str, float]
    galaxy_number_density_mpc3: float | None
    random_seed: int | None
    survey_selection_applied: bool
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "box_size_mpc": self.box_size_mpc,
            "grid_size": self.grid_size,
            "scale_factor": self.scale_factor,
            "H_km_s_mpc": self.H_km_s_mpc,
            "bias_model": self.bias_model,
            "bias_parameters": self.bias_parameters,
            "galaxy_number_density_mpc3": self.galaxy_number_density_mpc3,
            "random_seed": self.random_seed,
            "survey_selection_applied": self.survey_selection_applied,
            "caveats": list(self.caveats),
        }


def _validate_cubic_field(field: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(field, float)
    if arr.ndim != 3 or len(set(arr.shape)) != 1 or arr.shape[0] < 4 or np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite cubic 3-D field with >=4 cells per axis")
    return arr


def local_galaxy_overdensity(
    matter_overdensity: np.ndarray,
    *,
    linear_bias: float = 1.5,
    quadratic_bias: float = 0.0,
    stochastic_sigma: float = 0.0,
    seed: int | None = None,
    positivity_preserving: str = "lognormal",
) -> np.ndarray:
    r"""Map a matter overdensity to a toy biased tracer overdensity.

    ``linear`` returns b1 δ + b2/2(δ²-<δ²>) and may be below -1. ``lognormal`` returns
    1+δ_g = exp[b1 δ - Var(b1δ)/2], normalized to unit mean. The second mode guarantees positive
    intensity but is only a phenomenological bias prescription; neither mode models a halo occupation
    distribution or nonlinear galaxy bias calibrated to survey tracers.
    """
    delta = _validate_cubic_field(matter_overdensity, "matter_overdensity")
    if not math.isfinite(linear_bias) or linear_bias <= 0 or not math.isfinite(quadratic_bias):
        raise ValueError("linear_bias must be positive finite and quadratic_bias finite")
    if stochastic_sigma < 0 or not math.isfinite(stochastic_sigma):
        raise ValueError("stochastic_sigma must be finite and non-negative")
    centered = delta - float(np.mean(delta))
    if positivity_preserving not in {"linear", "lognormal"}:
        raise ValueError("positivity_preserving must be 'linear' or 'lognormal'")
    rng = np.random.default_rng(seed)
    if positivity_preserving == "linear":
        tracer = linear_bias * centered + 0.5 * quadratic_bias * (
            centered**2 - float(np.mean(centered**2))
        )
        if stochastic_sigma:
            tracer = tracer + rng.normal(0, stochastic_sigma, size=tracer.shape)
        return tracer - float(np.mean(tracer))
    latent = linear_bias * centered + 0.5 * quadratic_bias * (
        centered**2 - float(np.mean(centered**2))
    )
    if stochastic_sigma:
        latent = latent + rng.normal(0, stochastic_sigma, size=latent.shape)
    # Subtract log mean rather than just half the variance so finite-grid intensity has mean exactly one.
    intensity = np.exp(np.clip(latent, -50, 50))
    intensity /= np.mean(intensity)
    return intensity - 1.0


def redshift_space_shift_positions(
    positions_mpc: np.ndarray,
    peculiar_velocity_km_s: np.ndarray,
    *,
    scale_factor: float,
    H_km_s_mpc: float,
    box_size_mpc: float,
    los_axis: int = 2,
    periodic: bool = True,
) -> np.ndarray:
    r"""Apply plane-parallel RSD displacement ``s = x + v_los/(a H)`` to positions.

    Positions are comoving Mpc and velocities are peculiar km/s. The plane-parallel approximation
    assumes one fixed line of sight. Positions are wrapped into a periodic cube by default.
    """
    x, v = np.asarray(positions_mpc, float), np.asarray(peculiar_velocity_km_s, float)
    if x.ndim != 2 or x.shape[1] != 3 or v.shape != x.shape or np.any(~np.isfinite(x + v)):
        raise ValueError("positions and velocities must be matching finite arrays of shape (n,3)")
    if not 0 < scale_factor <= 1 or not np.isfinite(scale_factor):
        raise ValueError("scale_factor must be in (0,1]")
    if (
        not np.isfinite(H_km_s_mpc)
        or H_km_s_mpc <= 0
        or not np.isfinite(box_size_mpc)
        or box_size_mpc <= 0
    ):
        raise ValueError("H and box size must be positive and finite")
    if los_axis not in (0, 1, 2):
        raise ValueError("los_axis must be 0, 1 or 2")
    out = x.copy()
    out[:, los_axis] += v[:, los_axis] / (scale_factor * H_km_s_mpc)
    if periodic:
        out %= box_size_mpc
    elif np.any((out < 0) | (out >= box_size_mpc)):
        raise ValueError("non-periodic redshift-space positions left the declared box")
    return out


def expected_counts_from_overdensity(
    galaxy_overdensity: np.ndarray,
    *,
    number_density_mpc3: float,
    box_size_mpc: float,
    selection_fraction: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Convert a galaxy overdensity to expected voxel counts with an optional completeness fraction."""
    delta = _validate_cubic_field(galaxy_overdensity, "galaxy_overdensity")
    if (
        not np.isfinite(number_density_mpc3)
        or number_density_mpc3 <= 0
        or not np.isfinite(box_size_mpc)
        or box_size_mpc <= 0
    ):
        raise ValueError("number density and box size must be positive and finite")
    frac = np.asarray(selection_fraction, float)
    if frac.ndim == 0:
        frac = np.full_like(delta, float(frac))
    if (
        frac.shape != delta.shape
        or np.any(~np.isfinite(frac))
        or np.any(frac < 0)
        or np.any(frac > 1)
    ):
        raise ValueError("selection_fraction must be scalar or match the field and lie in [0,1]")
    intensity = 1.0 + delta
    if np.any(intensity < -1e-8):
        raise ValueError(
            "galaxy overdensity yields negative expected intensity; use a positivity-preserving bias model"
        )
    intensity = np.maximum(intensity, 0.0)
    cell_volume = (box_size_mpc / delta.shape[0]) ** 3
    expected = number_density_mpc3 * cell_volume * intensity * frac
    if np.any(~np.isfinite(expected)) or np.any(expected < 0):
        raise ArithmeticError("expected-count construction failed positivity")
    return expected


def poisson_sample_counts(expected_counts: np.ndarray, *, seed: int) -> np.ndarray:
    """Draw independent Poisson counts for each voxel with an explicit seed."""
    lam = np.asarray(expected_counts, float)
    if lam.ndim != 3 or np.any(~np.isfinite(lam)) or np.any(lam < 0):
        raise ValueError("expected_counts must be a finite non-negative 3-D array")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return np.random.default_rng(seed).poisson(lam).astype(np.int64)


def counts_to_overdensity(
    counts: np.ndarray,
    expected_mean_counts: np.ndarray | float,
    *,
    valid_mask: np.ndarray | None = None,
    masked_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize observed counts by an expected-selection map and return ``(delta, valid_mask)``.

    A cell with expected count zero is excluded from the returned validity mask. The default value for
    invalid cells is zero only as a storage convention; it must not be mistaken for observed zero density.
    Keep and propagate the mask through feature extraction and likelihood construction.
    """
    c = np.asarray(counts, float)
    exp = np.asarray(expected_mean_counts, float)
    if c.ndim != 3 or np.any(~np.isfinite(c)) or np.any(c < 0):
        raise ValueError("counts must be a finite non-negative 3-D array")
    if exp.ndim == 0:
        exp = np.full_like(c, float(exp))
    if exp.shape != c.shape or np.any(~np.isfinite(exp)) or np.any(exp < 0):
        raise ValueError(
            "expected_mean_counts must be scalar or match counts, finite and non-negative"
        )
    mask = exp > 0
    if valid_mask is not None:
        vm = np.asarray(valid_mask, bool)
        if vm.shape != c.shape:
            raise ValueError("valid_mask must match counts")
        mask &= vm
    result = np.full_like(c, float(masked_value))
    result[mask] = c[mask] / exp[mask] - 1.0
    return result, mask


def counts_to_positions(counts: np.ndarray, box_size_mpc: float, *, seed: int) -> np.ndarray:
    """Sample uniform positions within occupied voxels from an integer count grid.

    This is convenient for creating a mock catalog from counts; the resulting point process inherits
    grid-scale discretization and does not recreate the original sub-cell particle distribution.
    """
    grid = np.asarray(counts)
    if (
        grid.ndim != 3
        or len(set(grid.shape)) != 1
        or np.any(grid < 0)
        or np.any(~np.isfinite(grid))
    ):
        raise ValueError("counts must be a finite non-negative cubic grid")
    if not np.allclose(grid, np.round(grid)):
        raise ValueError("counts must be integer-valued")
    if not np.isfinite(box_size_mpc) or box_size_mpc <= 0 or seed < 0:
        raise ValueError("box size positive and seed non-negative required")
    n = grid.shape[0]
    flat = grid.astype(np.int64).ravel()
    cell_index = np.repeat(np.arange(flat.size), flat)
    if not len(cell_index):
        return np.empty((0, 3), dtype=float)
    ix, iy, iz = np.unravel_index(cell_index, grid.shape)
    rng = np.random.default_rng(seed)
    jitter = rng.random((len(cell_index), 3))
    return (np.column_stack((ix, iy, iz)) + jitter) * (box_size_mpc / n)


def build_forward_model_manifest(
    *,
    box_size_mpc: float,
    grid_size: int,
    scale_factor: float,
    H_km_s_mpc: float,
    bias_model: str,
    bias_parameters: dict[str, float],
    galaxy_number_density_mpc3: float | None,
    random_seed: int | None,
    survey_selection_applied: bool,
) -> ForwardModelManifest:
    """Create a provenance object for a controlled galaxy-field forward-model run."""
    if box_size_mpc <= 0 or grid_size < 4 or not 0 < scale_factor <= 1 or H_km_s_mpc <= 0:
        raise ValueError("invalid box/grid/cosmological-time inputs")
    if galaxy_number_density_mpc3 is not None and galaxy_number_density_mpc3 <= 0:
        raise ValueError("galaxy_number_density_mpc3 must be positive if specified")
    if random_seed is not None and random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    if not bias_model:
        raise ValueError("bias_model must be explicitly named")
    return ForwardModelManifest(
        float(box_size_mpc),
        int(grid_size),
        float(scale_factor),
        float(H_km_s_mpc),
        bias_model,
        {str(k): float(v) for k, v in bias_parameters.items()},
        galaxy_number_density_mpc3,
        random_seed,
        bool(survey_selection_applied),
        (
            "Toy local bias and Poisson sampling are not a calibrated galaxy-formation model.",
            "RSD uses a plane-parallel mapping and assumes comoving coordinates.",
            "Survey mask/selection must be propagated into both data and theory.",
        ),
    )


__all__ = [
    "ForwardModelManifest",
    "local_galaxy_overdensity",
    "redshift_space_shift_positions",
    "expected_counts_from_overdensity",
    "poisson_sample_counts",
    "counts_to_overdensity",
    "counts_to_positions",
    "build_forward_model_manifest",
]
