"""Physics-aware augmentation for periodic simulation boxes.

Rotations/reflections/cyclic shifts are suitable only where the underlying simulation volume and
observable are invariant under cubic symmetries and periodic translations. Do not apply these
augmentations blindly to DESI footprint maps, masked sky maps, lensing maps with a fixed geometry,
or coordinate-aware catalogs.
"""

from __future__ import annotations

from itertools import permutations, product
from typing import Any

import numpy as np


def _cube_check(field: Any) -> np.ndarray:
    arr = np.asarray(field)
    if arr.ndim < 3 or arr.shape[-3] != arr.shape[-2] or arr.shape[-2] != arr.shape[-1]:
        raise ValueError("last three dimensions must be a cubic grid")
    if not np.isfinite(arr).all():
        raise ValueError("field must be finite")
    return arr


def cube_symmetry_transforms() -> list[tuple[tuple[int, int, int], tuple[bool, bool, bool]]]:
    """Enumerate 48 signed coordinate permutations of a cube (24 rotations + reflections)."""
    return [
        (tuple(p), tuple(signs))
        for p in permutations(range(3))
        for signs in product((False, True), repeat=3)
    ]


def apply_cube_symmetry(field: Any, transform_index: int) -> np.ndarray:
    """Apply one of 48 cube symmetries to the last three axes, preserving batch/channel axes."""
    arr = _cube_check(field)
    transforms = cube_symmetry_transforms()
    if not 0 <= transform_index < len(transforms):
        raise ValueError("transform_index must be between 0 and 47")
    perm, flip = transforms[transform_index]
    prefix = list(range(arr.ndim - 3))
    axes = prefix + [arr.ndim - 3 + i for i in perm]
    result = np.transpose(arr, axes)
    for i, do_flip in enumerate(flip):
        if do_flip:
            result = np.flip(result, axis=arr.ndim - 3 + i)
    return np.ascontiguousarray(result)


def random_cube_symmetry(field: Any, *, rng: np.random.Generator) -> np.ndarray:
    """Apply a reproducible random cube symmetry using a caller-owned NumPy generator."""
    return apply_cube_symmetry(field, int(rng.integers(0, 48)))


def periodic_roll(
    field: Any, *, rng: np.random.Generator, shifts: tuple[int, int, int] | None = None
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Apply a cyclic shift on the final three periodic spatial axes."""
    arr = _cube_check(field)
    if shifts is None:
        shifts = tuple(int(rng.integers(0, n)) for n in arr.shape[-3:])
    if len(shifts) != 3:
        raise ValueError("shifts must have three integers")
    resolved = tuple(int(s) % arr.shape[-3 + i] for i, s in enumerate(shifts))
    axes = tuple(range(arr.ndim - 3, arr.ndim))
    return np.roll(arr, shift=resolved, axis=axes), resolved


def additive_noise(
    field: Any, *, rng: np.random.Generator, standard_deviation: float, preserve_mean: bool = True
) -> np.ndarray:
    """Add explicit, dimensionless Gaussian noise for robustness experiments.

    This operation is a stress test unless the noise model is calibrated to the observation process.
    It is not a generic replacement for Poisson shot noise, survey systematics, or instrument noise.
    """
    arr = np.asarray(field, float)
    if not np.isfinite(arr).all() or standard_deviation < 0 or not np.isfinite(standard_deviation):
        raise ValueError("field must be finite and standard_deviation nonnegative")
    noise = rng.normal(0.0, standard_deviation, size=arr.shape)
    output = arr + noise
    if preserve_mean:
        output += arr.mean(
            axis=tuple(range(max(0, arr.ndim - 3), arr.ndim)), keepdims=True
        ) - output.mean(axis=tuple(range(max(0, arr.ndim - 3), arr.ndim)), keepdims=True)
    return output


def augment_periodic_batch(
    fields: Any,
    *,
    rng: np.random.Generator,
    probability_symmetry: float = 0.5,
    probability_roll: float = 0.5,
) -> np.ndarray:
    """Randomly rotate/reflect and roll each batch item while preserving array shape."""
    x = np.asarray(fields)
    if x.ndim < 4:
        raise ValueError("batch must have shape [N,...,D,H,W]")
    if not 0 <= probability_symmetry <= 1 or not 0 <= probability_roll <= 1:
        raise ValueError("augmentation probabilities must be in [0,1]")
    result = []
    for sample in x:
        value = sample
        if rng.random() < probability_symmetry:
            value = random_cube_symmetry(value, rng=rng)
        if rng.random() < probability_roll:
            value, _ = periodic_roll(value, rng=rng)
        result.append(value)
    return np.stack(result, axis=0)
