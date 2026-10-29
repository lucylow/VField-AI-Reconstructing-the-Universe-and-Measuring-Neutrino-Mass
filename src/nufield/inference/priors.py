"""Prior sampling and support utilities for toy experiments."""

from __future__ import annotations

import numpy as np

from nufield.parameters import CosmologyParameters

PARAMETER_BOUNDS = np.array(
    [[0.06, 0.30], [0.27, 0.36], [0.74, 0.88], [0.60, 0.75], [0.92, 1.01], [0.8, 1.8]],
    dtype=np.float64,
)


def sample_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform toy prior. Not a preferred cosmology prior."""
    if n < 1:
        raise ValueError("n must be positive")
    lo, hi = PARAMETER_BOUNDS[:, 0], PARAMETER_BOUNDS[:, 1]
    return rng.uniform(lo, hi, size=(n, len(lo))).astype(np.float32)


def is_in_support(theta: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64)
    if arr.shape[-1] != PARAMETER_BOUNDS.shape[0]:
        raise ValueError("last dimension does not match parameter vector")
    return np.all(
        (arr >= PARAMETER_BOUNDS[:, 0] - atol) & (arr <= PARAMETER_BOUNDS[:, 1] + atol), axis=-1
    )


def to_parameters(row: np.ndarray) -> CosmologyParameters:
    vals = np.asarray(row).tolist()
    if len(vals) != 6:
        raise ValueError("expected six parameters")
    return CosmologyParameters(*map(float, vals))
