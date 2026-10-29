"""Parameter design strategies for simulator campaigns."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from nufield.inference.priors import PARAMETER_BOUNDS
from nufield.parameters import PARAMETER_NAMES


def latin_hypercube(n: int, seed: int = 0, bounds: np.ndarray | None = None) -> np.ndarray:
    """Draw a Latin-hypercube design, scaled to declared bounds.

    Latin-hypercube designs improve marginal coverage but are not the same as posterior proposals or
    guaranteed optimal experimental designs. Record the design and its seed in the experiment manifest.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    b = PARAMETER_BOUNDS if bounds is None else np.asarray(bounds, dtype=float)
    if b.shape != (len(PARAMETER_NAMES), 2) or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("bounds must have shape (n_parameters,2) and positive widths")
    sampler = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=seed)
    unit = sampler.random(n)
    return qmc.scale(unit, b[:, 0], b[:, 1]).astype(np.float32)


def unit_cube_to_parameters(unit: np.ndarray, bounds: np.ndarray | None = None) -> np.ndarray:
    u = np.asarray(unit, dtype=float)
    b = PARAMETER_BOUNDS if bounds is None else np.asarray(bounds, dtype=float)
    if u.shape[-1] != len(PARAMETER_NAMES) or np.any((u < 0) | (u > 1)):
        raise ValueError("unit coordinates must end with n_parameters and lie in [0,1]")
    return (b[:, 0] + u * (b[:, 1] - b[:, 0])).astype(np.float32)


def design_summary(samples: np.ndarray) -> dict:
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("sample matrix has incorrect shape")
    return {
        "n": int(len(x)),
        "parameter_names": PARAMETER_NAMES,
        "minimum": dict(zip(PARAMETER_NAMES, x.min(0).tolist())),
        "maximum": dict(zip(PARAMETER_NAMES, x.max(0).tolist())),
        "mean": dict(zip(PARAMETER_NAMES, x.mean(0).tolist())),
        "correlation": np.corrcoef(x, rowvar=False).tolist(),
    }
