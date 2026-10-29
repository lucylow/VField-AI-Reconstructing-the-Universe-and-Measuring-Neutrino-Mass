"""Small deterministic numeric helpers used by demo-data generators.

These helpers intentionally use Python's standard library. They do not depend
on NumPy's global random state and cannot alter a scientific experiment's RNG.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Sequence
from statistics import mean, median, pstdev
from typing import Any


def make_rng(seed: int) -> random.Random:
    """Create a private RNG whose state is local to a demo bundle."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Demo seed must be an integer.")
    return random.Random(seed & 0xFFFFFFFF)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a finite scalar to a closed interval."""
    if low > high:
        raise ValueError("Lower bound cannot exceed upper bound.")
    if not math.isfinite(float(value)):
        raise ValueError("Cannot clamp a non-finite value into a demo range.")
    return min(high, max(low, float(value)))


def rounded(value: float, digits: int = 6) -> float:
    """Round a numeric value and avoid accidentally serializing NaN/Infinity."""
    if not math.isfinite(float(value)):
        raise ValueError("Generated numeric values must be finite.")
    return round(float(value), digits)


def safe_count(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    """Parse and clamp integer-like user options for a generation request."""
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(maximum, max(minimum, numeric))


def normal(rng: random.Random, mean_value: float = 0.0, sigma: float = 1.0) -> float:
    """Return one Gaussian draw while guarding against negative scale."""
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError("Normal-distribution scale must be finite and non-negative.")
    return rng.gauss(mean_value, sigma)


def uniform(rng: random.Random, low: float, high: float) -> float:
    """Return one uniform scalar from a private RNG."""
    if low > high:
        raise ValueError("Uniform low bound cannot exceed high bound.")
    return rng.uniform(low, high)


def quantile(values: Sequence[float] | Iterable[float], q: float) -> float:
    """Linearly interpolated sample quantile without a NumPy dependency."""
    if not 0 <= q <= 1:
        raise ValueError("Quantile q must lie between zero and one.")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def descriptive_stats(values: Sequence[float] | Iterable[float]) -> dict[str, float | int | None]:
    """Summarize a sequence, returning explicit nulls for an empty sequence."""
    items = [float(value) for value in values if math.isfinite(float(value))]
    if not items:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std_population": None,
            "min": None,
            "max": None,
            "q16": None,
            "q84": None,
        }
    return {
        "count": len(items),
        "mean": rounded(mean(items), 8),
        "median": rounded(median(items), 8),
        "std_population": rounded(pstdev(items), 8),
        "min": rounded(min(items), 8),
        "max": rounded(max(items), 8),
        "q16": rounded(quantile(items, 0.16), 8),
        "q84": rounded(quantile(items, 0.84), 8),
    }


def stable_digest(value: str) -> str:
    """Create a stable short digest for traceable synthetic artifact IDs."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def split_counts(
    total: int, train_fraction: float = 0.65, validation_fraction: float = 0.15
) -> dict[str, int]:
    """Partition a count with no lost or duplicated rows."""
    if total < 0:
        raise ValueError("Total row count cannot be negative.")
    if not 0 <= train_fraction <= 1 or not 0 <= validation_fraction <= 1:
        raise ValueError("Split fractions must lie in [0, 1].")
    if train_fraction + validation_fraction > 1:
        raise ValueError("Train + validation fractions cannot exceed one.")
    train = int(total * train_fraction)
    validation = int(total * validation_fraction)
    return {"train": train, "validation": validation, "test": total - train - validation}


def mean_or_none(values: Iterable[float]) -> float | None:
    """Convenience helper for optional means."""
    items = list(values)
    return rounded(mean(items), 8) if items else None


def finite_range(values: Iterable[float]) -> tuple[float, float] | None:
    """Return the finite min/max, or None if no finite values exist."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return (min(finite), max(finite)) if finite else None
