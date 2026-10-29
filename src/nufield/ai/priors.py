"""Prior specifications and reproducible quasi-random simulation designs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import qmc

PriorKind = Literal["uniform", "log_uniform"]


@dataclass(frozen=True)
class ParameterPrior:
    name: str
    low: float
    high: float
    units: str
    kind: PriorKind = "uniform"

    def __post_init__(self):
        if not self.name.strip() or not self.units.strip():
            raise ValueError("prior name and units must be declared")
        if not np.isfinite(self.low) or not np.isfinite(self.high) or self.high <= self.low:
            raise ValueError("prior bounds must be finite with high > low")
        if self.kind not in {"uniform", "log_uniform"}:
            raise ValueError("kind must be uniform or log_uniform")
        if self.kind == "log_uniform" and self.low <= 0:
            raise ValueError("log-uniform prior requires positive lower bound")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PriorSpecification:
    parameters: tuple[ParameterPrior, ...]
    identifier: str = "prior-v1"

    def __post_init__(self):
        if not self.parameters:
            raise ValueError("at least one parameter prior is required")
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def low(self) -> np.ndarray:
        return np.asarray([p.low for p in self.parameters], float)

    @property
    def high(self) -> np.ndarray:
        return np.asarray([p.high for p in self.parameters], float)

    @property
    def units(self) -> dict[str, str]:
        return {p.name: p.units for p in self.parameters}

    def validate(self, values: Any) -> np.ndarray:
        x = np.asarray(values, float)
        if x.shape[-1] != len(self.parameters) or not np.isfinite(x).all():
            raise ValueError(
                "parameters must be finite and have final dimension equal to prior width"
            )
        if np.any(x < self.low) or np.any(x > self.high):
            raise ValueError("parameter point lies outside declared prior support")
        if any(
            p.kind == "log_uniform" and np.any(x[..., i] <= 0)
            for i, p in enumerate(self.parameters)
        ):
            raise ValueError("log-uniform prior parameters must be positive")
        return x

    def sample(
        self, n: int, *, method: str = "sobol", seed: int = 42, scramble: bool = True
    ) -> np.ndarray:
        """Draw points uniformly in the declared coordinate measure.

        For log_uniform dimensions, a uniform draw is made in log(parameter). Latin hypercube and
        Sobol samples can improve space coverage relative to iid Monte Carlo for deterministic design,
        but they are not posterior samples.
        """
        if n < 1:
            raise ValueError("n must be positive")
        d = len(self.parameters)
        if method == "sobol":
            engine = qmc.Sobol(d=d, scramble=scramble, seed=seed)
            # Generate a balanced power-of-two Sobol block, then retain the requested prefix.
            # This preserves deterministic first-n points without emitting SciPy's warning for
            # non-power-of-two counts. The truncated set should not be described as a fully balanced
            # Sobol net; callers should still prefer power-of-two n for design-of-experiments work.
            m = int(np.ceil(np.log2(n)))
            u = engine.random_base2(m)[:n]
        elif method == "latin_hypercube":
            u = qmc.LatinHypercube(d=d, seed=seed, scramble=scramble).random(n)
        elif method == "random":
            u = np.random.default_rng(seed).random((n, d))
        else:
            raise ValueError("method must be sobol/latin_hypercube/random")
        result = np.empty_like(u)
        for j, prior in enumerate(self.parameters):
            if prior.kind == "uniform":
                result[:, j] = prior.low + u[:, j] * (prior.high - prior.low)
            else:
                result[:, j] = np.exp(np.log(prior.low) + u[:, j] * np.log(prior.high / prior.low))
        return result

    def log_prob(self, values: Any) -> np.ndarray:
        """Return log prior density per row in the specified physical coordinate measure."""
        x = np.asarray(values, float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(self.parameters):
            raise ValueError("values must be [N,P]")
        in_support = np.all((x >= self.low) & (x <= self.high), axis=1) & np.isfinite(x).all(axis=1)
        out = np.full(len(x), -np.inf)
        if not np.any(in_support):
            return out
        density = np.zeros(int(in_support.sum()), float)
        xin = x[in_support]
        for j, p in enumerate(self.parameters):
            if p.kind == "uniform":
                density -= np.log(p.high - p.low)
            else:
                if np.any(xin[:, j] <= 0):
                    in_support[np.flatnonzero(in_support)[xin[:, j] <= 0]] = False
                    continue
                density -= np.log(xin[:, j]) + np.log(np.log(p.high / p.low))
        valid_rows = np.flatnonzero(in_support)
        if len(valid_rows) == len(density):
            out[valid_rows] = density
        else:
            valid = np.all(np.isfinite(xin) & (xin > 0), axis=1)
            out[valid_rows[valid]] = density[valid]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "parameters": [p.to_dict() for p in self.parameters],
            "parameter_names": list(self.names),
            "units": self.units,
            "warning": "Changing the prior changes the target posterior; document prior revisions.",
        }


def design_quality_summary(points: Any, prior: PriorSpecification) -> dict[str, Any]:
    """Report space-filling diagnostics in prior-normalized coordinates."""
    x = prior.validate(points)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("points must be [N>=2,P]")
    scaled = np.empty_like(x)
    for j, p in enumerate(prior.parameters):
        if p.kind == "log_uniform":
            scaled[:, j] = (np.log(x[:, j]) - np.log(p.low)) / np.log(p.high / p.low)
        else:
            scaled[:, j] = (x[:, j] - p.low) / (p.high - p.low)
    # Small n^2 matrix is acceptable for diagnostics; avoid this on millions of points.
    from scipy.spatial.distance import pdist

    distances = pdist(scaled)
    nearest = []
    for i in range(len(scaled)):
        d = np.linalg.norm(scaled - scaled[i], axis=1)
        d[i] = np.inf
        nearest.append(float(d.min()))
    return {
        "n_points": int(len(x)),
        "dimension": int(x.shape[1]),
        "nearest_neighbor_min": float(np.min(nearest)),
        "nearest_neighbor_median": float(np.median(nearest)),
        "pair_distance_median": float(np.median(distances)),
        "coordinate_mean": scaled.mean(axis=0).tolist(),
        "coordinate_std": scaled.std(axis=0).tolist(),
        "warning": "These are design diagnostics, not proofs of adequate posterior coverage.",
    }
