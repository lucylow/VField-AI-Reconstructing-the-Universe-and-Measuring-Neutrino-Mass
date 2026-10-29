"""Classical discrete Gaussian-summary likelihood baseline for controlled mass-grid experiments.

The estimator is intentionally restricted: it estimates a mean summary at each mass and a pooled
within-mass covariance. It expects all nuisance parameters to be fixed, or nuisance variation to be
controlled externally before fitting. It returns a discrete mass posterior on observed simulation
mass levels and never pretends a sparse grid defines a validated continuous posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from nufield.benchmarking.covariance import (
    estimate_regularized_covariance,
    multivariate_gaussian_logpdf,
    normalized_discrete_grid_prior,
)


@dataclass
class ClassicalMassGridModel:
    mass_grid: np.ndarray
    means: np.ndarray
    covariance: np.ndarray
    prior_weights: np.ndarray
    covariance_report: dict[str, Any]
    fit_summary: dict[str, Any]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        mass_ev: np.ndarray,
        *,
        group_ids: np.ndarray | None = None,
        nuisance_parameters: np.ndarray | None = None,
        prior_bounds: tuple[float, float],
        shrinkage: float = 0.15,
        diagonal_floor: float = 1e-5,
        min_replicates_per_mass: int = 3,
        require_fixed_nuisance: bool = True,
        nuisance_tolerance: float = 1e-8,
    ) -> ClassicalMassGridModel:
        x = np.asarray(features, dtype=np.float64)
        m = np.asarray(mass_ev, dtype=np.float64).reshape(-1)
        groups = (
            np.asarray(group_ids).astype(str).reshape(-1)
            if group_ids is not None
            else np.asarray([f"row{i}" for i in range(len(m))])
        )
        if (
            x.ndim != 2
            or len(x) != len(m)
            or len(groups) != len(m)
            or not np.isfinite(x).all()
            or not np.isfinite(m).all()
        ):
            raise ValueError("finite features, mass labels and group IDs must align")
        if min_replicates_per_mass < 2:
            raise ValueError("min_replicates_per_mass must be at least 2")
        lo, hi = map(float, prior_bounds)
        if not lo < hi or m.min() < lo - 1e-10 or m.max() > hi + 1e-10:
            raise ValueError("mass labels must lie inside declared prior bounds")
        if nuisance_parameters is not None:
            nuisance = np.asarray(nuisance_parameters, dtype=float)
            if nuisance.ndim != 2 or len(nuisance) != len(m) or not np.isfinite(nuisance).all():
                raise ValueError("nuisance_parameters must be finite [N,Q]")
            if require_fixed_nuisance and np.any(np.ptp(nuisance, axis=0) > nuisance_tolerance):
                raise ValueError(
                    "classical mass-grid baseline assumes fixed nuisance parameters; use matched strata, a validated emulator/likelihood, or NPE for varying nuisance parameters"
                )
        grid = np.unique(m)
        if len(grid) < 2:
            raise ValueError("mass grid requires at least two target levels")
        means = []
        residuals = []
        counts = []
        group_counts = []
        for mass in grid:
            idx = np.flatnonzero(np.isclose(m, mass, rtol=0, atol=1e-10))
            independent = np.unique(groups[idx]).size
            if independent < min_replicates_per_mass:
                raise ValueError(
                    f"mass={mass:g} eV has only {independent} independent groups; need >= {min_replicates_per_mass}"
                )
            mu = x[idx].mean(axis=0)
            means.append(mu)
            residuals.append(x[idx] - mu)
            counts.append(len(idx))
            group_counts.append(int(independent))
        pooled = np.concatenate(residuals, axis=0)
        cov, cov_report = estimate_regularized_covariance(
            pooled, shrinkage=shrinkage, diagonal_floor=diagonal_floor
        )
        weights = normalized_discrete_grid_prior(grid, lo, hi)
        return cls(
            grid,
            np.asarray(means),
            cov,
            weights,
            cov_report.to_dict(),
            {
                "n_rows": len(m),
                "n_features": x.shape[1],
                "counts_by_mass": counts,
                "independent_groups_by_mass": group_counts,
                "n_independent_groups": int(np.unique(groups).size),
                "shrinkage": shrinkage,
                "prior_bounds_ev": [lo, hi],
                "grid_spans_prior_bounds": bool(
                    np.isclose(grid[0], lo) and np.isclose(grid[-1], hi)
                ),
                "prior_approximation": "uniform-continuous prior integrated over nearest-grid/Voronoi cells",
                "nuisance_fixed_assumption": bool(require_fixed_nuisance),
                "posterior_type": "discrete mass grid; no interpolation",
            },
        )

    def posterior_weights(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=float).reshape(-1)
        if x.shape != (self.means.shape[1],) or not np.isfinite(x).all():
            raise ValueError("observation must be finite and match feature width")
        logl = np.asarray(
            [multivariate_gaussian_logpdf(x, mu, self.covariance) for mu in self.means]
        )
        logp = logl + np.log(self.prior_weights)
        return np.exp(logp - logsumexp(logp))

    def sample_posterior(
        self, observation: np.ndarray, *, n_samples: int = 1000, seed: int = 42
    ) -> np.ndarray:
        if n_samples < 2:
            raise ValueError("n_samples must be at least two")
        weights = self.posterior_weights(observation)
        rng = np.random.default_rng(seed)
        return rng.choice(self.mass_grid, size=n_samples, replace=True, p=weights)[:, None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "mass_grid_ev": self.mass_grid.tolist(),
            "means": self.means.tolist(),
            "covariance": self.covariance.tolist(),
            "prior_weights": self.prior_weights.tolist(),
            "covariance_report": self.covariance_report,
            "fit_summary": self.fit_summary,
            "scientific_approval": False,
            "warning": "Controlled discrete Gaussian summary-likelihood baseline; not valid with unmodelled nuisance changes or survey-window effects.",
        }
