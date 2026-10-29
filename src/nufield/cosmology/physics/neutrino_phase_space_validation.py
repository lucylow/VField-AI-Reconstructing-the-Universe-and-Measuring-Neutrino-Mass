"""Statistical checks for Fermi--Dirac relic-neutrino momentum samplers.

A successful thermal momentum sampler does not make a complete massive-neutrino N-body initialization.
The validator checks the one-particle homogeneous distribution only; density perturbations, bulk
velocity fields, transfer functions, particle weights, and gravitational evolution remain separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import zeta
from scipy.stats import kstest

from nufield.cosmology.physics.neutrino_phase_space import (
    fermi_dirac_momentum_pdf,
    sample_dimensionless_momentum,
    sample_neutrino_thermal_velocities,
)


@dataclass(frozen=True, slots=True)
class ThermalSamplerReport:
    n_samples: int
    seed: int
    empirical_mean_q: float
    theoretical_mean_q: float
    standardized_mean_offset: float
    empirical_rms_q: float
    theoretical_rms_q: float
    ks_statistic: float
    ks_p_value: float
    velocity_mean_vector_km_s: tuple[float, float, float]
    speed_quantiles_km_s: tuple[float, float, float]
    passed_configured_checks: bool
    thresholds: dict[str, float]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_samples": self.n_samples,
            "seed": self.seed,
            "empirical_mean_q": self.empirical_mean_q,
            "theoretical_mean_q": self.theoretical_mean_q,
            "standardized_mean_offset": self.standardized_mean_offset,
            "empirical_rms_q": self.empirical_rms_q,
            "theoretical_rms_q": self.theoretical_rms_q,
            "ks_statistic": self.ks_statistic,
            "ks_p_value": self.ks_p_value,
            "velocity_mean_vector_km_s": list(self.velocity_mean_vector_km_s),
            "speed_quantiles_km_s": list(self.speed_quantiles_km_s),
            "passed_configured_checks": self.passed_configured_checks,
            "thresholds": self.thresholds,
            "caveats": list(self.caveats),
        }


def theoretical_fd_momentum_moments() -> dict[str, float]:
    """Return analytic number-weighted q=p/T moments for q²/(exp(q)+1)."""
    mean = 7 * math.pi**4 / (180 * float(zeta(3, 1)))
    second = 15 * float(zeta(5, 1)) / float(zeta(3, 1))
    return {
        "mean_q": mean,
        "second_moment_q2": second,
        "rms_q": math.sqrt(second),
        "variance_q": second - mean**2,
    }


def validate_thermal_momentum_sampler(
    n_samples: int = 100_000,
    *,
    seed: int = 2026,
    q_max: float = 40.0,
    table_size: int = 100_000,
    max_ks_statistic: float = 0.012,
    max_standardized_mean_offset: float = 5.0,
) -> ThermalSamplerReport:
    """Compare reproducible inverse-CDF draws with the analytical Fermi--Dirac momentum law."""
    if n_samples < 100 or seed < 0:
        raise ValueError("n_samples>=100 and non-negative seed are required")
    if not 0 < max_ks_statistic < 1 or max_standardized_mean_offset <= 0:
        raise ValueError("diagnostic thresholds must be positive and KS threshold <1")
    samples = sample_dimensionless_momentum(
        n_samples, seed=seed, q_max=q_max, table_size=table_size
    )
    qgrid = np.linspace(0, q_max, table_size)
    pdf = fermi_dirac_momentum_pdf(qgrid)
    cdf = np.concatenate(([0.0], cumulative_trapezoid(pdf, qgrid)))
    cdf /= cdf[-1]
    # `kstest` assumes a continuous null CDF; numerical CDF interpolation error is negligible here
    # for the selected dense table, but should be checked if q_max/table_size are changed drastically.
    from scipy.interpolate import PchipInterpolator

    cdf_interp = PchipInterpolator(qgrid, cdf, extrapolate=False)
    ks = kstest(samples, lambda x: np.clip(cdf_interp(x), 0, 1))
    theory = theoretical_fd_momentum_moments()
    mean = float(np.mean(samples))
    std_mean = math.sqrt(theory["variance_q"] / n_samples)
    mean_offset = abs(mean - theory["mean_q"]) / std_mean
    velocities = sample_neutrino_thermal_velocities(0.1, 0.0, n_samples, seed=seed)
    velocity_mean = np.mean(velocities["velocity_km_s"], axis=0)
    speeds = velocities["speed_km_s"]
    passed = bool(ks.statistic <= max_ks_statistic and mean_offset <= max_standardized_mean_offset)
    return ThermalSamplerReport(
        int(n_samples),
        int(seed),
        mean,
        theory["mean_q"],
        mean_offset,
        float(np.sqrt(np.mean(samples**2))),
        theory["rms_q"],
        float(ks.statistic),
        float(ks.pvalue),
        tuple(float(x) for x in velocity_mean),
        tuple(float(x) for x in np.quantile(speeds, [0.1, 0.5, 0.9])),
        passed,
        {
            "max_ks_statistic": max_ks_statistic,
            "max_standardized_mean_offset": max_standardized_mean_offset,
        },
        (
            "The KS p-value is sample-size dependent and should not be used alone as a release gate.",
            "The analytic CDF is numerically tabulated before testing the inverse-CDF samples.",
            "This covers thermal momenta and isotropic directions only, not perturbation or bulk velocities.",
        ),
    )


def check_rng_reproducibility(seed: int = 123, n_samples: int = 1024) -> bool:
    """Return whether a fixed seed reproduces the same thermal momentum sequence exactly."""
    a = sample_dimensionless_momentum(n_samples, seed=seed)
    b = sample_dimensionless_momentum(n_samples, seed=seed)
    return bool(np.array_equal(a, b))


__all__ = [
    "ThermalSamplerReport",
    "theoretical_fd_momentum_moments",
    "validate_thermal_momentum_sampler",
    "check_rng_reproducibility",
]
