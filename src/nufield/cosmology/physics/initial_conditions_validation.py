"""Statistical validation tools for Gaussian cosmological initial density fields.

These diagnostics test the field-generation statistics and Fourier conventions. A correct Gaussian
field is not yet a complete N-body initial condition: cosmological transfer functions, 2LPT/Zel'dovich
displacements, velocity fields, neutrino particles, gauge conventions, and particle loads must also be
consistent with the simulation being targeted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kstest, norm

from nufield.cosmology.physics.initial_conditions import generate_gaussian_density_field
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class InitialFieldReport:
    seed: int
    grid_size: int
    box_size_mpc: float
    mean: float
    standard_deviation: float
    skewness: float
    excess_kurtosis: float
    normalized_mode_power_mean: float
    normalized_mode_power_variance: float
    gaussian_ks_statistic: float
    gaussian_ks_p_value: float
    zero_mode_abs: float
    finite: bool
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "grid_size": self.grid_size,
            "box_size_mpc": self.box_size_mpc,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "normalized_mode_power_mean": self.normalized_mode_power_mean,
            "normalized_mode_power_variance": self.normalized_mode_power_variance,
            "gaussian_ks_statistic": self.gaussian_ks_statistic,
            "gaussian_ks_p_value": self.gaussian_ks_p_value,
            "zero_mode_abs": self.zero_mode_abs,
            "finite": self.finite,
            "caveats": list(self.caveats),
        }


def validate_gaussian_initial_field(
    spectrum: MatterPowerSpectrum,
    grid_size: int,
    box_size_mpc: float,
    *,
    seed: int = 0,
    strict_k_support: bool = True,
) -> tuple[np.ndarray, InitialFieldReport]:
    """Generate a field and report first-pass real-space Gaussian and mode-statistic diagnostics.

    The normalized mode-power distribution is evaluated only over non-empty non-zero Fourier modes
    after comparison to the spectrum interpolated at the same Fourier magnitudes. Finite-grid binning,
    the real-field Hermitian relation and mode correlations mean the reported scalar moments are
    diagnostics, not a single exact hypothesis test for the entire correlated field.
    """
    field = generate_gaussian_density_field(
        spectrum, grid_size, box_size_mpc, seed=seed, strict_k_support=strict_k_support
    )
    if np.any(~np.isfinite(field)):
        raise FloatingPointError("generated field contains non-finite values")
    flat = np.asarray(field, float).ravel()
    std = float(np.std(flat, ddof=1))
    if std <= 0:
        raise ValueError("field has zero variance; spectrum may be zero or unsupported")
    standardized = (flat - np.mean(flat)) / std
    skew = float(np.mean(standardized**3))
    kurt = float(np.mean(standardized**4) - 3.0)
    ks = kstest(standardized, norm.cdf)
    fk = np.fft.fftn(field)
    n = grid_size**3
    volume = box_size_mpc**3
    pk = np.abs((volume / n) * fk) ** 2 / volume
    kf = 2 * np.pi * np.fft.fftfreq(grid_size, d=box_size_mpc / grid_size)
    kx, ky, kz = np.meshgrid(kf, kf, kf, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    nonzero = kmag > 0
    inside = nonzero & (kmag >= spectrum.k_mpc_inv[0]) & (kmag <= spectrum.k_mpc_inv[-1])
    if not np.any(inside):
        norm_mean, norm_var = float("nan"), float("nan")
    else:
        expected = spectrum.evaluate(kmag[inside])
        norm_mode = pk[inside] / expected
        norm_mean = float(np.mean(norm_mode))
        norm_var = float(np.var(norm_mode, ddof=1)) if norm_mode.size > 1 else float("nan")
    report = InitialFieldReport(
        seed,
        grid_size,
        box_size_mpc,
        float(np.mean(flat)),
        std,
        skew,
        kurt,
        norm_mean,
        norm_var,
        float(ks.statistic),
        float(ks.pvalue),
        float(abs(fk[0, 0, 0])),
        bool(np.all(np.isfinite(field))),
        (
            "The field is a Gaussian linear realization, not an evolved nonlinear density field.",
            "Fourier modes of a real field are conjugate paired and not independent as a full array.",
            "Compare ensemble averages across many independent seeds for a meaningful mode-power test.",
        ),
    )
    return field, report


def ensemble_initial_field_check(
    spectrum: MatterPowerSpectrum,
    grid_size: int,
    box_size_mpc: float,
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
) -> dict[str, object]:
    """Run independent initial-field summaries and aggregate seed-level statistics."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("use at least two unique non-negative seeds")
    reports = [
        validate_gaussian_initial_field(spectrum, grid_size, box_size_mpc, seed=s)[1] for s in seeds
    ]
    means = np.asarray([r.mean for r in reports])
    stds = np.asarray([r.standard_deviation for r in reports])
    powers = np.asarray([r.normalized_mode_power_mean for r in reports])
    return {
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "field_mean_across_seeds": float(means.mean()),
        "field_mean_scatter": float(means.std(ddof=1)),
        "field_std_mean": float(stds.mean()),
        "field_std_scatter": float(stds.std(ddof=1)),
        "normalized_power_mean": float(np.nanmean(powers)),
        "normalized_power_seed_scatter": float(np.nanstd(powers, ddof=1)),
        "per_seed": [r.to_dict() for r in reports],
        "warning": "Few seeds give a weak ensemble test; run a larger campaign before interpreting deviations.",
    }


__all__ = ["InitialFieldReport", "validate_gaussian_initial_field", "ensemble_initial_field_check"]
