"""Distributional and Fourier-phase diagnostics for cosmological density fields.

These summary statistics are designed for simulation validation and ML failure analysis. One-point
moments, quantiles and phase-randomized null fields can reveal morphology differences hidden by a
similar isotropic power spectrum. They are not substitutes for a likelihood or for the full covariance
of spatially correlated voxels. In particular, naive voxel-bootstrap errors are not valid inferential
uncertainties for a correlated cosmological field.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FieldDistributionSummary:
    shape: tuple[int, ...]
    n_values: int
    mean: float
    standard_deviation: float
    skewness: float
    excess_kurtosis: float
    minimum: float
    maximum: float
    quantiles: dict[str, float]
    positive_fraction: float
    zero_fraction: float
    finite: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["shape"] = list(self.shape)
        return out


def summarize_field(
    field: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.16, 0.5, 0.84, 0.95, 0.99),
    subtract_mean_for_moments: bool = False,
) -> FieldDistributionSummary:
    """Compute standardized one-point moments and quantiles of a finite field.

    ``subtract_mean_for_moments`` centers the sample before estimating skewness/kurtosis but does not
    alter the reported mean, positive fraction, quantiles or extrema. The returned values treat voxels
    as a descriptive distribution and do not assume that voxels are independent observations.
    """
    x = np.asarray(field, float)
    if x.ndim < 1 or x.size < 3 or np.any(~np.isfinite(x)):
        raise ValueError("field must be a finite array with at least three values")
    q = np.asarray(quantiles, float)
    if q.ndim != 1 or np.any(q < 0) or np.any(q > 1) or np.any(np.diff(q) < 0):
        raise ValueError("quantiles must be an increasing sequence in [0,1]")
    flat = x.ravel()
    mean = float(np.mean(flat))
    centered = flat - mean if subtract_mean_for_moments else flat
    centered = centered - np.mean(centered)
    variance = float(np.mean(centered**2))
    std = math.sqrt(max(variance, 0.0))
    if std > 0:
        standardized = centered / std
        skew = float(np.mean(standardized**3))
        kurt = float(np.mean(standardized**4) - 3.0)
    else:
        skew = 0.0
        kurt = 0.0
    return FieldDistributionSummary(
        tuple(int(s) for s in x.shape),
        int(flat.size),
        mean,
        std,
        skew,
        kurt,
        float(np.min(flat)),
        float(np.max(flat)),
        {f"q{100 * value:g}": float(np.quantile(flat, value)) for value in q},
        float(np.mean(flat > 0)),
        float(np.mean(flat == 0)),
        True,
        (
            "Moments describe the voxel distribution; correlated voxels are not independent samples.",
            "Compare summaries on matched smoothing, grid assignment and redshift-space conventions.",
        ),
    )


def histogram_density(
    field: np.ndarray,
    *,
    bins: int | np.ndarray = 80,
    value_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return density-normalized one-point histogram and bin centers."""
    x = np.asarray(field, float)
    if x.size < 2 or np.any(~np.isfinite(x)):
        raise ValueError("field must contain at least two finite values")
    if isinstance(bins, int) and bins < 2:
        raise ValueError("bins must be >=2")
    counts, edges = np.histogram(x.ravel(), bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if np.any(~np.isfinite(counts)):
        raise ValueError("histogram produced non-finite density; check value_range and binning")
    return counts, centers


def phase_randomized_surrogate(field: np.ndarray, *, seed: int | None = None) -> np.ndarray:
    r"""Generate a real surrogate preserving the Fourier amplitudes but randomizing phases.

    The Fourier magnitude is preserved mode-by-mode to floating-point accuracy. A random real field
    supplies phases with Hermitian symmetry; preserving the original zero mode retains the mean. This
    operation destroys phase correlations while retaining the two-point spectrum, making it useful for
    separating power-spectrum information from higher-order morphology in controlled tests.
    """
    x = np.asarray(field, float)
    if x.ndim != 3 or len(set(x.shape)) != 1 or x.shape[0] < 4 or np.any(~np.isfinite(x)):
        raise ValueError("field must be a finite cubic 3-D array with side length >=4")
    rng = np.random.default_rng(seed)
    fk = np.fft.fftn(x)
    random_fk = np.fft.fftn(rng.normal(size=x.shape))
    phase = np.exp(1j * np.angle(random_fk))
    surrogate_fk = np.abs(fk) * phase
    surrogate_fk[0, 0, 0] = fk[0, 0, 0]
    surrogate = np.fft.ifftn(surrogate_fk)
    imag_max = float(np.max(np.abs(surrogate.imag)))
    if imag_max > 1e-8 * max(1.0, float(np.max(np.abs(surrogate.real)))):
        raise ArithmeticError("random phase construction did not preserve Hermitian symmetry")
    return surrogate.real


def FourierPhaseDiagnostics(
    original: np.ndarray,
    surrogate: np.ndarray,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
    box_size_mpc: float = 1.0,
) -> dict[str, float | int | bool | str]:
    """Compare Fourier amplitudes and phases for two matching cubic fields.

    The phase-coherence statistic is the circular resultant magnitude of phase differences, with 1
    denoting aligned phases and values near zero denoting broadly incoherent phases. This diagnostic is
    most informative after removing the DC mode and a low/high-k region that has insufficient support.
    """
    a, b = np.asarray(original, float), np.asarray(surrogate, float)
    if a.shape != b.shape or a.ndim != 3 or len(set(a.shape)) != 1 or np.any(~np.isfinite(a + b)):
        raise ValueError("original and surrogate must be matching finite cubic fields")
    if box_size_mpc <= 0 or not math.isfinite(box_size_mpc):
        raise ValueError("box_size_mpc must be positive")
    fa, fb = np.fft.fftn(a), np.fft.fftn(b)
    amp_a, amp_b = np.abs(fa), np.abs(fb)
    phase_delta = np.angle(fa * np.conjugate(fb))
    n = a.shape[0]
    kval = 2 * np.pi * np.fft.fftfreq(n, d=box_size_mpc / n)
    kx, ky, kz = np.meshgrid(kval, kval, kval, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    valid = kmag > 0
    if k_min_mpc_inv is not None:
        valid &= kmag >= k_min_mpc_inv
    if k_max_mpc_inv is not None:
        valid &= kmag <= k_max_mpc_inv
    if np.count_nonzero(valid) < 4:
        raise ValueError("requested k range contains too few non-DC Fourier modes")
    frac_amp_rms = float(
        np.sqrt(np.mean(((amp_b[valid] - amp_a[valid]) / np.maximum(amp_a[valid], 1e-30)) ** 2))
    )
    phase_resultant = float(np.abs(np.mean(np.exp(1j * phase_delta[valid]))))
    p_a = np.mean(amp_a[valid] ** 2)
    p_b = np.mean(amp_b[valid] ** 2)
    return {
        "n_modes": int(np.count_nonzero(valid)),
        "fractional_amplitude_rms": frac_amp_rms,
        "phase_difference_resultant": phase_resultant,
        "mean_power_ratio": float(p_b / p_a) if p_a > 0 else float("nan"),
        "amplitude_preserved_to_tolerance": bool(frac_amp_rms < 1e-10),
        "interpretation": "Use as a controlled two-point-preserving phase-randomization null; not an inference uncertainty.",
    }


def spatial_correlation_function(
    field_a: np.ndarray,
    box_size_mpc: float,
    *,
    field_b: np.ndarray | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Estimate periodic-grid correlation ξ(r) via inverse FFT of cross power.

    The returned array retains the native three-dimensional grid, with zero lag at index [0,0,0].
    It is a discrete periodic estimator; radial shell averaging and survey-window corrections are
    intentionally left to the caller.
    """
    a = np.asarray(field_a, float)
    b = a if field_b is None else np.asarray(field_b, float)
    if a.ndim != 3 or a.shape != b.shape or len(set(a.shape)) != 1 or np.any(~np.isfinite(a + b)):
        raise ValueError("fields must be matching finite cubic arrays")
    if box_size_mpc <= 0:
        raise ValueError("box_size_mpc must be positive")
    a = a - np.mean(a)
    if field_b is not None:
        b = b - np.mean(b)
    fa, fb = np.fft.fftn(a), np.fft.fftn(b)
    xi = np.fft.ifftn(fa * np.conjugate(fb)).real / np.prod(a.shape)
    if normalize:
        denom = math.sqrt(float(np.mean(a * a) * np.mean(b * b)))
        if denom <= 0:
            raise ValueError("cannot normalize correlation of a zero-variance field")
        xi = xi / denom
    return xi


def radial_average_periodic_grid(
    values: np.ndarray,
    box_size_mpc: float,
    *,
    n_bins: int = 32,
    maximum_radius_mpc: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radially average a periodic cubic grid by minimum-image spatial separation."""
    arr = np.asarray(values, float)
    if arr.ndim != 3 or len(set(arr.shape)) != 1 or np.any(~np.isfinite(arr)):
        raise ValueError("values must be a finite cubic 3-D grid")
    if box_size_mpc <= 0 or n_bins < 1:
        raise ValueError("box size and n_bins must be positive")
    n = arr.shape[0]
    dx = box_size_mpc / n
    coords = np.minimum(np.arange(n), n - np.arange(n)) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    rr = np.sqrt(x * x + y * y + z * z)
    rmax = box_size_mpc / 2 if maximum_radius_mpc is None else float(maximum_radius_mpc)
    if rmax <= 0 or rmax > math.sqrt(3) * box_size_mpc / 2:
        raise ValueError("maximum_radius_mpc lies outside the periodic-box distance range")
    edges = np.linspace(0, rmax, n_bins + 1)
    which = np.searchsorted(edges, rr.ravel(), side="right") - 1
    which[rr.ravel() == edges[-1]] = n_bins - 1
    good = (which >= 0) & (which < n_bins)
    count = np.bincount(which[good], minlength=n_bins)
    total = np.bincount(which[good], weights=arr.ravel()[good], minlength=n_bins)
    mean = np.divide(total, count, out=np.zeros(n_bins), where=count > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, mean, count


__all__ = [
    "FieldDistributionSummary",
    "summarize_field",
    "histogram_density",
    "phase_randomized_surrogate",
    "FourierPhaseDiagnostics",
    "spatial_correlation_function",
    "radial_average_periodic_grid",
]
