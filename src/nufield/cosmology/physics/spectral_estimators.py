"""FFT estimators for periodic-box isotropic and anisotropic clustering observables.

The routines in this module deliberately require the box size and unit convention at the call site.
They operate on dimensionless real-space fields sampled uniformly in a periodic Cartesian cube. They
are not survey estimators: masks, radial selection, fiber assignment, wide-angle effects, and survey
window convolution must be modeled separately.

Normalization convention
------------------------
NumPy's forward FFT is unnormalised. For a field sampled on Ncell voxels in a box of volume V,
we define the physical Fourier coefficient ``delta_k = V/Ncell * fftn(delta)``. The estimator
``P(k) = |delta_k|^2/V`` then has units Mpc^3 when coordinates are in Mpc. For cross-power we use
``Re(delta_a * conj(delta_b))/V``, which may be negative in an individual bin.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import legval


@dataclass(frozen=True, slots=True)
class BinnedSpectrum:
    """Binned 3-D spectrum with explicit units and mode counts.

    Cross spectra are permitted to have negative bandpowers. Auto-spectra should be non-negative
    within sampling noise; a negative auto-bandpower usually indicates an invalid preprocessing or
    estimator implementation.
    """

    k_centers_mpc_inv: np.ndarray
    power_mpc3: np.ndarray
    mode_counts: np.ndarray
    bin_edges_mpc_inv: np.ndarray
    box_size_mpc: float
    estimator: str = "isotropic_fft"
    is_cross: bool = False
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        k = np.asarray(self.k_centers_mpc_inv, dtype=float)
        p = np.asarray(self.power_mpc3, dtype=float)
        n = np.asarray(self.mode_counts, dtype=np.int64)
        e = np.asarray(self.bin_edges_mpc_inv, dtype=float)
        if k.ndim != 1 or p.shape != k.shape or n.shape != k.shape:
            raise ValueError("k, power and mode_counts must be matching 1-D vectors")
        if e.shape != (k.size + 1,):
            raise ValueError("bin_edges must have one more entry than the bandpower arrays")
        if k.size == 0 or np.any(~np.isfinite(k + p)) or np.any(k <= 0):
            raise ValueError("bandpowers must contain finite values and positive k centers")
        if np.any(n < 0) or self.box_size_mpc <= 0 or not math.isfinite(self.box_size_mpc):
            raise ValueError("mode counts must be non-negative and box_size_mpc positive")
        if np.any(np.diff(e) <= 0) or np.any(np.diff(k) <= 0):
            raise ValueError("k centers and bin edges must be strictly increasing")
        if np.any(p[n == 0] != 0):
            raise ValueError("empty bins must have zero power and zero mode count")
        object.__setattr__(self, "k_centers_mpc_inv", k.copy())
        object.__setattr__(self, "power_mpc3", p.copy())
        object.__setattr__(self, "mode_counts", n.copy())
        object.__setattr__(self, "bin_edges_mpc_inv", e.copy())

    @property
    def valid_bins(self) -> np.ndarray:
        """Boolean mask for bins containing at least one Fourier mode."""
        return self.mode_counts > 0

    @property
    def dimensionless_power(self) -> np.ndarray:
        r"""Return :math:`\Delta^2(k)=k^3P(k)/(2\pi^2)` for valid bins."""
        return self.k_centers_mpc_inv**3 * self.power_mpc3 / (2.0 * math.pi**2)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly estimator results, retaining units and provenance."""
        return {
            "k_centers_mpc_inv": self.k_centers_mpc_inv.tolist(),
            "power_mpc3": self.power_mpc3.tolist(),
            "mode_counts": self.mode_counts.tolist(),
            "bin_edges_mpc_inv": self.bin_edges_mpc_inv.tolist(),
            "box_size_mpc": float(self.box_size_mpc),
            "estimator": self.estimator,
            "is_cross": bool(self.is_cross),
            "metadata": dict(self.metadata),
            "units": {"k": "Mpc^-1", "P": "Mpc^3"},
        }


@dataclass(frozen=True, slots=True)
class MultipoleSpectrum:
    """Anisotropic power multipoles measured from a periodic field."""

    k_centers_mpc_inv: np.ndarray
    multipoles: dict[int, np.ndarray]
    mode_counts: np.ndarray
    k_edges_mpc_inv: np.ndarray
    box_size_mpc: float
    los_axis: int
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        k = np.asarray(self.k_centers_mpc_inv, float)
        counts = np.asarray(self.mode_counts, np.int64)
        edges = np.asarray(self.k_edges_mpc_inv, float)
        if k.ndim != 1 or counts.shape != k.shape or edges.shape != (k.size + 1,):
            raise ValueError("invalid k/count/edge shapes")
        copied: dict[int, np.ndarray] = {}
        for ell, values in self.multipoles.items():
            arr = np.asarray(values, float)
            if not isinstance(ell, int) or ell < 0 or arr.shape != k.shape:
                raise ValueError("multipole keys must be non-negative integers and arrays match k")
            if np.any(~np.isfinite(arr)):
                raise ValueError("multipole contains non-finite values")
            copied[ell] = arr.copy()
        if not copied:
            raise ValueError("at least one multipole is required")
        if self.los_axis not in (0, 1, 2) or self.box_size_mpc <= 0:
            raise ValueError("los_axis must be 0, 1 or 2 and box size positive")
        object.__setattr__(self, "k_centers_mpc_inv", k.copy())
        object.__setattr__(self, "mode_counts", counts.copy())
        object.__setattr__(self, "k_edges_mpc_inv", edges.copy())
        object.__setattr__(self, "multipoles", copied)


def default_log_k_edges(
    box_size_mpc: float,
    grid_size: int,
    *,
    n_bins: int = 24,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
    nyquist_fraction: float = 0.8,
) -> np.ndarray:
    """Construct logarithmic Fourier-shell edges respecting box and grid scales.

    The default lower edge is the fundamental mode 2π/L. The default upper edge is a configurable
    fraction of the axis Nyquist frequency πN/L. The spherical shell includes corner modes above the
    axis Nyquist frequency, so the recommended default keeps a conservative margin below that limit.
    """
    if not math.isfinite(box_size_mpc) or box_size_mpc <= 0 or grid_size < 4 or n_bins < 1:
        raise ValueError("box size must be positive, grid_size>=4, and n_bins>=1")
    if not (0 < nyquist_fraction <= 1):
        raise ValueError("nyquist_fraction must lie in (0,1]")
    lo = 2 * math.pi / box_size_mpc if k_min_mpc_inv is None else float(k_min_mpc_inv)
    hi = (
        nyquist_fraction * math.pi * grid_size / box_size_mpc
        if k_max_mpc_inv is None
        else float(k_max_mpc_inv)
    )
    if lo <= 0 or hi <= lo:
        raise ValueError("require 0 < k_min < k_max")
    return np.geomspace(lo, hi, int(n_bins) + 1)


def _validate_field(field: np.ndarray, *, name: str = "field") -> np.ndarray:
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 3 or len(set(arr.shape)) != 1 or arr.shape[0] < 4:
        raise ValueError(f"{name} must be a cubic 3-D array with at least 4 cells per axis")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _fourier_geometry(
    n: int, box_size_mpc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not math.isfinite(box_size_mpc) or box_size_mpc <= 0:
        raise ValueError("box_size_mpc must be positive and finite")
    dx = box_size_mpc / n
    k1 = 2 * math.pi * np.fft.fftfreq(n, d=dx)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij", sparse=False)
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    volume = box_size_mpc**3
    return kx, ky, kz, volume


def _mode_bin_indices(kmag: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, kmag.ravel(), side="right") - 1
    # Put a mode exactly on the final edge into the last bin, following histogram convention.
    idx[kmag.ravel() == edges[-1]] = len(edges) - 2
    invalid = (kmag.ravel() < edges[0]) | (kmag.ravel() > edges[-1])
    idx[invalid] = -1
    return idx


def estimate_isotropic_power(
    field_a: np.ndarray,
    box_size_mpc: float,
    *,
    field_b: np.ndarray | None = None,
    bin_edges_mpc_inv: np.ndarray | None = None,
    n_bins: int = 24,
    subtract_mean: bool = True,
    deconvolve_assignment: bool = False,
    assignment_window: np.ndarray | None = None,
    nyquist_fraction: float = 0.8,
    metadata: Iterable[tuple[str, str]] = (),
) -> BinnedSpectrum:
    r"""Estimate auto- or cross-power from a pair of periodic 3-D fields.

    Fields are assumed to be dimensionless overdensity-like maps on a uniform grid. For counts or
    dimensional density, normalize to the intended overdensity before calling this function. If a
    field has been constructed with CIC/TSC assignment, pass the Fourier assignment window and opt in
    to deconvolution; modes where the assignment window is close to zero are dropped for stability.
    Shot noise is not automatically subtracted because its correct form depends on the catalog and
    weighting scheme.
    """
    a = _validate_field(field_a, name="field_a")
    b = a if field_b is None else _validate_field(field_b, name="field_b")
    if b.shape != a.shape:
        raise ValueError("field_a and field_b must have matching shape")
    n = a.shape[0]
    edges = (
        default_log_k_edges(
            box_size_mpc,
            n,
            n_bins=n_bins,
            k_max_mpc_inv=nyquist_fraction * math.pi * n / box_size_mpc,
        )
        if bin_edges_mpc_inv is None
        else np.asarray(bin_edges_mpc_inv, float)
    )
    if edges.ndim != 1 or len(edges) < 2 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges_mpc_inv must be positive and strictly increasing")
    if subtract_mean:
        a = a - np.mean(a)
        if field_b is not None:
            b = b - np.mean(b)
    fa = np.fft.fftn(a)
    fb = fa if field_b is None else np.fft.fftn(b)
    _, _, _, volume = _fourier_geometry(n, box_size_mpc)
    # δ_k physical = V/Ncell * fft(δ), so P = δ_k δ_k*/V.
    prefactor = volume / n**6
    raw_power = np.real(fa * np.conjugate(fb)) * prefactor
    k1 = 2 * math.pi * np.fft.fftfreq(n, d=box_size_mpc / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    if deconvolve_assignment:
        if assignment_window is None:
            raise ValueError("assignment_window is required for deconvolution")
        win = np.asarray(assignment_window, float)
        if win.shape != a.shape or np.any(~np.isfinite(win)):
            raise ValueError("assignment_window must be finite and match field shape")
        denom = win**2
        # The same assignment window is applied to both legs of a cross-spectrum.
        # For distinct windows, divide by their product in a future explicit two-window API.
        valid_window = np.abs(denom) > 1e-3
        raw_power = np.divide(raw_power, denom, out=np.zeros_like(raw_power), where=valid_window)
        kmag = np.where(valid_window, kmag, np.nan)
    flat_k = kmag.ravel()
    flat_p = raw_power.ravel()
    finite = np.isfinite(flat_k) & np.isfinite(flat_p)
    idx = np.searchsorted(edges, flat_k[finite], side="right") - 1
    idx[flat_k[finite] == edges[-1]] = len(edges) - 2
    keep = (idx >= 0) & (idx < len(edges) - 1)
    idx, values, kvals = idx[keep], flat_p[finite][keep], flat_k[finite][keep]
    nb = len(edges) - 1
    counts = np.bincount(idx, minlength=nb).astype(np.int64)
    psum = np.bincount(idx, weights=values, minlength=nb)
    ksum = np.bincount(idx, weights=kvals, minlength=nb)
    pmean = np.divide(psum, counts, out=np.zeros(nb), where=counts > 0)
    kmean = np.divide(ksum, counts, out=np.sqrt(edges[:-1] * edges[1:]), where=counts > 0)
    # The DC mode should not contaminate a positive-k bin if user supplied bins reaching k=0; guard it.
    kmean[counts == 0] = np.sqrt(edges[:-1][counts == 0] * edges[1:][counts == 0])
    return BinnedSpectrum(
        kmean,
        pmean,
        counts,
        edges,
        float(box_size_mpc),
        "isotropic_fft_cross" if field_b is not None else "isotropic_fft_auto",
        field_b is not None,
        tuple(metadata),
    )


def estimate_power_multipoles(
    field: np.ndarray,
    box_size_mpc: float,
    *,
    k_edges_mpc_inv: np.ndarray | None = None,
    n_k_bins: int = 20,
    multipoles: tuple[int, ...] = (0, 2, 4),
    los_axis: int = 2,
    subtract_mean: bool = True,
    mu_min: float = 0.0,
    mu_max: float = 1.0,
    metadata: Iterable[tuple[str, str]] = (),
) -> MultipoleSpectrum:
    r"""Estimate even Legendre multipoles from a periodic 3-D field in a fixed LOS frame.

    For each Fourier mode the estimator accumulates ``(2ℓ+1) P(k,μ) L_ℓ(μ)`` and divides by the
    number of modes in the k-shell. With a real field and a full Fourier lattice, this is equivalent
    to the usual average over μ in [-1,1] for even multipoles. ``mu_min``/``mu_max`` optionally limit
    the absolute cosine, useful to diagnose mode cuts; such cuts require the matching normalization.
    This implementation normalizes by the retained mode count, so tests should compare the full-
    μ case to direct theoretical multipoles rather than assume a survey estimator.
    """
    arr = _validate_field(field)
    if not 0 <= los_axis <= 2:
        raise ValueError("los_axis must be 0, 1 or 2")
    if not (0 <= mu_min < mu_max <= 1):
        raise ValueError("require 0 <= mu_min < mu_max <= 1")
    if not multipoles or any(int(l) != l or l < 0 or l % 2 for l in multipoles):
        raise ValueError("multipoles must be a non-empty tuple of non-negative even integers")
    n = arr.shape[0]
    if k_edges_mpc_inv is None:
        edges = default_log_k_edges(box_size_mpc, n, n_bins=n_k_bins, nyquist_fraction=0.8)
    else:
        edges = np.asarray(k_edges_mpc_inv, float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("k edges must be positive and increasing")
    if subtract_mean:
        arr = arr - np.mean(arr)
    ft = np.fft.fftn(arr)
    _, _, _, volume = _fourier_geometry(n, box_size_mpc)
    p_mode = np.abs(ft) ** 2 * (volume / n**6)
    k1 = 2 * math.pi * np.fft.fftfreq(n, d=box_size_mpc / n)
    grids = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(sum(g * g for g in grids))
    los_k = grids[los_axis]
    mu = np.divide(np.abs(los_k), kmag, out=np.zeros_like(kmag), where=kmag > 0)
    valid = (kmag >= edges[0]) & (kmag <= edges[-1]) & (kmag > 0)
    valid &= (mu >= mu_min) & (mu <= mu_max)
    ids = np.searchsorted(edges, kmag.ravel(), side="right") - 1
    ids[kmag.ravel() == edges[-1]] = len(edges) - 2
    valid &= (ids.reshape(kmag.shape) >= 0) & (ids.reshape(kmag.shape) < len(edges) - 1)
    flat_id = ids[valid.ravel()]
    flat_k = kmag[valid]
    flat_mu = mu[valid]
    flat_p = p_mode[valid]
    nb = len(edges) - 1
    counts = np.bincount(flat_id, minlength=nb).astype(np.int64)
    ksum = np.bincount(flat_id, weights=flat_k, minlength=nb)
    kmean = np.divide(ksum, counts, out=np.sqrt(edges[:-1] * edges[1:]), where=counts > 0)
    output: dict[int, np.ndarray] = {}
    for ell in multipoles:
        coef = np.zeros(ell + 1)
        coef[ell] = 1
        leg = legval(flat_mu, coef)
        sums = np.bincount(flat_id, weights=(2 * ell + 1) * flat_p * leg, minlength=nb)
        output[int(ell)] = np.divide(sums, counts, out=np.zeros(nb), where=counts > 0)
    return MultipoleSpectrum(
        kmean, output, counts, edges, float(box_size_mpc), los_axis, tuple(metadata)
    )


def cross_correlation_coefficient(
    p_ab: np.ndarray,
    p_aa: np.ndarray,
    p_bb: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return r_ab=P_ab/sqrt(P_aa P_bb), with invalid auto-power bins rejected."""
    ab, aa, bb = map(lambda x: np.asarray(x, float), (p_ab, p_aa, p_bb))
    if ab.shape != aa.shape or ab.shape != bb.shape or np.any(~np.isfinite(ab + aa + bb)):
        raise ValueError("spectra must be finite and have matching shapes")
    if np.any(aa < 0) or np.any(bb < 0) or np.any(aa * bb <= tolerance**2):
        raise ValueError("auto-power must be positive and non-degenerate")
    result = ab / np.sqrt(aa * bb)
    # Values outside [-1,1] can arise from inconsistent/noisy independent estimators; do not clip.
    return result


def shot_noise_poisson(number_density_mpc3: float) -> float:
    """Return unweighted Poisson shot noise 1/nbar in Mpc^3."""
    if not math.isfinite(number_density_mpc3) or number_density_mpc3 <= 0:
        raise ValueError("number density must be positive and finite")
    return 1.0 / number_density_mpc3


def dimensionless_bandpower(k_mpc_inv: np.ndarray, p_mpc3: np.ndarray) -> np.ndarray:
    """Compute Δ² from matching k [Mpc^-1] and P [Mpc³] arrays."""
    k, p = np.asarray(k_mpc_inv, float), np.asarray(p_mpc3, float)
    if k.shape != p.shape or np.any(k <= 0) or np.any(p < 0) or np.any(~np.isfinite(k + p)):
        raise ValueError("k and non-negative P must be finite, positive/matching arrays")
    return k**3 * p / (2 * math.pi**2)


def gaussian_auto_power_error(
    power_mpc3: np.ndarray, mode_counts: np.ndarray, *, shot_noise_mpc3: float | np.ndarray = 0.0
) -> np.ndarray:
    """Return Gaussian standard error sqrt(2)(P+N)/sqrt(N_modes)."""
    p, n, noise = (
        np.asarray(power_mpc3, float),
        np.asarray(mode_counts, float),
        np.asarray(shot_noise_mpc3, float),
    )
    if p.shape != n.shape or np.any(p < 0) or np.any(n <= 0) or np.any(noise < 0):
        raise ValueError("power/noise must be non-negative and mode_counts positive/matching")
    return np.sqrt(2.0) * (p + noise) / np.sqrt(n)


def gaussian_cross_power_error(
    p_aa: np.ndarray,
    p_bb: np.ndarray,
    p_ab: np.ndarray,
    mode_counts: np.ndarray,
    *,
    noise_a_mpc3: float | np.ndarray = 0.0,
    noise_b_mpc3: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Gaussian standard error for P_ab, including auto-spectra and noise terms."""
    aa, bb, ab, n = map(lambda v: np.asarray(v, float), (p_aa, p_bb, p_ab, mode_counts))
    na, nb = np.asarray(noise_a_mpc3, float), np.asarray(noise_b_mpc3, float)
    if aa.shape != bb.shape or aa.shape != ab.shape or aa.shape != n.shape:
        raise ValueError("spectra and mode-count arrays must have matching shapes")
    variance = (ab**2 + (aa + na) * (bb + nb)) / n
    if np.any(variance < 0) or np.any(n <= 0):
        raise ValueError("variance must be non-negative and mode count positive")
    return np.sqrt(variance)


__all__ = [
    "BinnedSpectrum",
    "MultipoleSpectrum",
    "default_log_k_edges",
    "estimate_isotropic_power",
    "estimate_power_multipoles",
    "cross_correlation_coefficient",
    "shot_noise_poisson",
    "dimensionless_bandpower",
    "gaussian_auto_power_error",
    "gaussian_cross_power_error",
]
