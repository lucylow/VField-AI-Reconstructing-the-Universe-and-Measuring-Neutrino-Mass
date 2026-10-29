"""Monte-Carlo Fourier-triangle bispectrum estimators for periodic simulation boxes.

The bispectrum measures phase-coupled three-point structure not captured by the power spectrum alone.
This module implements a transparent random-triangle estimator for regular periodic meshes. It is
intended for validation and method development rather than high-throughput production measurements;
large survey analyses should use a validated, optimized estimator with the correct mask/window terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TriangleEstimate:
    k1_range_mpc_inv: tuple[float, float]
    k2_range_mpc_inv: tuple[float, float]
    k3_range_mpc_inv: tuple[float, float]
    bispectrum_mpc6: float
    standard_error_mpc6: float
    requested_samples: int
    accepted_samples: int
    attempted_samples: int
    acceptance_rate: float
    box_size_mpc: float
    seed: int | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "k1_range_mpc_inv": list(self.k1_range_mpc_inv),
            "k2_range_mpc_inv": list(self.k2_range_mpc_inv),
            "k3_range_mpc_inv": list(self.k3_range_mpc_inv),
            "bispectrum_mpc6": self.bispectrum_mpc6,
            "standard_error_mpc6": self.standard_error_mpc6,
            "requested_samples": self.requested_samples,
            "accepted_samples": self.accepted_samples,
            "attempted_samples": self.attempted_samples,
            "acceptance_rate": self.acceptance_rate,
            "box_size_mpc": self.box_size_mpc,
            "seed": self.seed,
            "notes": list(self.notes),
            "units": {"B": "Mpc^6", "k": "Mpc^-1"},
        }


def _field_and_modes(field: np.ndarray, box_size_mpc: float):
    x = np.asarray(field, float)
    if x.ndim != 3 or len(set(x.shape)) != 1 or x.shape[0] < 4:
        raise ValueError("field must be a cubic 3-D array with side length >=4")
    if np.any(~np.isfinite(x)) or not math.isfinite(box_size_mpc) or box_size_mpc <= 0:
        raise ValueError("field must be finite and box size positive")
    n = x.shape[0]
    x = x - np.mean(x)
    volume = box_size_mpc**3
    delta_k = np.fft.fftn(x) * (volume / n**3)
    mode_numbers = np.fft.fftfreq(n) * n
    fund = 2 * math.pi / box_size_mpc
    k1 = mode_numbers * fund
    grids = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(sum(g * g for g in grids))
    # Nyquist integer representation is valid for the discrete periodic lattice.
    return x, delta_k, kmag, volume, n


def _validate_range(label: str, interval: tuple[float, float]) -> tuple[float, float]:
    if len(interval) != 2:
        raise ValueError(f"{label} must contain (k_min,k_max)")
    lo, hi = map(float, interval)
    if not math.isfinite(lo + hi) or lo <= 0 or hi <= lo:
        raise ValueError(f"{label} requires finite 0<k_min<k_max")
    return lo, hi


def estimate_bispectrum_triangle(
    field: np.ndarray,
    box_size_mpc: float,
    k1_range_mpc_inv: tuple[float, float],
    k2_range_mpc_inv: tuple[float, float],
    k3_range_mpc_inv: tuple[float, float],
    *,
    n_samples: int = 20_000,
    seed: int | None = 0,
    max_attempt_factor: int = 500,
    chunk_size: int = 50_000,
) -> TriangleEstimate:
    r"""Estimate B(k1,k2,k3)=Re[δ(k1)δ(k2)δ(k3)]/V from closed Fourier triangles.

    Wavevectors satisfy exact periodic-lattice closure k1+k2+k3=0 modulo the grid lattice. k1 and k2
    are drawn uniformly from their radial shells; accepted pairs whose closing k3 falls in the third
    shell therefore form a Monte-Carlo sample of valid ordered triangles. Standard error is the sample
    standard deviation of the real triple products divided by sqrt(Naccepted); it omits finite-volume
    dependence between repeated/overlapping triangles. Increase samples and vary seeds for convergence.
    """
    r1 = _validate_range("k1_range_mpc_inv", k1_range_mpc_inv)
    r2 = _validate_range("k2_range_mpc_inv", k2_range_mpc_inv)
    r3 = _validate_range("k3_range_mpc_inv", k3_range_mpc_inv)
    if n_samples < 2 or max_attempt_factor < 1 or chunk_size < 1:
        raise ValueError("n_samples>=2, max_attempt_factor>=1 and chunk_size>=1 required")
    _, fk, kmag, volume, n = _field_and_modes(field, box_size_mpc)
    shell1 = np.flatnonzero((kmag >= r1[0]) & (kmag < r1[1]))
    shell2 = np.flatnonzero((kmag >= r2[0]) & (kmag < r2[1]))
    if shell1.size == 0 or shell2.size == 0:
        raise ValueError("one of the first two k shells has no Fourier modes")
    # Full signed integer Fourier mode numbers represented by actual array indices.
    coord = np.fft.fftfreq(n) * n
    integer_coords = np.rint(coord).astype(int)
    ix, iy, iz = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    index_triplets = np.column_stack((ix.ravel(), iy.ravel(), iz.ravel()))
    signed_grid_modes = integer_coords[index_triplets]
    flat_k = kmag.ravel()
    flat_fk = fk.ravel()
    rng = np.random.default_rng(seed)
    products: list[np.ndarray] = []
    accepted = 0
    attempted = 0
    max_attempts = int(n_samples * max_attempt_factor)
    while accepted < n_samples and attempted < max_attempts:
        batch = min(chunk_size, max_attempts - attempted, max(1024, (n_samples - accepted) * 20))
        a_indices = rng.choice(shell1, size=batch, replace=True)
        b_indices = rng.choice(shell2, size=batch, replace=True)
        attempted += batch
        k3_integer = -(signed_grid_modes[a_indices] + signed_grid_modes[b_indices])
        k3_indices = np.mod(k3_integer, n)
        c_indices = (k3_indices[:, 0] * n + k3_indices[:, 1]) * n + k3_indices[:, 2]
        valid = (flat_k[c_indices] >= r3[0]) & (flat_k[c_indices] < r3[1])
        good_a, good_b, good_c = a_indices[valid], b_indices[valid], c_indices[valid]
        take = min(good_a.size, n_samples - accepted)
        if take:
            tri = flat_fk[good_a[:take]] * flat_fk[good_b[:take]] * flat_fk[good_c[:take]] / volume
            products.append(np.real(tri))
            accepted += take
    if accepted < 2:
        raise ValueError(
            f"only {accepted} valid triangle samples found after {attempted} attempts; widen shells"
        )
    vals = np.concatenate(products)
    estimate = float(np.mean(vals))
    stderr = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    return TriangleEstimate(
        r1,
        r2,
        r3,
        estimate,
        stderr,
        int(n_samples),
        int(vals.size),
        attempted,
        float(vals.size / attempted),
        float(box_size_mpc),
        seed,
        (
            "Periodic-box estimator only; not corrected for survey masks or window coupling.",
            "Monte-Carlo standard error does not account for covariance among repeated triangles.",
            "Bispectrum sign and magnitude depend on field convention and triangle bins.",
        ),
    )


def reduced_bispectrum(
    bispectrum_mpc6: float | np.ndarray,
    p1_mpc3: float | np.ndarray,
    p2_mpc3: float | np.ndarray,
    p3_mpc3: float | np.ndarray,
) -> np.ndarray:
    r"""Compute reduced bispectrum Q=B/(P1P2+P2P3+P3P1)."""
    b, p1, p2, p3 = map(
        lambda v: np.asarray(v, float), (bispectrum_mpc6, p1_mpc3, p2_mpc3, p3_mpc3)
    )
    if np.any(~np.isfinite(b + p1 + p2 + p3)) or np.any(p1 < 0) or np.any(p2 < 0) or np.any(p3 < 0):
        raise ValueError("inputs must be finite and powers non-negative")
    denom = p1 * p2 + p2 * p3 + p3 * p1
    if np.any(denom <= 0):
        raise ValueError("reduced bispectrum denominator must be positive")
    return b / denom


def perturbative_tree_level_bispectrum(
    k1_mpc_inv: np.ndarray,
    k2_mpc_inv: np.ndarray,
    mu12: np.ndarray,
    p1_mpc3: np.ndarray,
    p2_mpc3: np.ndarray,
    p3_mpc3: np.ndarray,
) -> np.ndarray:
    r"""Tree-level matter bispectrum B=2F2(k1,k2)P1P2+cyc for an Einstein-de Sitter F2 kernel.

    This is a useful baseline on sufficiently large, weakly nonlinear scales. It is not a precision
    massive-neutrino prediction: scale-dependent growth and neutrino perturbations are not included.
    Arrays should be broadcast-compatible and represent closed triangle sides, with k3 opposite k1+k2.
    """
    k1, k2, mu, p1, p2, p3 = map(
        lambda v: np.asarray(v, float), (k1_mpc_inv, k2_mpc_inv, mu12, p1_mpc3, p2_mpc3, p3_mpc3)
    )
    if np.any(k1 <= 0) or np.any(k2 <= 0) or np.any(np.abs(mu) > 1):
        raise ValueError("k sides must be positive and cosine within [-1,1]")
    if np.any(p1 < 0) or np.any(p2 < 0) or np.any(p3 < 0):
        raise ValueError("power values must be non-negative")
    r = k1 / k2
    f2_12 = 5 / 7 + 0.5 * mu * (r + 1 / r) + 2 / 7 * mu**2
    # Cyclic kernels use the cosine and ratio for each pair implied by the triangle law.
    k3_sq = k1 * k1 + k2 * k2 + 2 * k1 * k2 * mu
    k3 = np.sqrt(np.maximum(k3_sq, 1e-30))
    mu23 = np.clip((-k2 - k1 * mu) / k3, -1, 1)
    mu31 = np.clip((-k1 - k2 * mu) / k3, -1, 1)
    f2_23 = 5 / 7 + 0.5 * mu23 * (k2 / k3 + k3 / k2) + 2 / 7 * mu23**2
    f2_31 = 5 / 7 + 0.5 * mu31 * (k3 / k1 + k1 / k3) + 2 / 7 * mu31**2
    return 2 * (f2_12 * p1 * p2 + f2_23 * p2 * p3 + f2_31 * p3 * p1)


__all__ = [
    "TriangleEstimate",
    "estimate_bispectrum_triangle",
    "reduced_bispectrum",
    "perturbative_tree_level_bispectrum",
]
