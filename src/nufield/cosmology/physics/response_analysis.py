"""Quantify neutrino-mass response, mode information and multi-probe complementarity.

The diagnostics in this module are intentionally tied to a supplied spectrum-response grid and explicit
covariance. They do not infer a neutrino-mass constraint by themselves. Power-mode Fisher estimates
assume Gaussian independent Fourier modes and should be used as a baseline; nonlinear mode coupling,
survey windows, finite-mock covariance, nuisance parameters and field-level likelihood information can
change the final information content.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from nufield.cosmology.physics.neutrino_response_grid import NeutrinoResponseGrid


@dataclass(frozen=True, slots=True)
class NeutrinoSensitivitySummary:
    redshift: float
    reference_mass_ev: float
    k_mpc_inv: np.ndarray
    dlnp_dsum_mnu_per_ev: np.ndarray
    suppression_at_reference: np.ndarray
    peak_abs_response_k_mpc_inv: float
    maximum_abs_response_per_ev: float
    low_k_rms_response_per_ev: float
    high_k_rms_response_per_ev: float
    variation_design: str
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        k, d, s = map(
            lambda x: np.asarray(x, float),
            (self.k_mpc_inv, self.dlnp_dsum_mnu_per_ev, self.suppression_at_reference),
        )
        if k.ndim != 1 or d.shape != k.shape or s.shape != k.shape:
            raise ValueError("sensitivity arrays must be matching 1-D vectors")
        object.__setattr__(self, "k_mpc_inv", k.copy())
        object.__setattr__(self, "dlnp_dsum_mnu_per_ev", d.copy())
        object.__setattr__(self, "suppression_at_reference", s.copy())

    def to_dict(self, *, include_curves: bool = True) -> dict[str, object]:
        out = asdict(self)
        out["k_mpc_inv"] = self.k_mpc_inv.tolist() if include_curves else None
        out["dlnp_dsum_mnu_per_ev"] = self.dlnp_dsum_mnu_per_ev.tolist() if include_curves else None
        out["suppression_at_reference"] = (
            self.suppression_at_reference.tolist() if include_curves else None
        )
        out["caveats"] = list(self.caveats)
        return out


def summarize_mass_response(
    grid: NeutrinoResponseGrid,
    *,
    redshift: float,
    reference_mass_ev: float,
    split_k_mpc_inv: float = 0.1,
) -> NeutrinoSensitivitySummary:
    """Summarize ∂lnP/∂Σmν and suppression relative to a selected reference mass."""
    if reference_mass_ev < grid.masses_ev[0] or reference_mass_ev > grid.masses_ev[-1]:
        raise ValueError("reference mass outside response-grid support")
    mass_response = grid.dlnp_dmass(redshift=redshift)
    nearest = int(np.argmin(np.abs(grid.masses_ev - reference_mass_ev)))
    response = mass_response[nearest]
    reference_power = grid.power_at_mass(reference_mass_ev, redshift=redshift)
    p_all = grid.pk_mpc3_at_redshift(redshift)
    nearest_suppression = 1.0 - p_all[nearest] / reference_power
    low = response[grid.k_mpc_inv < split_k_mpc_inv]
    high = response[grid.k_mpc_inv >= split_k_mpc_inv]
    peak_index = int(np.argmax(np.abs(response)))
    return NeutrinoSensitivitySummary(
        float(redshift),
        float(reference_mass_ev),
        grid.k_mpc_inv,
        response,
        nearest_suppression,
        float(grid.k_mpc_inv[peak_index]),
        float(abs(response[peak_index])),
        float(np.sqrt(np.mean(low**2))) if low.size else float("nan"),
        float(np.sqrt(np.mean(high**2))) if high.size else float("nan"),
        grid.variation_design,
        (
            "Response is conditional on the grid's fixed-parameter variation design.",
            "Mass derivatives are finite differences across grid nodes; assess step-size stability.",
            "Suppression is not a posterior constraint and depends on the reference cosmology.",
        ),
    )


def gaussian_power_mode_fisher(
    k_mpc_inv: np.ndarray,
    pk_mpc3: np.ndarray,
    dlnp_dparameter: np.ndarray,
    mode_counts: np.ndarray,
    *,
    noise_mpc3: np.ndarray | float = 0.0,
) -> dict[str, object]:
    r"""Estimate Fisher information per k bin for one parameter from Gaussian power modes.

    Uses F_i ≈ 1/2 N_modes [P/(P+N)]² (∂lnP/∂θ)². This assumes a single auto-spectrum, Gaussian
    independent modes and additive known white noise N. It ignores the uncertainty of the covariance,
    window mixing and nuisance-parameter marginalization. Output information has units θ⁻².
    """
    k, p, d, n = map(
        lambda x: np.asarray(x, float), (k_mpc_inv, pk_mpc3, dlnp_dparameter, mode_counts)
    )
    noise = np.asarray(noise_mpc3, float)
    if k.ndim != 1 or p.shape != k.shape or d.shape != k.shape or n.shape != k.shape:
        raise ValueError("k, P, response and mode counts must have matching 1-D shapes")
    if np.any(k <= 0) or np.any(p <= 0) or np.any(n < 0) or np.any(~np.isfinite(k + p + d + n)):
        raise ValueError("k/P must be positive and all inputs finite; mode counts non-negative")
    if np.any(noise < 0) or np.any(~np.isfinite(noise)):
        raise ValueError("noise must be finite and non-negative")
    total = p + noise
    signal_fraction = p / total
    per_bin = 0.5 * n * signal_fraction**2 * d**2
    cumulative = np.cumsum(per_bin)
    fisher = float(np.sum(per_bin))
    return {
        "k_mpc_inv": k.tolist(),
        "fisher_per_bin": per_bin.tolist(),
        "cumulative_fisher": cumulative.tolist(),
        "fisher_total": fisher,
        "sigma_parameter_gaussian_mode_limit": float(1 / math.sqrt(fisher))
        if fisher > 0
        else float("inf"),
        "assumptions": [
            "Gaussian independent Fourier modes",
            "known additive noise",
            "no window coupling",
            "fixed nuisance parameters",
            "local derivative at supplied fiducial cosmology",
        ],
    }


def whitened_response_cosine(
    derivative_a: np.ndarray,
    derivative_b: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Cosine angle between two observable derivative vectors under the C⁻¹ inner product.

    Values near ±1 indicate a near-degenerate local response direction; values near zero indicate
    locally orthogonal responses under the selected data covariance.
    """
    a, b, c = (
        np.asarray(derivative_a, float),
        np.asarray(derivative_b, float),
        np.asarray(covariance, float),
    )
    if a.ndim != 1 or b.shape != a.shape or c.shape != (a.size, a.size):
        raise ValueError("derivatives and covariance dimensions differ")
    if np.any(~np.isfinite(a + b)) or not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("derivatives must be finite and covariance symmetric")
    try:
        chol = np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    wa, wb = np.linalg.solve(chol, a), np.linalg.solve(chol, b)
    denom = np.linalg.norm(wa) * np.linalg.norm(wb)
    if denom == 0:
        raise ValueError("cannot calculate response angle for a zero derivative")
    return float(np.clip(np.dot(wa, wb) / denom, -1, 1))


def combine_fisher_blocks(blocks: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, object]]:
    """Sum Fisher matrices from conditionally independent data blocks and report contributions."""
    if not blocks:
        raise ValueError("Fisher blocks cannot be empty")
    mats: dict[str, np.ndarray] = {}
    shape = None
    for name, value in blocks.items():
        f = np.asarray(value, float)
        if (
            f.ndim != 2
            or f.shape[0] != f.shape[1]
            or np.any(~np.isfinite(f))
            or not np.allclose(f, f.T, atol=1e-10)
        ):
            raise ValueError(f"Fisher block {name!r} must be finite, square and symmetric")
        if shape is None:
            shape = f.shape
        if f.shape != shape:
            raise ValueError("all Fisher blocks must have matching dimensions")
        if np.linalg.eigvalsh(f).min() < -1e-9 * max(1, float(np.max(np.abs(f)))):
            raise ValueError(f"Fisher block {name!r} is not positive semidefinite")
        mats[name] = f
    total = np.sum(np.stack(list(mats.values())), axis=0)
    total = 0.5 * (total + total.T)
    summary = {
        "block_trace": {name: float(np.trace(f)) for name, f in mats.items()},
        "total_trace": float(np.trace(total)),
        "independence_assumption": "Fisher sum assumes conditionally independent blocks; account for cross-covariance otherwise.",
        "eigenvalues": np.linalg.eigvalsh(total).tolist(),
    }
    return total, summary


__all__ = [
    "NeutrinoSensitivitySummary",
    "summarize_mass_response",
    "gaussian_power_mode_fisher",
    "whitened_response_cosine",
    "combine_fisher_blocks",
]
