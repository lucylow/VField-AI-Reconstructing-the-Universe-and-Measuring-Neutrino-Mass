"""Massive-neutrino response diagnostics measured from a validated spectrum backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class NeutrinoResponse:
    k_mpc_inv: np.ndarray
    fractional_suppression: np.ndarray
    log_derivative_per_ev: np.ndarray | None
    low_k_mean_suppression: float
    high_k_mean_suppression: float
    notes: tuple[str, ...]


def suppression_ratio(
    reference: MatterPowerSpectrum,
    massive: MatterPowerSpectrum,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-P_massive/P_reference on a shared support interval.

    The meaning of this ratio depends on the parameter-variation design: fixed Ω_cb, fixed total Ω_m,
    or another convention. This function cannot ensure that the non-neutrino parameters are matched;
    record the comparison design and check the parameter metadata before interpreting it.
    """
    if reference.redshift != massive.redshift:
        raise ValueError("spectra must have the same redshift")
    lo = max(reference.k_mpc_inv[0], massive.k_mpc_inv[0], k_min_mpc_inv or 0.0)
    hi = min(
        reference.k_mpc_inv[-1],
        massive.k_mpc_inv[-1],
        k_max_mpc_inv if k_max_mpc_inv is not None else float("inf"),
    )
    k = reference.k_mpc_inv[(reference.k_mpc_inv >= lo) & (reference.k_mpc_inv <= hi)]
    if k.size < 3:
        raise ValueError("fewer than three common k values are available")
    p0 = reference.evaluate(k)
    pm = massive.evaluate(k)
    return k, 1.0 - pm / p0


def finite_difference_mnu_response(
    low_mass: MatterPowerSpectrum,
    high_mass: MatterPowerSpectrum,
    mass_low_ev: float,
    mass_high_ev: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Estimate ∂lnP/∂Σmν with a two-sided finite difference of backend spectra."""
    if mass_high_ev <= mass_low_ev or mass_low_ev < 0:
        raise ValueError("require mass_high_ev > mass_low_ev >= 0")
    if low_mass.redshift != high_mass.redshift:
        raise ValueError("spectra must share redshift")
    lo = max(low_mass.k_mpc_inv[0], high_mass.k_mpc_inv[0])
    hi = min(low_mass.k_mpc_inv[-1], high_mass.k_mpc_inv[-1])
    k = low_mass.k_mpc_inv[(low_mass.k_mpc_inv >= lo) & (low_mass.k_mpc_inv <= hi)]
    if k.size < 3:
        raise ValueError("not enough common k values for response estimate")
    p_lo, p_hi = low_mass.evaluate(k), high_mass.evaluate(k)
    deriv = (np.log(p_hi) - np.log(p_lo)) / (mass_high_ev - mass_low_ev)
    return k, deriv


def summarize_suppression(
    reference: MatterPowerSpectrum,
    massive: MatterPowerSpectrum,
    *,
    split_k_mpc_inv: float = 0.1,
) -> NeutrinoResponse:
    """Create a compact low/high-k suppression summary without assuming its sign."""
    k, supp = suppression_ratio(reference, massive)
    low = supp[k < split_k_mpc_inv]
    high = supp[k >= split_k_mpc_inv]
    return NeutrinoResponse(
        k,
        supp,
        None,
        float(np.mean(low)) if low.size else float("nan"),
        float(np.mean(high)) if high.size else float("nan"),
        (
            "Ratio depends on matched cosmological-parameter convention.",
            "A linear spectrum alone does not capture nonlinear neutrino response.",
        ),
    )


__all__ = [
    "NeutrinoResponse",
    "suppression_ratio",
    "finite_difference_mnu_response",
    "summarize_suppression",
]
