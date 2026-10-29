"""Transparent diagnostics for neutrino fraction, instantaneous free-streaming scales and P(k) response.

The formulas here are diagnostic approximations. In particular, the familiar small-scale suppression
rule of thumb is not a likelihood model and does not replace a Boltzmann solver with massive-neutrino
perturbations. The principal result should be empirical response measurements from matched CAMB/CLASS
or simulation spectra; the approximation is only provided to sanity-check magnitudes and trends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.neutrinos import thermal_velocity_km_s
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class NeutrinoSuppressionReport:
    sum_mnu_ev: float
    omega_nu: float
    omega_m: float
    f_nu: float
    high_k_rule_of_thumb_fractional_suppression: float
    z: float
    instantaneous_k_free_stream_mpc_inv: tuple[float, ...]
    masses_ev: tuple[float, ...]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "sum_mnu_ev": self.sum_mnu_ev,
            "omega_nu": self.omega_nu,
            "omega_m": self.omega_m,
            "f_nu": self.f_nu,
            "high_k_rule_of_thumb_fractional_suppression": self.high_k_rule_of_thumb_fractional_suppression,
            "redshift": self.z,
            "instantaneous_k_free_stream_mpc_inv": list(self.instantaneous_k_free_stream_mpc_inv),
            "masses_ev": list(self.masses_ev),
            "caveats": list(self.caveats),
        }


def neutrino_fraction(params: PhysicalCosmology) -> tuple[float, float, float]:
    """Return (Omega_nu, Omega_m including Omega_nu, f_nu) using the thermal-background density."""
    from nufield.cosmology.physics.neutrinos import omega_nu_massive

    onu = omega_nu_massive(params.neutrino_masses_ev, h=params.h, T_cmb_K=params.T_cmb_K)
    om = params.omega_b + params.omega_cdm + onu
    if om <= 0:
        raise ValueError("total matter density must be positive")
    return float(onu), float(om), float(onu / om)


def instantaneous_free_streaming_k(
    mass_ev: float,
    redshift: float,
    H_km_s_Mpc: float,
    *,
    T_cmb_K: float = 2.7255,
) -> float:
    r"""Estimate k_fs≈sqrt(3/2) aH/v_th in Mpc^-1 for one species.

    This is an instantaneous Jeans-like free-streaming scale using a characteristic thermal speed.
    It is not the integrated free-streaming horizon and must not be confused with a calibrated transfer
    function transition scale.
    """
    if (
        mass_ev <= 0
        or redshift < 0
        or H_km_s_Mpc <= 0
        or not np.isfinite(mass_ev + redshift + H_km_s_Mpc)
    ):
        raise ValueError("mass positive, redshift non-negative and H positive are required")
    a = 1 / (1 + redshift)
    v = thermal_velocity_km_s(mass_ev, redshift, T_cmb_K)
    return float(np.sqrt(1.5) * a * H_km_s_Mpc / v)


def neutrino_suppression_report(
    params: PhysicalCosmology,
    redshift: float,
    H_km_s_Mpc: float,
) -> NeutrinoSuppressionReport:
    """Assemble fraction, heuristic suppression and per-eigenmass free-streaming diagnostics."""
    if redshift < 0 or not np.isfinite(redshift):
        raise ValueError("redshift must be finite and non-negative")
    onu, om, fnu = neutrino_fraction(params)
    masses = tuple(float(x) for x in params.neutrino_masses_ev)
    kfs = tuple(
        instantaneous_free_streaming_k(m, redshift, H_km_s_Mpc, T_cmb_K=params.T_cmb_K)
        for m in masses
        if m > 0
    )
    # The -8 f_nu expression is only a classic order-of-magnitude heuristic for small f_nu and
    # sufficiently small scales, not a general prediction across k, z, or neutrino hierarchies.
    heuristic = -8.0 * fnu
    return NeutrinoSuppressionReport(
        params.sum_mnu_ev,
        onu,
        om,
        fnu,
        heuristic,
        float(redshift),
        kfs,
        masses,
        (
            "The Ων estimate uses the homogeneous thermal background, not the exact solver's density parameter mapping.",
            "The ∆P/P≈−8 fν heuristic is not valid as a general likelihood prediction.",
            "Instantaneous k_fs is a scale diagnostic, not the integrated free-streaming horizon.",
            "Use matched total or cb spectra from a massive-neutrino Boltzmann solver for quantitative response.",
        ),
    )


def empirical_neutrino_power_response(
    massive: MatterPowerSpectrum,
    reference: MatterPowerSpectrum,
    *,
    k_min_mpc_inv: float | None = None,
    k_max_mpc_inv: float | None = None,
    n_points: int = 300,
    same_component_verified: bool = False,
) -> dict[str, object]:
    """Calculate empirical ΔP/P and band summaries from matched spectra on a common k grid.

    The function cannot infer whether the spectra share component definition, redshift, primordial
    amplitude, or nuisance assumptions. ``same_component_verified`` is a deliberate caller attestation;
    absent that, output is returned but explicitly marked ineligible for interpretation as neutrino-only
    suppression.
    """
    if not np.isclose(massive.redshift, reference.redshift, atol=1e-8, rtol=0):
        raise ValueError("spectra must have identical redshift")
    lo = max(
        float(massive.k_mpc_inv[0]),
        float(reference.k_mpc_inv[0]),
        0 if k_min_mpc_inv is None else float(k_min_mpc_inv),
    )
    hi = min(
        float(massive.k_mpc_inv[-1]),
        float(reference.k_mpc_inv[-1]),
        np.inf if k_max_mpc_inv is None else float(k_max_mpc_inv),
    )
    if not 0 < lo < hi or n_points < 8:
        raise ValueError("invalid common k support or n_points")
    k = np.geomspace(lo, hi, n_points)
    p1, p0 = massive.evaluate(k), reference.evaluate(k)
    response = p1 / p0 - 1
    return {
        "k_mpc_inv": k.tolist(),
        "fractional_response": response.tolist(),
        "median_response": float(np.median(response)),
        "minimum_response": float(np.min(response)),
        "maximum_response": float(np.max(response)),
        "same_component_verified": bool(same_component_verified),
        "interpretation_eligible": bool(same_component_verified),
        "redshift": massive.redshift,
        "massive_source": massive.source,
        "reference_source": reference.source,
        "warning": "A ratio is not neutrino-only suppression unless all other cosmological parameters, component definitions and conventions are matched.",
    }


__all__ = [
    "NeutrinoSuppressionReport",
    "neutrino_fraction",
    "instantaneous_free_streaming_k",
    "neutrino_suppression_report",
    "empirical_neutrino_power_response",
]
