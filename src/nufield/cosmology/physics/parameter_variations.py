"""Controlled cosmological parameter variations and comparison-design helpers."""

from __future__ import annotations

from dataclasses import replace

from nufield.cosmology.physics.neutrinos import omega_nu_massive
from nufield.cosmology.physics.parameters import PhysicalCosmology


def vary_mnu_fixed_cb(params: PhysicalCosmology, sum_mnu_ev: float) -> PhysicalCosmology:
    """Vary total neutrino mass while holding Ω_b and Ω_cdm fixed.

    This changes total Ωm. Use when the experiment definition fixes the cold+baryonic sector.
    """
    return replace(params, sum_mnu_ev=float(sum_mnu_ev))


def vary_mnu_fixed_omega_m(params: PhysicalCosmology, sum_mnu_ev: float) -> PhysicalCosmology:
    """Vary sum_mnu while keeping today's Ωm=Ωb+Ωcdm+Ων fixed by compensating Ωcdm.

    A reduced neutrino prescription may alter the exact background density; use the same hierarchy,
    temperature, H0 and species count across both models. If the required Ωcdm becomes non-positive,
    the variation is rejected.
    """
    target = float(sum_mnu_ev)
    if target < 0:
        raise ValueError("sum_mnu_ev must be non-negative")
    old_masses = params.neutrino_masses_ev
    old_nu = omega_nu_massive(old_masses, h=params.h, T_cmb_K=params.T_cmb_K)
    target_params = replace(params, sum_mnu_ev=target)
    new_nu = omega_nu_massive(target_params.neutrino_masses_ev, h=params.h, T_cmb_K=params.T_cmb_K)
    omega_m_target = params.omega_b + params.omega_cdm + old_nu
    cdm_new = omega_m_target - params.omega_b - new_nu
    if cdm_new <= 0:
        raise ValueError("fixed-Ωm variation requires non-positive Ωcdm")
    return replace(target_params, omega_cdm=cdm_new)


def compare_variation_assumptions(
    params: PhysicalCosmology, mass_values_ev: list[float]
) -> list[dict[str, object]]:
    """Produce explicit metadata for both common Mnu comparison designs."""
    rows: list[dict[str, object]] = []
    for mass in mass_values_ev:
        for design, factory in (
            ("fixed_omega_cb", vary_mnu_fixed_cb),
            ("fixed_omega_m", vary_mnu_fixed_omega_m),
        ):
            varied = factory(params, mass)
            rows.append(
                {
                    "design": design,
                    "sum_mnu_ev": mass,
                    "omega_b": varied.omega_b,
                    "omega_cdm": varied.omega_cdm,
                    "hierarchy": varied.neutrino_hierarchy,
                    "n_massive_species": varied.n_massive_species,
                }
            )
    return rows


__all__ = ["vary_mnu_fixed_cb", "vary_mnu_fixed_omega_m", "compare_variation_assumptions"]
