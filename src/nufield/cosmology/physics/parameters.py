"""Explicit physical cosmological parameterization for research-grade calculations.

The legacy :mod:`nufield.parameters` remains the toy/demo schema. This schema separates physical
baryon/CDM densities, primordial amplitude, curvature, dark-energy parameters and neutrino model.
All density parameters ``omega_*`` in this module use the uppercase-Omega convention (fractions of
critical density today); ``omega_b_h2`` and ``omega_cdm_h2`` are exposed properties only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

NeutrinoHierarchy = Literal["degenerate", "normal", "inverted"]


@dataclass(frozen=True, slots=True)
class PhysicalCosmology:
    """Flat or curved w0-wa cosmology with a configurable massive-neutrino prescription.

    Defaults are illustrative near-standard values, not a fit to any dataset. For hierarchical
    neutrino masses, sums below the oscillation-implied minimum are rejected by ``neutrino_masses``.
    The default degenerate model is an approximation, even when the total mass is physically
    motivated; report the hierarchy and Boltzmann-code settings alongside every result.
    """

    H0_km_s_Mpc: float = 67.4
    omega_b: float = 0.0493
    omega_cdm: float = 0.2640
    omega_k: float = 0.0
    sum_mnu_ev: float = 0.06
    neutrino_hierarchy: NeutrinoHierarchy = "degenerate"
    n_massive_species: int = 3
    N_eff: float = 3.046
    T_cmb_K: float = 2.7255
    A_s: float = 2.1e-9
    n_s: float = 0.965
    alpha_s: float = 0.0
    k_pivot_Mpc_inv: float = 0.05
    tau_reio: float = 0.054
    w0: float = -1.0
    wa: float = 0.0

    def __post_init__(self) -> None:
        finite_positive = {
            "H0_km_s_Mpc": self.H0_km_s_Mpc,
            "omega_b": self.omega_b,
            "omega_cdm": self.omega_cdm,
            "T_cmb_K": self.T_cmb_K,
            "A_s": self.A_s,
            "k_pivot_Mpc_inv": self.k_pivot_Mpc_inv,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}")
        finite = {
            "omega_k": self.omega_k,
            "sum_mnu_ev": self.sum_mnu_ev,
            "N_eff": self.N_eff,
            "n_s": self.n_s,
            "alpha_s": self.alpha_s,
            "tau_reio": self.tau_reio,
            "w0": self.w0,
            "wa": self.wa,
        }
        for name, value in finite.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value!r}")
        if self.sum_mnu_ev < 0:
            raise ValueError("sum_mnu_ev must be non-negative")
        if self.N_eff < 0 or self.n_massive_species < 1 or self.n_massive_species > 3:
            raise ValueError("N_eff must be non-negative and n_massive_species must be 1, 2 or 3")
        if self.n_massive_species > self.N_eff + 1e-8:
            raise ValueError(
                "n_massive_species cannot exceed N_eff in this effective-species model"
            )
        if self.neutrino_hierarchy not in ("degenerate", "normal", "inverted"):
            raise ValueError("neutrino_hierarchy must be 'degenerate', 'normal', or 'inverted'")
        if self.neutrino_hierarchy != "degenerate" and self.n_massive_species != 3:
            raise ValueError("normal/inverted hierarchies require n_massive_species=3")
        if not 0 <= self.tau_reio < 1:
            raise ValueError("tau_reio must be in [0, 1)")
        if self.omega_b + self.omega_cdm >= 1.5:
            raise ValueError("omega_b + omega_cdm is outside a reasonable physical range")

    @property
    def h(self) -> float:
        """Dimensionless Hubble parameter, ``H0 / (100 km s^-1 Mpc^-1)``."""
        return self.H0_km_s_Mpc / 100.0

    @property
    def omega_b_h2(self) -> float:
        return self.omega_b * self.h**2

    @property
    def omega_cdm_h2(self) -> float:
        return self.omega_cdm * self.h**2

    @property
    def neutrino_masses_ev(self) -> tuple[float, ...]:
        """Return masses implied by the selected hierarchy and total mass."""
        from nufield.cosmology.physics.neutrinos import solve_neutrino_masses

        if self.neutrino_hierarchy == "degenerate":
            return tuple([self.sum_mnu_ev / self.n_massive_species] * self.n_massive_species)
        return solve_neutrino_masses(self.sum_mnu_ev, self.neutrino_hierarchy)

    def to_dict(self) -> dict[str, object]:
        """JSON-compatible parameter mapping, including explicit conventions."""
        out: dict[str, object] = asdict(self)
        out["h"] = self.h
        out["omega_b_h2"] = self.omega_b_h2
        out["omega_cdm_h2"] = self.omega_cdm_h2
        out["neutrino_masses_ev"] = list(self.neutrino_masses_ev)
        out["density_convention"] = "Omega_i = rho_i(z=0)/rho_critical(z=0)"
        out["length_convention"] = "Mpc unless a method explicitly says Mpc/h"
        return out

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> PhysicalCosmology:
        allowed = set(cls.__dataclass_fields__)
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"Unknown physical cosmology parameter(s): {sorted(extra)}")
        return cls(**values)  # type: ignore[arg-type]


__all__ = ["PhysicalCosmology", "NeutrinoHierarchy"]
