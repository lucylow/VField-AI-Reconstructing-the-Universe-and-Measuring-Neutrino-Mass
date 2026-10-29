"""Physical constants and unit conversions used by the physics-grade layer.

Values are pinned in source to make results reproducible. These constants are not intended to
replace a modern constants package when an analysis requires a specified CODATA release.
"""

from __future__ import annotations

C_KM_S = 299_792.458
C_M_S = 299_792_458.0
G_SI = 6.67430e-11
K_B_J_K = 1.380649e-23
K_B_EV_K = 8.617333262145e-5
HBAR_C_EV_CM = 1.973269804e-5
MPC_M = 3.085677581491367e22
MPC_CM = MPC_M * 100.0
JOULE_PER_EV = 1.602176634e-19
KG_PER_EV_C2 = JOULE_PER_EV / C_M_S**2
TCMB0_K = 2.7255
TNU_TO_TGAMMA = (4.0 / 11.0) ** (1.0 / 3.0)
N_EFF_STANDARD = 3.046
# rho_crit = 1.05375e4 h^2 eV/cm^3, commonly used for cosmology conversions.
RHO_CRIT_H2_EV_CM3 = 1.05375e4
# Omega_gamma h^2 for a blackbody at T_CMB=2.7255 K, rounded conventionally.
OMEGA_GAMMA_H2_T0 = 2.472975e-5
# Relativistic neutrino/gamma density ratio for one effective species.
NEUTRINO_TO_PHOTON_REL = (7.0 / 8.0) * TNU_TO_TGAMMA**4
MASS_DENSITY_CONVERSION_MSUN_MPC3 = 2.77536627e11  # rho_crit,0 / h^2 in Msun Mpc^-3

__all__ = [
    "C_KM_S",
    "C_M_S",
    "G_SI",
    "K_B_J_K",
    "K_B_EV_K",
    "HBAR_C_EV_CM",
    "MPC_M",
    "MPC_CM",
    "JOULE_PER_EV",
    "KG_PER_EV_C2",
    "TCMB0_K",
    "TNU_TO_TGAMMA",
    "N_EFF_STANDARD",
    "RHO_CRIT_H2_EV_CM3",
    "OMEGA_GAMMA_H2_T0",
    "NEUTRINO_TO_PHOTON_REL",
    "MASS_DENSITY_CONVERSION_MSUN_MPC3",
]
