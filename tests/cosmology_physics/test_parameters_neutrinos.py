import numpy as np
import pytest

from nufield.cosmology.physics.constants import NEUTRINO_TO_PHOTON_REL
from nufield.cosmology.physics.neutrinos import (
    neutrino_energy_density_eV_cm3,
    neutrino_temperature_eV,
    omega_nu_massive,
    omega_nu_nonrelativistic_approx,
    solve_neutrino_masses,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology


def test_physical_parameter_schema_and_units():
    p = PhysicalCosmology()
    assert p.h == pytest.approx(0.674)
    assert p.omega_b_h2 == pytest.approx(p.omega_b * p.h**2)
    assert sum(p.neutrino_masses_ev) == pytest.approx(p.sum_mnu_ev)
    assert p.to_dict()["density_convention"].startswith("Omega_i")


def test_parameter_rejects_invalid_values():
    with pytest.raises(ValueError):
        PhysicalCosmology(H0_km_s_Mpc=-1)
    with pytest.raises(ValueError):
        PhysicalCosmology(sum_mnu_ev=-0.1)
    with pytest.raises(ValueError):
        PhysicalCosmology(N_eff=1.0, n_massive_species=3)


def test_massless_neutrino_limit_recovers_relativistic_density_ratio():
    # rho_nu/rho_gamma must approach (7/8)(Tnu/Tgamma)^4 for one nu+anti-nu species.
    tnu = float(neutrino_temperature_eV(0.0))
    rho_nu = float(neutrino_energy_density_eV_cm3(0.0))
    # Photon blackbody energy density in matching natural-unit conversion.
    from nufield.cosmology.physics.constants import HBAR_C_EV_CM, K_B_EV_K, TCMB0_K

    tgamma = K_B_EV_K * TCMB0_K
    rho_gamma = np.pi**2 / 15 * tgamma**4 / HBAR_C_EV_CM**3
    assert rho_nu / rho_gamma == pytest.approx(NEUTRINO_TO_PHOTON_REL, rel=2e-5)
    assert tnu > 0


def test_neutrino_temperature_redshifts_linearly():
    t0, t4 = neutrino_temperature_eV(0), neutrino_temperature_eV(4)
    assert float(t4 / t0) == pytest.approx(5.0)


def test_mass_hierarchy_sums_and_minimums():
    normal = solve_neutrino_masses(0.06, "normal")
    assert sum(normal) == pytest.approx(0.06, abs=1e-10)
    assert normal[2] > normal[1] > normal[0]
    with pytest.raises(ValueError, match="below"):
        solve_neutrino_masses(0.05, "normal")
    inverted = solve_neutrino_masses(0.11, "inverted")
    assert sum(inverted) == pytest.approx(0.11, abs=1e-10)
    assert inverted[0] > inverted[2]


def test_omega_nu_fd_matches_late_time_standard_approximation_reasonably():
    p = PhysicalCosmology(sum_mnu_ev=0.12)
    exact_background = omega_nu_massive(p.neutrino_masses_ev, h=p.h, T_cmb_K=p.T_cmb_K)
    late_time_approx = omega_nu_nonrelativistic_approx(p.sum_mnu_ev, p.h)
    assert exact_background == pytest.approx(late_time_approx, rel=0.02)


def test_neutrino_density_rejects_negative_mass_or_redshift():
    with pytest.raises(ValueError):
        neutrino_energy_density_eV_cm3(-0.1)
    with pytest.raises(ValueError):
        neutrino_temperature_eV(-1)
