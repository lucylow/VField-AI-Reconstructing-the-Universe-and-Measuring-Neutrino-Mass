import numpy as np
import pytest

from nufield.cosmology.physics.analytic_models import eds_comoving_distance_mpc, eds_E
from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.distances import (
    angular_diameter_distance_mpc,
    comoving_volume_shell_mpc3,
    luminosity_distance_mpc,
    radial_comoving_distance_mpc,
    redshift_at_comoving_distance,
    transverse_comoving_distance_mpc,
    volume_element_mpc3_sr_dz,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology


def test_background_closes_at_z0_and_density_fractions_sum_to_one():
    bg = BackgroundCosmology(PhysicalCosmology())
    assert float(bg.E(0)) == pytest.approx(1.0, abs=1e-8)
    assert sum(bg.density_fractions(0).values()) == pytest.approx(1.0, abs=1e-8)
    assert bg.omega_de0 > 0


def test_background_vectorization_matches_scalar():
    bg = BackgroundCosmology(PhysicalCosmology())
    zs = np.array([0.0, 0.5, 1.0, 2.0])
    vector = bg.H(zs)
    scalar = np.array([bg.H(float(z)) for z in zs])
    assert np.allclose(vector, scalar, rtol=1e-10)
    assert np.all(np.diff(vector) > 0)


def test_distance_relations_and_monotonicity():
    bg = BackgroundCosmology(PhysicalCosmology())
    dc = radial_comoving_distance_mpc(bg, 1.0)
    dm = transverse_comoving_distance_mpc(bg, 1.0)
    da = angular_diameter_distance_mpc(bg, 1.0)
    dl = luminosity_distance_mpc(bg, 1.0)
    assert dc == pytest.approx(dm, rel=1e-7)  # default is flat
    assert da == pytest.approx(dm / 2)
    assert dl == pytest.approx(2 * dm)
    assert radial_comoving_distance_mpc(bg, 2.0) > dc
    assert volume_element_mpc3_sr_dz(bg, 1.0) > 0
    assert comoving_volume_shell_mpc3(bg, 0.2, 1.0, 0.1) > 0


def test_distance_inverse_recovers_redshift():
    bg = BackgroundCosmology(PhysicalCosmology())
    distance = radial_comoving_distance_mpc(bg, 1.7)
    assert redshift_at_comoving_distance(bg, distance, z_bracket=(0, 3)) == pytest.approx(
        1.7, abs=1e-6
    )


def test_curvature_transverse_distance_has_expected_sign():
    open_bg = BackgroundCosmology(PhysicalCosmology(omega_k=0.02))
    flat_bg = BackgroundCosmology(PhysicalCosmology())
    closed_bg = BackgroundCosmology(PhysicalCosmology(omega_k=-0.02))
    z = 1.0
    assert transverse_comoving_distance_mpc(open_bg, z) > radial_comoving_distance_mpc(open_bg, z)
    assert transverse_comoving_distance_mpc(closed_bg, z) < radial_comoving_distance_mpc(
        closed_bg, z
    )
    assert transverse_comoving_distance_mpc(flat_bg, z) == pytest.approx(
        radial_comoving_distance_mpc(flat_bg, z)
    )


def test_analytic_eds_expansion_and_distance_are_finite_and_monotonic():
    z = np.array([0.0, 0.5, 1.0, 3.0])
    assert np.all(np.diff(eds_E(z)) > 0)
    d = eds_comoving_distance_mpc(z)
    assert d[0] == pytest.approx(0.0)
    assert np.all(np.diff(d) > 0)


def test_redshift_drift_has_expected_zero_and_signs():
    from nufield.cosmology.physics.redshift_drift import redshift_drift_velocity_cm_s_per_year

    background = BackgroundCosmology(PhysicalCosmology())
    assert redshift_drift_velocity_cm_s_per_year(background, 0.0) == pytest.approx(0.0, abs=1e-14)
    assert redshift_drift_velocity_cm_s_per_year(background, 1.0) > 0
    assert redshift_drift_velocity_cm_s_per_year(background, 5.0) < 0


def test_redshift_drift_rejects_invalid_redshift():
    from nufield.cosmology.physics.redshift_drift import redshift_drift_velocity_cm_s_per_year

    background = BackgroundCosmology(PhysicalCosmology())
    with pytest.raises(ValueError, match="finite and non-negative"):
        redshift_drift_velocity_cm_s_per_year(background, -0.1)
