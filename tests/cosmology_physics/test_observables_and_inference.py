import numpy as np
import pytest

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.bao import (
    bao_dilation_parameters,
    bao_distances,
    isotropic_bao_alpha,
)
from nufield.cosmology.physics.fisher import finite_difference_jacobian, fisher_forecast
from nufield.cosmology.physics.galaxy_clustering import ap_remapped_galaxy_power, power_multipoles
from nufield.cosmology.physics.likelihoods import (
    covariance_shrinkage,
    gaussian_chi_squared,
    gaussian_log_likelihood,
)
from nufield.cosmology.physics.multi_probe import (
    ProbeBlock,
    assemble_joint_vector,
    joint_log_likelihood,
)
from nufield.cosmology.physics.observables import S8
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.redshift_space import kaiser_power_mu, kaiser_power_multipoles


def test_bao_distance_and_fiducial_alphas():
    bg = BackgroundCosmology(PhysicalCosmology())
    out = bao_distances(bg, 0.8, 147.0)
    assert out.D_M_over_rdrag > 0 and out.D_H_over_rdrag > 0 and out.D_V_over_rdrag > 0
    ap = bao_dilation_parameters(
        D_M_mpc=1000,
        H_km_s_mpc=100,
        r_drag_mpc=150,
        D_M_fid_mpc=1000,
        H_fid_km_s_mpc=100,
        r_drag_fid_mpc=150,
    )
    assert ap == (pytest.approx(1.0), pytest.approx(1.0))
    assert isotropic_bao_alpha(100, 150, 100, 150) == pytest.approx(1.0)


def test_kaiser_multipoles_match_analytic_coefficients():
    p = np.array([100.0, 200.0])
    b = 1.5
    f = 0.6
    multipoles = kaiser_power_multipoles(p, b, f)
    assert multipoles[0] == pytest.approx((b * b + 2 * b * f / 3 + f * f / 5) * p)
    assert multipoles[2] == pytest.approx((4 * b * f / 3 + 4 * f * f / 7) * p)
    assert multipoles[4] == pytest.approx((8 * f * f / 35) * p)
    assert kaiser_power_mu(p, 0, b, f) == pytest.approx(b * b * p)


def test_ap_power_and_multipoles_are_finite():
    k = np.geomspace(0.01, 0.2, 40)

    def pk(kval):
        return 1000 / (1 + kval) ** 2

    p = ap_remapped_galaxy_power(
        k, 0.5, pk, bias=1.3, growth_rate=0.7, alpha_perp=1.02, alpha_parallel=0.98, sigma_fog_mpc=2
    )
    assert p.shape == k.shape and np.all(p > 0)
    out = power_multipoles(k, lambda kval, mu: (1 + 0.5 * mu**2) ** 2 * pk(kval))
    assert set(out) == {0, 2, 4}
    assert all(v.shape == k.shape for v in out.values())


def test_s8_and_gaussian_likelihoods():
    assert float(S8(0.8, 0.3)) == pytest.approx(0.8)
    y = np.array([1.0, 2.0])
    mu = np.array([1.0, 1.0])
    cov = np.eye(2)
    assert gaussian_chi_squared(y, mu, cov) == pytest.approx(1.0)
    assert np.isfinite(gaussian_log_likelihood(y, mu, cov))
    reg = covariance_shrinkage(np.array([[1, 0.4], [0.4, 1.0]]), 0.5)
    assert np.linalg.eigvalsh(reg).min() > 0


def test_fisher_recovers_linear_model_derivatives():
    def model(theta):
        return np.array([2 * theta[0] + theta[1], theta[0] - 3 * theta[1]])

    theta = np.array([0.3, -0.2])
    step = np.array([1e-5, 1e-5])
    jac = finite_difference_jacobian(model, theta, step)
    assert jac == pytest.approx(np.array([[2, 1], [1, -3]]), abs=1e-8)
    result = fisher_forecast(("a", "b"), theta, step, model, np.eye(2))
    assert result.rank == 2
    assert np.all(result.marginalized_sigma > 0)


def test_joint_probe_assembly_and_cross_covariance():
    a = ProbeBlock("galaxy", np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.eye(2), "dimensionless")
    b = ProbeBlock("lensing", np.array([0.3]), np.array([0.2]), np.array([[0.5]]), "dimensionless")
    joint = assemble_joint_vector([a, b], {("galaxy", "lensing"): np.array([[0.01], [0.02]])})
    assert joint.data.shape == (3,)
    assert joint.covariance.shape == (3, 3)
    assert np.isfinite(joint_log_likelihood(joint))
    with pytest.raises(ValueError):
        assemble_joint_vector([a, b], {("galaxy", "lensing"): np.ones((3, 2))})
