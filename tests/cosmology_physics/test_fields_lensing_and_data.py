import numpy as np
import pytest
from scipy.integrate import trapezoid

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.initial_conditions import (
    generate_gaussian_density_field,
    wave_number_magnitude_grid,
)
from nufield.cosmology.physics.lensing import (
    convergence_kernel_mpc_inv,
    lensing_efficiency,
    limber_convergence_cl,
    normalize_n_of_z,
    shear_shape_noise_cl,
)
from nufield.cosmology.physics.lensing_tomography import (
    source_bin_distributions,
    tomographic_lensing_kernels,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.particle_mesh import cic_density_grid, random_catalogue
from nufield.cosmology.physics.photoz import observed_bin_distribution, photoz_migration_matrix
from nufield.cosmology.physics.power_estimator import radial_power_spectrum_periodic
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum
from nufield.cosmology.physics.survey import effective_volume_mpc3, fkp_weight, shot_noise_power


def test_power_spectrum_estimator_returns_expected_shapes():
    rng = np.random.default_rng(3)
    field = rng.normal(size=(16, 16, 16))
    k, p, n, e = radial_power_spectrum_periodic(field, box_size_mpc=500, n_bins=8)
    assert k.shape == p.shape == n.shape == e.shape == (8,)
    assert np.all(n >= 0)
    assert np.all(np.isfinite(p[n > 0]))
    assert np.all(e[n > 0] > 0)


def test_cic_assignment_conserves_weight_and_uniform_random_means_zero():
    pos = random_catalogue(3000, 100, seed=14)
    rho = cic_density_grid(pos, 100, 16, overdensity=False)
    delta = cic_density_grid(pos, 100, 16, overdensity=True)
    assert rho.sum() == pytest.approx(3000)
    assert np.mean(delta) == pytest.approx(0, abs=1e-6)


def test_linear_gaussian_field_reproducible_and_finite():
    k = np.geomspace(0.04, 1.0, 150)
    p = 1000 * k / (1 + k) ** 3
    s = MatterPowerSpectrum(k, p, 0, True, "test-spectrum")
    a = generate_gaussian_density_field(s, 8, 100, seed=2, strict_k_support=False)
    b = generate_gaussian_density_field(s, 8, 100, seed=2, strict_k_support=False)
    assert a.shape == (8, 8, 8) and np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert np.mean(a) == pytest.approx(0, abs=1e-6)
    assert wave_number_magnitude_grid(8, 100).shape == (8, 8, 8)


def test_lensing_source_normalization_and_kernel_shapes():
    bg = BackgroundCosmology(PhysicalCosmology())
    z = np.linspace(0.01, 2, 35)
    nz = z * z * np.exp(-z / 0.5)
    norm = normalize_n_of_z(z, nz)
    assert trapezoid(norm, z) == pytest.approx(1, abs=1e-3)
    g = lensing_efficiency(bg, z, z, norm)
    w = convergence_kernel_mpc_inv(bg, z, g)
    assert g.shape == w.shape == z.shape
    assert np.all(g >= 0) and np.all(w >= 0)
    assert shear_shape_noise_cl(0.26, 20) > 0


def test_limber_convergence_spectrum_is_nonnegative():
    bg = BackgroundCosmology(PhysicalCosmology())
    z = np.linspace(0.01, 2.0, 30)
    nz = z * z * np.exp(-z / 0.5)

    def p(k, z):
        return 1000 * k / (1 + k) ** 3 / (1 + z) ** 2

    cl = limber_convergence_cl(np.array([50, 100, 300]), bg, z, nz, p)
    assert cl.shape == (3,)
    assert np.all(np.isfinite(cl)) and np.all(cl >= 0)


def test_tomographic_kernels_are_ordered_and_finite():
    bg = BackgroundCosmology(PhysicalCosmology())
    z = np.linspace(0.01, 2.0, 60)
    nz = z * z * np.exp(-z / 0.6)
    bins = source_bin_distributions(z, nz, np.array([0.01, 0.7, 1.3, 2.0]))
    assert bins.shape == (3, len(z))
    kernels = tomographic_lensing_kernels(bg, z, bins)
    assert kernels.shape == bins.shape
    assert np.isfinite(kernels).all()


def test_photoz_migration_is_probability_bounded():
    zt = np.linspace(0, 2, 41)
    edges = np.linspace(0, 2, 21)
    mat = photoz_migration_matrix(zt, edges, sigma0=0.04, outlier_fraction=0.02)
    assert mat.shape == (20, 41)
    assert np.all(mat >= 0) and np.all(mat <= 1)
    obs = observed_bin_distribution(np.ones(41), mat)
    assert obs.shape == (20,) and np.all(obs >= 0)


def test_survey_noise_and_effective_volume():
    bg = BackgroundCosmology(PhysicalCosmology())
    z = np.linspace(0.1, 1.0, 25)
    nbar = np.full_like(z, 1e-3)
    assert shot_noise_power(1e-3) == pytest.approx(1000)
    assert np.all((fkp_weight(nbar, 1000) >= 0) & (fkp_weight(nbar, 1000) <= 1))
    assert effective_volume_mpc3(bg, z, nbar, 1000, 0.5) > 0
