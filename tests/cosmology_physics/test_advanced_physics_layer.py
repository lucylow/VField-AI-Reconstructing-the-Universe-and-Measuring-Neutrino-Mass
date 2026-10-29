from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from nufield.cosmology.physics.artifact_manifest import (
    build_manifest,
    canonical_configuration_hash,
    sha256_file,
    write_manifest,
)
from nufield.cosmology.physics.bao_damping import damped_bao_power, gaussian_bao_damping
from nufield.cosmology.physics.boltzmann_validation import (
    assert_spectrum_agreement,
    compare_matter_spectra,
)
from nufield.cosmology.physics.finite_covariance_likelihood import (
    diagnose_mock_covariance,
    estimate_covariance_from_mocks,
    hartlap_factor,
    multivariate_t_loglike_from_sample_covariance,
)
from nufield.cosmology.physics.linear_nuisance import (
    fit_linear_nuisance,
    marginalized_gaussian_loglike_linear_templates,
    polynomial_nuisance_templates,
)
from nufield.cosmology.physics.neutrino_phase_space import (
    sample_dimensionless_momentum,
    sample_neutrino_thermal_velocities,
)
from nufield.cosmology.physics.parameter_forecast import (
    figure_of_merit,
    gaussian_fisher_with_covariance_derivatives,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.physics_validation_suite import run_core_physics_validation
from nufield.cosmology.physics.posterior_diagnostics import (
    equal_tailed_interval,
    expected_calibration_rms,
    interval_coverage,
    randomized_pit,
)
from nufield.cosmology.physics.reconstruction_metrics import (
    aggregate_reconstruction_score,
    reconstruction_metrics_periodic,
)
from nufield.cosmology.physics.redshift_distributions import (
    effective_redshift,
    gaussian_photoz_bin_probability,
    photoz_migration_matrix,
    validate_redshift_distribution,
)
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum
from nufield.cosmology.physics.survey_window_convolution import (
    fold_theory_through_window,
    propagate_windowed_covariance,
    validate_window_operator,
)
from nufield.cosmology.physics.validity import PhysicsLevel, default_scope_for_level
from nufield.cosmology.physics.window import PowerWindow, identity_window


def test_redshift_distribution_normalization_and_photoz_probability_conservation():
    z = np.linspace(0, 3, 301)
    nz = z**2 * np.exp(-z)
    dist = validate_redshift_distribution(z, nz, normalize=True, label="toy")
    assert trapezoid(dist.density_per_redshift, z) == pytest.approx(1.0, abs=1e-10)
    probs = photoz_migration_matrix(
        z, np.linspace(0, 3, 7), sigma0=0.05, include_under_overflow=True
    )
    assert np.allclose(probs.sum(axis=0), 1.0, atol=2e-10)
    assert np.all((probs >= 0) & (probs <= 1))
    assert gaussian_photoz_bin_probability(np.array([0.1, 0.5]), 0, 1, sigma0=0).tolist() == [
        1.0,
        1.0,
    ]
    assert effective_redshift(z, nz) > 0


def test_neutrino_phase_space_seed_and_speed_limit():
    q1 = sample_dimensionless_momentum(1000, seed=14)
    q2 = sample_dimensionless_momentum(1000, seed=14)
    assert np.array_equal(q1, q2)
    assert np.mean(q1) == pytest.approx(3.151, abs=0.2)
    state = sample_neutrino_thermal_velocities(0.1, 0.0, 500, seed=7)
    assert state["velocity_km_s"].shape == (500, 3)
    assert np.max(state["speed_km_s"]) < 299792.458
    assert isinstance(state["metadata_note"], str)


def test_finite_covariance_likelihood_and_diagnostics():
    rng = np.random.default_rng(8)
    mocks = rng.normal(size=(150, 4))
    c = estimate_covariance_from_mocks(mocks)
    report = diagnose_mock_covariance(c, 150)
    assert report.can_hartlap_correct
    assert hartlap_factor(150, 4) == pytest.approx(144 / 149)
    loglike = multivariate_t_loglike_from_sample_covariance(np.zeros(4), np.zeros(4), c, 150)
    assert np.isfinite(loglike)
    with pytest.raises(ValueError):
        hartlap_factor(6, 4)


def test_linear_nuisance_fit_and_marginal_likelihood():
    x = np.linspace(-1, 1, 30)
    t = polynomial_nuisance_templates(x, 1)
    cov = np.eye(x.size) * 0.1**2
    true_a = np.array([2.0, -0.7])
    data = 1.0 + t @ true_a
    result = fit_linear_nuisance(data, np.ones_like(data), t, cov)
    assert result.coefficients == pytest.approx(true_a, abs=1e-10)
    assert result.chi_squared < 1e-18
    loglike = marginalized_gaussian_loglike_linear_templates(
        data, np.ones_like(data), t, cov, np.zeros(2), np.eye(2)
    )
    assert np.isfinite(loglike)


def test_bao_damping_limits_and_spectrum_comparison():
    k = np.geomspace(0.005, 0.4, 500)
    smooth = 1e4 * k**-1.2
    linear = smooth * (1 + 0.04 * np.sin(105 * k))
    assert np.all(gaussian_bao_damping(k, 0.0) == 1.0)
    assert damped_bao_power(k, linear, smooth, 0.0) == pytest.approx(linear)
    assert damped_bao_power(k, linear, smooth, 2000.0) == pytest.approx(smooth, rel=1e-8, abs=1e-12)
    a = MatterPowerSpectrum(k, linear, 0, source="reference")
    b = MatterPowerSpectrum(k, linear * 1.001, 0, source="candidate")
    comp = compare_matter_spectra(a, b, n_points=64)
    assert comp.log_ratio_rms == pytest.approx(np.log(1.001), abs=1e-10)
    assert_spectrum_agreement(comp, max_log_rms=0.002)
    with pytest.raises(AssertionError):
        assert_spectrum_agreement(comp, max_log_rms=0.0001)


def test_generalized_gaussian_fisher_with_covariance_derivative():
    theta = np.array([1.5])
    result = gaussian_fisher_with_covariance_derivatives(
        ["a"],
        theta,
        np.array([1e-4]),
        mean_fn=lambda p: np.array([p[0], 2 * p[0]]),
        covariance_fn=lambda p: np.eye(2) * (1 + p[0] ** 2),
    )
    assert result.fisher_matrix[0, 0] > 0
    assert result.marginalized_sigma[0] > 0
    assert figure_of_merit(np.diag([0.25, 1.0]), (0, 1)) == pytest.approx(2.0)


def test_reconstruction_metrics_same_field_and_amplitude_rescaling():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(16, 16, 16))
    exact = reconstruction_metrics_periodic(truth, truth, 200, n_bins=8)
    valid = np.isfinite(exact.cross_correlation)
    assert np.allclose(exact.cross_correlation[valid], 1.0, atol=1e-10)
    doubled = reconstruction_metrics_periodic(truth, 2 * truth, 200, n_bins=8)
    valid = np.isfinite(doubled.propagator)
    assert np.allclose(doubled.propagator[valid], 2.0, atol=1e-10)
    assert aggregate_reconstruction_score(
        exact, k_min=0, k_max=1, metric="cross_correlation"
    ) == pytest.approx(1.0)


def test_survey_window_forward_folding_and_covariance():
    w = PowerWindow(np.array([[0.8, 0.2], [0.1, 0.9]]), provenance="synthetic-test")
    t = np.array([2.0, 4.0])
    c = np.diag([1.0, 4.0])
    folded = fold_theory_through_window(t, w, covariance_native=c)
    assert folded.theory_windowed == pytest.approx([2.4, 3.8])
    assert folded.covariance_windowed == pytest.approx(w.matrix @ c @ w.matrix.T)
    assert propagate_windowed_covariance(w, c) == pytest.approx(folded.covariance_windowed)
    assert validate_window_operator(w)["rows_sum_to_one"]
    assert fold_theory_through_window(t, identity_window(2)).theory_windowed == pytest.approx(t)


def test_coverage_and_pit_utilities():
    truths = np.linspace(-1, 1, 100)
    samples = np.repeat(truths[:, None], 300, axis=1) + np.random.default_rng(3).normal(
        0, 0.1, (100, 300)
    )
    lo, hi = equal_tailed_interval(samples[0], 0.68)
    assert lo < hi
    cov = interval_coverage(truths, samples)
    assert len(cov) == 2 and all(0 <= x.empirical_coverage <= 1 for x in cov)
    pit = randomized_pit(truths, samples, seed=2)
    assert np.all((pit >= 0) & (pit <= 1))
    assert expected_calibration_rms(np.array([0.7, 0.94]), np.array([0.68, 0.95])) < 0.03


def test_physics_validation_report_passes_reference_background():
    report = run_core_physics_validation(PhysicalCosmology())
    assert report.passed, report.to_dict()
    assert any("not validation against CAMB/CLASS" in w for w in report.warnings)


def test_artifact_manifest_hashes_are_deterministic(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("nu-field\n")
    assert sha256_file(a) == sha256_file(a)
    assert canonical_configuration_hash({"b": 2, "a": 1}) == canonical_configuration_hash(
        {"a": 1, "b": 2}
    )
    scope = default_scope_for_level(PhysicsLevel.HOMOGENEOUS_BACKGROUND)
    manifest = build_manifest(
        "test",
        PhysicalCosmology(),
        scope,
        backend_name="toy",
        backend_version="0",
        output_units={"H": "km/s/Mpc"},
        dataset_ids=("x",),
        data_fingerprints=(sha256_file(a),),
    )
    out = write_manifest(tmp_path / "manifest.json", manifest)
    assert out.exists() and "configuration_hash" in out.read_text()
