"""Regression and limiting-case tests for the extended cosmological physics toolkit.

These tests verify software contracts, analytic limits, and intentionally simplified models. Passing
this file is not a substitute for solver-reference validation or independent survey mocks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nufield.cosmology.physics.cosmology_units_audit import (
    SpectrumConvention,
    audit_spectrum_pair,
    convert_spectrum_units,
)
from nufield.cosmology.physics.field_topology import (
    excursion_topology,
    topology_curve,
    topology_resolution_comparison,
)
from nufield.cosmology.physics.growth_equation_validation import (
    einstein_de_sitter_reference,
    validate_growth_history,
)
from nufield.cosmology.physics.initial_conditions_validation import (
    ensemble_initial_field_check,
    validate_gaussian_initial_field,
)
from nufield.cosmology.physics.neutrino_background_consistency import (
    continuity_resolution_study,
    validate_neutrino_continuity,
)
from nufield.cosmology.physics.neutrino_hierarchy_prior import (
    NeutrinoMassPrior,
    draw_neutrino_mass_prior,
    validate_mass_prior_samples,
)
from nufield.cosmology.physics.neutrino_mass_observables import (
    minimum_mass_sum_ev,
    solve_mass_eigenvalues,
)
from nufield.cosmology.physics.neutrino_suppression_diagnostics import (
    empirical_neutrino_power_response,
    instantaneous_free_streaming_k,
    neutrino_fraction,
)
from nufield.cosmology.physics.observable_likelihood_diagnostics import (
    correlation_from_covariance,
    covariance_eigenmode_report,
    leave_one_bin_out_sensitivity,
    residual_bin_diagnostics,
    whiten_residuals,
)
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.physics_pipeline_gates import (
    GateEvidence,
    assess_physics_campaign,
    standard_evidence_from_mapping,
    write_physics_gate_report,
)
from nufield.cosmology.physics.redshift_space_covariance import (
    covariance_finite_mode_report,
    gaussian_multipole_covariance,
    kaiser_multipoles_analytic,
    kaiser_power_mu,
    multipole_from_mu_samples,
)
from nufield.cosmology.physics.sigma8_validation import (
    dimensionless_power,
    sigma8_from_spectrum,
    sigma8_integration_convergence,
    top_hat_window,
)
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum
from nufield.cosmology.physics.survey_window_validation import (
    audit_survey_window,
    perturbation_amplification,
    regularized_window_inverse,
    window_inverse_resolution_report,
)
from nufield.cosmology.physics.theory_covariance_likelihood import (
    covariance_regularization_scan,
    diagonal_target_shrinkage,
    evaluate_cosmology_likelihood,
    hartlap_precision_factor,
    sample_covariance_from_mocks,
)
from nufield.cosmology.physics.transfer_function_diagnostics import (
    compare_power_ratio,
    spectral_support_report,
    summarize_spectrum_shape,
)
from nufield.cosmology.physics.tree_bispectrum_model import (
    f2_kernel_einstein_de_sitter,
    reduced_tree_bispectrum,
    tree_bispectrum,
    triangle_cosine,
)
from nufield.cosmology.physics.window import PowerWindow


def make_spectrum(redshift: float = 0.0, amplitude: float = 1.0) -> MatterPowerSpectrum:
    k = np.geomspace(1e-4, 5.0, 1200)
    p = amplitude * 1.2e4 * k**0.96 / (1 + (k / 0.18) ** 2) ** 2
    return MatterPowerSpectrum(k, p, redshift, True, "controlled-test")


def test_spectrum_unit_conversion_is_paired_and_reversible():
    k = np.geomspace(0.001, 1, 16)
    p = np.geomspace(100, 1000, 16)
    kh, ph = convert_spectrum_units(
        k,
        p,
        h=0.7,
        input_k_unit="Mpc^-1",
        input_power_unit="Mpc^3",
        output_k_unit="h/Mpc",
        output_power_unit="(Mpc/h)^3",
    )
    assert kh == pytest.approx(k / 0.7)
    assert ph == pytest.approx(p * 0.7**3)
    k0, p0 = convert_spectrum_units(
        kh, ph, h=0.7, input_k_unit="h/Mpc", input_power_unit="(Mpc/h)^3"
    )
    assert k0 == pytest.approx(k)
    assert p0 == pytest.approx(p)


def test_unit_audit_blocks_component_mismatch():
    k = np.geomspace(0.01, 1, 20)
    p = np.ones_like(k)
    a = SpectrumConvention(component="total", redshift=0.0)
    b = SpectrumConvention(k_unit="h/Mpc", power_unit="(Mpc/h)^3", component="cb", redshift=0.0)
    kh, ph = convert_spectrum_units(
        k,
        p,
        h=0.7,
        input_k_unit="Mpc^-1",
        input_power_unit="Mpc^3",
        output_k_unit="h/Mpc",
        output_power_unit="(Mpc/h)^3",
    )
    result = audit_spectrum_pair(k, p, a, kh, ph, b, h=0.7)
    assert not result["valid_comparison"]
    assert any("components differ" in x for x in result["errors"])


def test_sigma8_integral_is_positive_and_sampling_stability_report_is_serializable():
    s = make_spectrum()
    assert top_hat_window(0) == pytest.approx(1.0)
    assert top_hat_window(np.array([0.0, 1e-5, 1.0])).shape == (3,)
    assert np.all(dimensionless_power(s.k_mpc_inv, s.pk_mpc3) > 0)
    val = sigma8_from_spectrum(s, 0.674)
    assert np.isfinite(val) and val > 0
    report = sigma8_integration_convergence(s, 0.674, n_k_values=(64, 128, 256))
    assert len(report.estimates) == 3
    json.dumps(report.to_dict())


def test_sigma8_rejects_missing_spectral_support():
    s = make_spectrum()
    from nufield.cosmology.physics.sigma8_validation import sigma_r_squared

    with pytest.raises(ValueError, match="support"):
        sigma_r_squared(s, 8, k_min_mpc_inv=1e-8, k_max_mpc_inv=0.2)


def test_hierarchy_mass_solution_and_mass_prior_support():
    for hierarchy in ("normal", "inverted"):
        floor = minimum_mass_sum_ev(hierarchy)
        masses = solve_mass_eigenvalues(floor + 0.05, hierarchy)
        assert sum(masses) == pytest.approx(floor + 0.05, abs=1e-10)
        with pytest.raises(ValueError):
            solve_mass_eigenvalues(floor * 0.8, hierarchy)
        prior = NeutrinoMassPrior(hierarchy, floor, floor + 0.2, "uniform_sum")
        samples = draw_neutrino_mass_prior(prior, 500, seed=17)
        assert np.all((samples >= floor) & (samples <= floor + 0.2))
        assert np.allclose(samples, draw_neutrino_mass_prior(prior, 500, seed=17))
        assert validate_mass_prior_samples(samples, prior)["support_valid"]


def test_log_lightest_prior_and_bad_bounds():
    floor = minimum_mass_sum_ev("normal")
    prior = NeutrinoMassPrior("normal", floor + 1e-5, floor + 0.2, "log_lightest")
    transformed = prior.transform_unit_cube(np.linspace(0, 1, 20))
    assert np.all(np.diff(transformed) > 0)
    with pytest.raises(ValueError):
        NeutrinoMassPrior("normal", floor - 0.01, floor + 0.2)


def test_neutrino_continuity_equation_is_consistent_in_relativistic_limit():
    report = validate_neutrino_continuity(0.0, z_max=100, n_samples=256, tolerance=0.03)
    assert report.w_min == pytest.approx(1 / 3, abs=1e-8)
    assert report.w_max == pytest.approx(1 / 3, abs=1e-8)
    assert report.median_fractional_continuity_residual < 0.03
    study = continuity_resolution_study(0.1, resolutions=(64, 128, 256), z_max=100)
    assert len(study["median_residuals"]) == 3


def test_growth_equation_validation_rejects_bad_grids():
    # The function is checked against an EdS analytic model through the public validator.
    eds = einstein_de_sitter_reference(256, a_min=0.1)
    report = validate_growth_history(
        eds["scale_factor"], eds["growth"], eds["H_km_s_mpc"], eds["omega_m_clustering"]
    )
    assert report.max_abs_normalized_residual_interior < 0.05
    with pytest.raises(ValueError):
        validate_growth_history(np.array([1.0, 0.5]), np.ones(2), np.ones(2), np.ones(2))


def test_kaiser_analytic_multipoles_match_mu_integration():
    k = np.array([0.1])
    mu = np.linspace(-1, 1, 20001)
    p_mu = kaiser_power_mu(k[0], mu, 1500.0, 1.7, 0.8)
    numeric = multipole_from_mu_samples(p_mu, mu)
    analytic = kaiser_multipoles_analytic(np.array([1500.0]), 1.7, 0.8)
    assert numeric[0] == pytest.approx(analytic[0][0], rel=1e-7)
    assert numeric[2] == pytest.approx(analytic[2][0], rel=1e-6)
    assert numeric[4] == pytest.approx(analytic[4][0], rel=5e-6)


def test_kaiser_covariance_positive_definite_and_scales_with_modes():
    k = np.array([0.05, 0.1, 0.15])
    n = np.array([500, 600, 700])
    p = np.array([3000.0, 1800.0, 900.0])
    c = gaussian_multipole_covariance(k, n, p, 1.8, 0.75, mu_order=80)
    report = covariance_finite_mode_report(c)
    assert report["positive_definite"]
    c2 = gaussian_multipole_covariance(k, 2 * n, p, 1.8, 0.75, mu_order=80)
    assert c2 == pytest.approx(c / 2)


def test_covariance_likelihood_families_and_hartlap_guard():
    rng = np.random.default_rng(12)
    mocks = rng.normal(size=(120, 5))
    cov = sample_covariance_from_mocks(mocks)
    assert hartlap_precision_factor(120, 5) > 0
    result_g = evaluate_cosmology_likelihood(np.zeros(5), np.ones(5) * 0.1, cov, family="gaussian")
    result_t = evaluate_cosmology_likelihood(
        np.zeros(5), np.ones(5) * 0.1, cov, family="sellentin_heavens", n_mocks=120
    )
    assert np.isfinite(result_g.log_likelihood)
    assert np.isfinite(result_t.log_likelihood)
    with pytest.raises(ValueError):
        hartlap_precision_factor(7, 5)
    with pytest.raises(ValueError):
        evaluate_cosmology_likelihood(
            np.zeros(5), np.zeros(5), cov, family="sellentin_heavens", n_mocks=5
        )


def test_covariance_shrinkage_preserves_diagonal_and_improves_near_singular_conditioning():
    c = np.array([[1.0, 0.9999, 0.9998], [0.9999, 1.0, 0.9997], [0.9998, 0.9997, 1.0]])
    shrunk = diagonal_target_shrinkage(c, 0.2)
    assert np.diag(shrunk) == pytest.approx(np.diag(c))
    scan = covariance_regularization_scan(c, (0.1, 0.5, 0.9))
    assert len(scan) == 3
    assert scan[-1]["condition_number"] < scan[0]["condition_number"]


def test_likelihood_residual_diagnostics_and_correlation():
    c = np.array([[4.0, 0.5], [0.5, 1.0]])
    corr = correlation_from_covariance(c)
    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[0, 1] == pytest.approx(0.25)
    w = whiten_residuals(np.array([1.0, -0.5]), c)
    assert np.sum(w * w) == pytest.approx(
        np.array([1.0, -0.5]) @ np.linalg.solve(c, np.array([1.0, -0.5]))
    )
    modes = covariance_eigenmode_report(c)
    assert modes["condition_number"] > 1
    leave = leave_one_bin_out_sensitivity(np.array([1.0, -0.5]), c)
    assert len(leave) == 2
    bins = residual_bin_diagnostics(
        np.array([1.0, 2.0]), np.array([0.5, 1.0]), c, labels=("a", "b")
    )
    assert [x.label for x in bins] == ["a", "b"]


def test_transfer_ratio_and_shape_diagnostics():
    a, b = make_spectrum(amplitude=1.1), make_spectrum(amplitude=1.0)
    report = compare_power_ratio(a, b, n_k=80)
    assert report.low_k_median == pytest.approx(1.1)
    assert report.high_k_median == pytest.approx(1.1)
    support = spectral_support_report(a, required_k=(1e-3, 1.0))
    assert support["fully_supported"]
    shape = summarize_spectrum_shape(a)
    assert shape["n_k"] == a.k_mpc_inv.size


def test_neutrino_response_only_marked_interpretable_with_attestation():
    reference = make_spectrum(amplitude=1)
    massive = make_spectrum(amplitude=0.97)
    raw = empirical_neutrino_power_response(massive, reference)
    assert not raw["interpretation_eligible"]
    matched = empirical_neutrino_power_response(massive, reference, same_component_verified=True)
    assert matched["median_response"] == pytest.approx(-0.03)
    assert instantaneous_free_streaming_k(0.1, 0, 67.4) > 0
    assert neutrino_fraction(PhysicalCosmology())[2] >= 0


def test_field_topology_known_topological_limits():
    n = 5
    full = excursion_topology(np.ones((n, n, n)), 0.5, 100, periodic=True)
    assert full.volume_fraction == 1
    assert full.surface_area_mpc2 == 0
    assert full.euler_characteristic == 0  # 3-torus
    empty = excursion_topology(np.zeros((n, n, n)), 1.1, 100, periodic=True)
    assert empty.volume_fraction == 0 and empty.euler_characteristic == 0
    one = np.zeros((n, n, n))
    one[0, 0, 0] = 2
    single = excursion_topology(one, 1, 100, periodic=True)
    assert single.volume_fraction == pytest.approx(1 / n**3)
    assert single.surface_area_mpc2 == pytest.approx(6 * (100 / n) ** 2)
    assert single.euler_characteristic == 1
    curve = topology_curve(
        np.random.default_rng(0).normal(size=(n, n, n)), np.array([-1.0, 0.0, 1.0]), 100
    )
    assert len(curve) == 3


def test_topology_resolution_comparison_matches_quantile_fractions():
    low = np.random.default_rng(0).normal(size=(4, 4, 4))
    high = np.random.default_rng(1).normal(size=(8, 8, 8))
    report = topology_resolution_comparison(low, high, 100, quantile_thresholds=(0.25, 0.5, 0.75))
    assert len(report["low_surface_area_mpc2"]) == 3
    assert report["low_grid_size"] == 4


def test_tree_level_bispectrum_equilateral_kernel_and_scaling():
    assert f2_kernel_einstein_de_sitter(1, 1, -0.5) == pytest.approx(2 / 7)
    assert triangle_cosine(1, 1, 1) == pytest.approx(-0.5)
    b = tree_bispectrum(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert b == pytest.approx(6 * (2 / 7))
    q = reduced_tree_bispectrum(b, 1.0, 1.0, 1.0)
    assert q == pytest.approx(4 / 7)
    with pytest.raises(ValueError):
        triangle_cosine(1, 1, 3)


def test_pipeline_gates_block_missing_or_failed_required_evidence(tmp_path):
    required = ("physics_equations", "solver_reference")
    evidence = [GateEvidence("physics_equations", True, "error", "test", "equations passed")]
    report = assess_physics_campaign("unit-test", evidence, required_checks=required)
    assert report.status == "blocked"
    assert "solver_reference" in report.missing_required_checks
    path = write_physics_gate_report(tmp_path / "gate.json", report)
    assert json.loads(path.read_text())["status"] == "blocked"
    mapped = standard_evidence_from_mapping({"x": {"passed": True, "summary": "ok"}})
    assert mapped[0].passed
    with pytest.raises(ValueError):
        standard_evidence_from_mapping({"x": {"summary": "forgot explicit status"}})


def test_window_audit_and_regularized_inverse_report():
    window = PowerWindow(np.array([[0.8, 0.2], [0.2, 0.8]]), provenance="synthetic-test")
    report = audit_survey_window(window)
    assert report.rank == 2
    t = np.array([2.0, 5.0])
    dt = np.array([0.1, -0.2])
    amp = perturbation_amplification(window, t, dt)
    assert amp["output_perturbation_norm"] > 0
    inverse = regularized_window_inverse(window, tikhonov_lambda=1e-3)
    assert inverse.shape == (2, 2)
    scan = window_inverse_resolution_report(window, lambdas=(1e-5, 1e-3, 1e-1))
    assert len(scan) == 3


def test_window_rank_deficiency_warns_against_direct_inverse():
    window = PowerWindow(np.array([[1.0, 0.0], [2.0, 0.0]]), provenance="synthetic-test")
    report = audit_survey_window(window)
    assert report.rank == 1
    assert "do not directly invert" in report.invertibility_recommendation


def test_initial_conditions_validation_reproducibility_and_ensemble():
    s = make_spectrum()
    # k support must cover modes up to the grid's diagonal Nyquist; use a roomy small-box spectrum.

    field, report = validate_gaussian_initial_field(s, 8, 50, seed=42, strict_k_support=False)
    assert field.shape == (8, 8, 8)
    assert abs(report.mean) < 1e-5
    assert report.finite
    ensemble = ensemble_initial_field_check(
        s,
        8,
        50,
        seeds=(1, 2, 3),
    )
    assert ensemble["n_seeds"] == 3


def test_covariance_mock_campaign_tracks_sample_size_without_claiming_independence():
    from nufield.cosmology.physics.covariance_mock_campaign import run_covariance_mock_campaign

    rng = np.random.default_rng(0)
    truth = np.array([[1.0, 0.2], [0.2, 2.0]])
    x = rng.multivariate_normal(np.zeros(2), truth, size=1000)
    report = run_covariance_mock_campaign(x, truth, mock_counts=(50, 100, 200), shrinkage_alpha=0.1)
    assert [r["n_mocks"] for r in report["trials"]] == [50, 100, 200]
    assert "do not increase" in report["interpretation"]
