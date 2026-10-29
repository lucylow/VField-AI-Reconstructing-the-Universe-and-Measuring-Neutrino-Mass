"""Independent generators for the NUFIELD synthetic demo artifacts.

All functions accept explicit random.Random instances or a bundle seed. The
module never imports the scientific simulator or inference code. Every data
row has a synthetic marker, and every report contains a warning string.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from .profiles import (
    PARAMETER_DEFINITIONS,
    PARAMETER_KEYS,
    SCIENCE_WARNING,
    DemoScenario,
)
from .randoms import clamp, normal, rounded, split_counts, uniform

SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "0.4.0"
SOURCE_KIND = "synthetic-demo-generator"


def generate_parameter_rows(
    rng: random.Random,
    scenario: DemoScenario,
    *,
    sample_count: int,
    seed: int,
    run_id: str,
) -> list[dict[str, Any]]:
    """Generate bounded parameter rows with split metadata and toy correlations."""
    split_sizes = split_counts(sample_count)
    cut_train = split_sizes["train"]
    cut_validation = cut_train + split_sizes["validation"]
    center = scenario.resolved_center()
    rows: list[dict[str, Any]] = []

    for index in range(sample_count):
        split = "train" if index < cut_train else "validation" if index < cut_validation else "test"
        heldout_shift = scenario.domain_shift if split == "test" else 0.0
        latent = normal(rng)
        shared_second = normal(rng)
        row: dict[str, Any] = {}

        for key in PARAMETER_KEYS:
            definition = PARAMETER_DEFINITIONS[key]
            center_value = center[key]
            perturbation = normal(rng, sigma=definition["sigma"] * scenario.spread)

            # Synthetic correlations are intentional art direction for plots.
            # They must not be interpreted as posterior covariance.
            if key == "omega_m":
                perturbation += latent * definition["sigma"] * 0.44
                perturbation += heldout_shift * 0.42
            elif key == "sigma8":
                perturbation -= latent * definition["sigma"] * 0.38
                perturbation -= heldout_shift * 0.25
            elif key == "h":
                perturbation += shared_second * definition["sigma"] * 0.62
            elif key == "sum_mnu_ev":
                perturbation += scenario.calibration_offset * 0.24
            elif key == "bias":
                perturbation += scenario.calibration_offset * 0.15

            row[key] = rounded(
                clamp(center_value + perturbation, definition["min"], definition["max"]),
                6,
            )

        score = clamp(
            0.94
            - scenario.observation_noise * rng.random() * 0.15
            - (0.35 if rng.random() < 0.025 else 0.0),
            0.18,
            0.995,
        )
        row.update(
            {
                "id": f"{run_id}-sample-{index + 1:06d}",
                "realization_id": f"realization-{index + 1:06d}",
                "split": split,
                "synthetic_score": rounded(score, 5),
                "quality_flag": "review" if score < 0.65 else "synthetic-ok",
                "synthetic_noise_scale": rounded(
                    scenario.observation_noise * uniform(rng, 0.75, 1.25), 5
                ),
                "random_seed": seed,
                "is_synthetic": True,
            }
        )
        rows.append(row)

    if scenario.split_leakage and len(rows) > 4:
        train = next((row for row in rows if row["split"] == "train"), None)
        test = next((row for row in rows if row["split"] == "test"), None)
        if train and test:
            test["realization_id"] = train["realization_id"]
            test["intentional_overlap_fixture"] = True
    return rows


def generate_posterior_chains(
    rng: random.Random,
    scenario: DemoScenario,
    *,
    chain_count: int,
    samples_per_chain: int,
    run_id: str,
) -> dict[str, Any]:
    """Generate random-walk-looking pseudo-traces for plots, never inference."""
    center = scenario.resolved_center()
    chains: list[dict[str, Any]] = []
    for chain_index in range(chain_count):
        previous = dict(center)
        samples: list[dict[str, Any]] = []
        for iteration in range(samples_per_chain):
            sample: dict[str, Any] = {"iteration": iteration}
            for key in PARAMETER_KEYS:
                metadata = PARAMETER_DEFINITIONS[key]
                proposal = previous[key] + normal(
                    rng,
                    sigma=metadata["sigma"] * (0.24 + 0.12 * scenario.spread),
                )
                bounded = clamp(proposal, metadata["min"], metadata["max"])
                previous[key] = bounded
                sample[key] = rounded(bounded, 6)
            log_density = -0.5 * sum(
                ((sample[key] - center[key]) / PARAMETER_DEFINITIONS[key]["sigma"]) ** 2
                for key in PARAMETER_KEYS
            )
            sample.update(
                {
                    "log_density_toy": rounded(log_density, 6),
                    "accepted": rng.random() > 0.22,
                    "burn_in": iteration < max(1, int(samples_per_chain * 0.1)),
                    "is_synthetic": True,
                }
            )
            samples.append(sample)
        chains.append(
            {
                "chain_id": f"chain-{chain_index + 1}",
                "seed_offset": chain_index * 100_003,
                "samples": samples,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "is_synthetic": True,
        "sampler_label": "random-walk illustration (not a production sampler)",
        "run_id": run_id,
        "chains": chains,
        "warning": (
            "Generated pseudo-chains exist for display and diagnostics testing only. "
            "They are not MCMC posterior samples and have no convergence guarantee."
        ),
    }


def generate_power_spectrum(
    rng: random.Random,
    scenario: DemoScenario,
    *,
    count: int,
) -> dict[str, Any]:
    """Generate a chart-ready arbitrary curve with a clearly stated toy model."""
    points: list[dict[str, Any]] = []
    for index in range(count):
        u = index / max(1, count - 1)
        wave_number = 10 ** (-2 + u * 3.2)
        shape = (wave_number / 0.08) ** 0.82 / (1 + (wave_number / 0.8) ** 1.1)
        turnover = 0.66 + 0.34 / (1 + (wave_number / 0.055) ** 1.55)
        wiggle = 1 + 0.05 * math.sin(7.1 * math.log(wave_number + 0.003))
        suppression = math.exp(
            -(0.25 + scenario.resolved_center()["sum_mnu_ev"] * 0.5) * (wave_number / 1.9) ** 0.62
        )
        noise_scale = 1 + normal(rng, sigma=0.025 * scenario.observation_noise)
        value = max(1e-7, shape * turnover * wiggle * suppression * noise_scale)
        reference = max(1e-7, shape * turnover * wiggle)
        points.append(
            {
                "k": rounded(wave_number, 8),
                "p_toy": rounded(value, 10),
                "p_reference_toy": rounded(reference, 10),
                "relative_noise": rounded(noise_scale - 1, 7),
                "is_synthetic": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "x_name": "k",
        "x_unit": "h/Mpc (display label only)",
        "y_name": "P_toy(k)",
        "y_unit": "arbitrary units",
        "transform": "hand-designed display curve; not a physical transfer function",
        "points": points,
        "is_synthetic": True,
        "scientific_warning": SCIENCE_WARNING,
    }


def generate_expansion_history(
    rng: random.Random,
    scenario: DemoScenario,
    *,
    count: int,
) -> dict[str, Any]:
    """Generate a redshift/scale-factor table and a nonphysical distance proxy."""
    points: list[dict[str, Any]] = []
    for index in range(count):
        fraction = index / max(1, count - 1)
        redshift = rounded(10 * (1 - fraction), 6)
        scale_factor = 1 / (1 + redshift)
        toy_distance = 1.25 * (1 - math.exp(-redshift / 2.8))
        toy_distance += normal(rng, sigma=0.006 * scenario.observation_noise)
        epoch = "late" if redshift < 0.1 else "intermediate" if redshift < 2 else "early"
        points.append(
            {
                "redshift": redshift,
                "scale_factor": rounded(scale_factor, 8),
                "toy_distance_index": rounded(max(0, toy_distance), 8),
                "epoch_label": epoch,
                "is_synthetic": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "x_name": "redshift",
        "y_names": ["scale_factor", "toy_distance_index"],
        "points": points,
        "is_synthetic": True,
        "scientific_warning": (
            "Only scale_factor = 1/(1+z) is a definitional relation here. "
            "toy_distance_index is arbitrary UI data, not a distance calculation."
        ),
    }


def generate_survey_catalog(
    rng: random.Random,
    scenario: DemoScenario,
    *,
    count: int,
    run_id: str,
) -> dict[str, Any]:
    """Generate an angular catalog with explicit toy-coordinate field names."""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        redshift = clamp(-math.log(max(0.0001, rng.random())) * 0.58, 0.005, 3.25)
        right_ascension = rng.random() * 360
        declination = math.degrees(math.asin(rng.random() * 2 - 1))
        latent = math.sin(math.radians(right_ascension + 13)) * math.cos(math.radians(declination))
        latent += normal(rng, sigma=0.35)
        noise = normal(rng, sigma=0.08 * scenario.observation_noise * (1 + redshift * 0.24))
        completeness = clamp(
            0.97 - redshift * 0.12 - rng.random() * 0.12 * scenario.observation_noise, 0.12, 0.995
        )
        masked_probability = 0.12 if scenario.observation_noise > 1.5 else 0.045
        rows.append(
            {
                "object_id": f"{run_id}-obj-{index + 1:07d}",
                "ra_deg": rounded(right_ascension, 7),
                "dec_deg": rounded(declination, 7),
                "redshift_toy": rounded(redshift, 7),
                "density_proxy_true_toy": rounded(latent, 7),
                "density_proxy_observed_toy": rounded(
                    latent + noise + scenario.calibration_offset, 7
                ),
                "uncertainty_toy": rounded(abs(noise) + 0.035 * scenario.observation_noise, 7),
                "completeness_fraction_toy": rounded(completeness, 6),
                "mask_flag": "masked-demo"
                if rng.random() < masked_probability
                else "unmasked-demo",
                "survey_region": rng.choice(["north-demo", "equatorial-demo", "south-demo"]),
                "is_synthetic": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_label": "Synthetic angular catalog",
        "row_count": len(rows),
        "columns": list(rows[0]) if rows else [],
        "rows": rows,
        "is_synthetic": True,
        "scientific_warning": (
            "Coordinates and redshift-like values are generated for UI performance tests; "
            "they are not a real survey catalog."
        ),
    }


def generate_cmb_summary(
    rng: random.Random, scenario: DemoScenario, *, count: int
) -> dict[str, Any]:
    """Generate a small synthetic C_ell-like curve, without observational provenance."""
    points: list[dict[str, Any]] = []
    for ell in range(2, count + 2):
        acoustic = 1 + 0.32 * math.sin(ell / 16) ** 2 + 0.18 * math.sin(ell / 42)
        damping = math.exp(-ell / 850)
        value = (0.18 + acoustic * damping) * (
            1 + normal(rng, sigma=0.018 * scenario.observation_noise)
        )
        error = 0.015 + rng.random() * 0.025 * scenario.observation_noise
        points.append(
            {
                "ell": ell,
                "cl_toy": rounded(max(0.0001, value), 9),
                "uncertainty_toy": rounded(error, 9),
                "is_synthetic": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "quantity": "C_ell toy summary",
        "points": points,
        "generated_texture": True,
        "pixelization": "not HEALPix",
        "is_synthetic": True,
        "scientific_warning": (
            "A hand-shaped demonstration curve, not Planck/WMAP measurements or a "
            "Boltzmann calculation."
        ),
    }


def generate_lensing_catalog(
    rng: random.Random,
    scenario: DemoScenario,
    *,
    count: int,
    run_id: str,
) -> dict[str, Any]:
    """Generate toy image positions from the dimensionless point-mass equation."""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        beta = (rng.random() * 1.8 - 0.9) * 1.1
        theta_e = 0.35 + rng.random() * 0.55
        root = math.sqrt(beta * beta + 4 * theta_e * theta_e)
        theta_plus = (beta + root) / 2
        theta_minus = (beta - root) / 2
        noise = scenario.observation_noise * 0.01
        rows.append(
            {
                "system_id": f"{run_id}-lens-{index + 1:05d}",
                "source_beta_arcsec_toy": rounded(beta, 6),
                "theta_e_arcsec_toy": rounded(theta_e, 6),
                "image_plus_arcsec_toy": rounded(theta_plus + normal(rng, sigma=noise), 6),
                "image_minus_arcsec_toy": rounded(theta_minus + normal(rng, sigma=noise), 6),
                "magnification_proxy_toy": rounded(1 + theta_e / math.sqrt(beta * beta + 0.04), 6),
                "fit_residual_toy": rounded(
                    abs(normal(rng)) * 0.02 * scenario.observation_noise, 7
                ),
                "is_synthetic": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "lens_model": "point-mass equation in arbitrary angular units",
        "rows": rows,
        "is_synthetic": True,
        "scientific_warning": "UI fixture only; values are not real lens masses or cosmological distances.",
    }


def generate_benchmark_report(
    rng: random.Random,
    scenario: DemoScenario,
    *,
    sample_rows: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    """Create importer-compatible benchmark metadata and coverage diagnostics."""
    median_absolute_error: dict[str, float] = {}
    for key in PARAMETER_KEYS:
        scale = float(PARAMETER_DEFINITIONS[key]["sigma"])
        base = scale * (0.07 + scenario.observation_noise * 0.075)
        error = base * (1 + abs(normal(rng)) * 0.2)
        error += abs(scenario.calibration_offset) * 0.2 + scenario.domain_shift * 0.18
        median_absolute_error[key] = rounded(max(0.0001, error), 7)

    coverages: list[dict[str, Any]] = []
    for nominal in (0.5, 0.68, 0.9, 0.95, 0.99):
        observed = clamp(nominal + scenario.coverage_shift + normal(rng, sigma=0.012), 0.02, 0.995)
        coverages.append(
            {
                "nominal": nominal,
                "observed": rounded(observed, 5),
                "n": max(20, len(sample_rows)),
                "interval_level": nominal,
                "status": "under-coverage-demo"
                if observed < nominal - 0.06
                else "over-coverage-demo"
                if observed > nominal + 0.06
                else "within-demo-band",
                "is_synthetic": True,
            }
        )

    split_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for row in sample_rows:
        split_ids[row["split"]].append(row["realization_id"])
    overlaps = any(
        set(split_ids[left]).intersection(split_ids[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": f"{run_id}-benchmark",
        "method": {
            "algorithm": "synthetic-demo-regressor",
            "version": GENERATOR_VERSION,
            "note": "A display placeholder; no predictive model was fitted.",
        },
        "parameter_names": list(PARAMETER_KEYS),
        # The frontend scatter chart needs point records in the benchmark JSON.
        # This is a bounded preview; the full table is in parameter_samples.json.
        "parameter_points": sample_rows[:600],
        "median_absolute_error": median_absolute_error,
        "coverage_diagnostic": coverages,
        "n_train": len(split_ids["train"]),
        "n_validation_held_out": len(split_ids["validation"]),
        "n_test": len(split_ids["test"]),
        "train_ids": sorted(set(split_ids["train"])),
        "validation_ids": sorted(set(split_ids["validation"])),
        "test_ids": sorted(set(split_ids["test"])),
        "split_disjoint": not overlaps,
        "physics_level": "toy",
        "scenario": scenario.scenario_id,
        "is_synthetic": True,
        "scientific_warning": SCIENCE_WARNING,
    }


def generate_validation_report(
    scenario: DemoScenario,
    benchmark_report: dict[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Build predictable PASS/WARN/FAIL states for frontend diagnostic cards."""
    checks: list[dict[str, Any]] = [
        make_check(
            "schema_shape", "Artifact shape", "PASS", "Required keys exist in generated fixtures."
        ),
        make_check(
            "finite_values",
            "Finite display values",
            "PASS",
            "The generator clamps generated numeric fields to bounded ranges.",
        ),
        make_check(
            "scenario_provenance",
            "Synthetic provenance label",
            "PASS",
            "Synthetic provenance is attached to exported artifacts.",
        ),
        make_check(
            "split_disjointness",
            "Realization split disjointness",
            "FAIL" if benchmark_report["split_disjoint"] is False else "PASS",
            "Intentional overlap was injected by this stress profile."
            if benchmark_report["split_disjoint"] is False
            else "No shared realization IDs were found in generated splits.",
        ),
    ]
    under_coverage = any(
        item["observed"] < item["nominal"] - 0.06
        for item in benchmark_report["coverage_diagnostic"]
    )
    checks.append(
        make_check(
            "coverage_demo",
            "Coverage diagnostic demo",
            "WARN" if under_coverage else "PASS",
            "Coverage values are artificially depressed in this profile."
            if under_coverage
            else "Illustrative coverage values are inside the broad demo tolerance.",
        )
    )
    if scenario.failure_mode == "calibration-offset":
        checks.append(
            make_check(
                "calibration_offset",
                "Estimator calibration",
                "WARN",
                "A synthetic calibration offset was injected; no real estimator was fitted.",
            )
        )
    if scenario.failure_mode == "domain-shift":
        checks.append(
            make_check(
                "domain_shift",
                "Held-out distribution",
                "WARN",
                "Held-out rows are shifted away from training rows.",
            )
        )
    if scenario.failure_mode == "noisy-observations":
        checks.append(
            make_check(
                "noise_stress",
                "Noise stress",
                "WARN",
                "Observation noise is increased for this UI demo.",
            )
        )

    counts = Counter(check["status"] for check in checks)
    overall = "FAIL" if counts["FAIL"] else "WARN" if counts["WARN"] else "PASS"
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": f"{run_id}-validation",
        "overall_status": overall,
        "checks": checks,
        "summary": {
            "total": len(checks),
            "pass": counts["PASS"],
            "warn": counts["WARN"],
            "fail": counts["FAIL"],
        },
        "is_synthetic": True,
        "scientific_warning": "PASS checks fixture consistency only; they do not validate scientific accuracy.",
    }


def generate_experiment_history(
    rng: random.Random,
    scenario: DemoScenario,
    *,
    count: int,
    sample_count: int,
    observation_count: int,
    seed: int,
    run_id: str,
) -> dict[str, Any]:
    """Create fictitious jobs, status transitions, warning states, and metrics."""
    phases = (
        "queued",
        "preparing",
        "simulating",
        "extracting-features",
        "fitting-demo",
        "validating",
        "complete",
    )
    labels = (
        "baseline-a",
        "noise-ablation",
        "mask-stress",
        "heldout-check",
        "parameter-sweep",
        "seed-repeat",
        "coverage-pass",
        "catalog-refresh",
    )
    runs: list[dict[str, Any]] = []
    failure_index = int(count * 0.68) if scenario.failure_mode else -1
    for index in range(count):
        is_failure = index == failure_index
        runtime = max(2, int(round(8 + rng.random() * 72 * scenario.observation_noise)))
        status = (
            "warning"
            if is_failure and scenario.failure_mode != "small-sample"
            else "running-demo"
            if index == count - 1
            else "complete-demo"
        )
        score = clamp(
            0.9
            - scenario.observation_noise * 0.07
            - abs(normal(rng)) * 0.05
            - (0.16 if is_failure else 0),
            0.1,
            0.99,
        )
        warnings = [scenario.failure_mode] if is_failure else []
        if scenario.domain_shift and index % 6 == 0:
            warnings.append("heldout-domain-shift-demo")
        runs.append(
            {
                "run_id": f"{run_id}-history-{index + 1:04d}",
                "label": f"{labels[index % len(labels)]}-{index + 1:03d}",
                "scenario": scenario.scenario_id,
                "status": status,
                "phase": rng.choice(phases),
                "started_at_offset_minutes": (count - index) * rng.randint(7, 28),
                "duration_seconds_toy": runtime,
                "seed": (seed + index * 997) & 0xFFFFFFFF,
                "simulations": max(20, int(sample_count * uniform(rng, 0.45, 1.35))),
                "observations": max(20, int(observation_count * uniform(rng, 0.4, 1.3))),
                "toy_score": rounded(score, 5),
                "warnings": warnings,
                "is_synthetic": True,
            }
        )
    runs.sort(key=lambda row: row["started_at_offset_minutes"], reverse=True)
    return {
        "schema_version": SCHEMA_VERSION,
        "history_kind": "synthetic-run-history",
        "run_count": len(runs),
        "runs": runs,
        "is_synthetic": True,
        "scientific_warning": "History records are fabricated for a timeline UI; they were not executed by the research pipeline.",
    }


def generate_ablation_report(
    rng: random.Random, scenario: DemoScenario, *, sample_count: int
) -> dict[str, Any]:
    """Fabricate ablation records so comparison tables have varied states."""
    components = (
        "field-summary",
        "power-features",
        "angular-catalog",
        "cmb-summary",
        "lensing-proxy",
        "late-fusion",
        "all-inputs",
    )
    results: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        enabled = 5 if component == "all-inputs" else index + 1
        score = clamp(
            0.58 + index * 0.035 - scenario.observation_noise * 0.025 + normal(rng, sigma=0.014),
            0.35,
            0.94,
        )
        results.append(
            {
                "component": component,
                "inputs_enabled": enabled,
                "toy_score": rounded(score, 6),
                "toy_score_error": rounded(0.015 + rng.random() * 0.025, 6),
                "sample_count": max(20, int(sample_count * uniform(rng, 0.72, 0.97))),
                "conclusion": "Synthetic comparison only; does not establish predictive utility.",
                "is_synthetic": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "ablation_kind": "fabricated UI comparison",
        "results": results,
        "is_synthetic": True,
        "scientific_warning": SCIENCE_WARNING,
    }


def generate_data_quality_report(
    scenario: DemoScenario, survey_catalog: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    """Compute meaningful descriptive counts of synthetic rows, not science."""
    rows = survey_catalog["rows"]
    masked_count = sum(row["mask_flag"] != "unmasked-demo" for row in rows)
    completeness_values = [float(row["completeness_fraction_toy"]) for row in rows]
    mean_completeness = (
        sum(completeness_values) / len(completeness_values) if completeness_values else 0.0
    )
    masked_fraction = masked_count / max(1, len(rows))
    metrics = [
        {
            "key": "row-count",
            "label": "Catalog rows",
            "value": len(rows),
            "unit": "rows",
            "status": "WARN" if len(rows) < 50 else "PASS",
        },
        {
            "key": "missingness",
            "label": "Masked fraction",
            "value": rounded(masked_fraction, 6),
            "unit": "fraction",
            "status": "WARN" if masked_fraction > 0.12 else "PASS",
        },
        {
            "key": "completeness",
            "label": "Mean completeness",
            "value": rounded(mean_completeness, 6),
            "unit": "fraction",
            "status": "WARN" if mean_completeness < 0.72 else "PASS",
        },
        {
            "key": "noise",
            "label": "Configured noise multiplier",
            "value": scenario.observation_noise,
            "unit": "relative toy scale",
            "status": "WARN" if scenario.observation_noise > 1.5 else "PASS",
        },
        {
            "key": "seed",
            "label": "Reproducibility seed",
            "value": seed,
            "unit": "integer",
            "status": "PASS",
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "synthetic quality summary",
        "metrics": metrics,
        "generated_checks": len(metrics),
        "is_synthetic": True,
        "scientific_warning": "These metrics describe generated demo rows only; they say nothing about any real survey.",
    }


def generate_field_summary(
    rng: random.Random, scenario: DemoScenario, *, count: int = 80
) -> dict[str, Any]:
    """Create a one-dimensional density/velocity pattern for charts and sparklines."""
    slices: list[dict[str, Any]] = []
    for index in range(count):
        fraction = index / max(1, count - 1)
        wave = math.sin((fraction * 4.8 + 0.1) * math.pi) * 0.42
        wave += math.sin((fraction * 11 + 0.17) * math.pi) * 0.17
        envelope = 0.55 + 0.45 * math.cos((fraction - 0.5) * math.pi)
        density = wave * envelope + normal(rng, sigma=0.06 * scenario.observation_noise)
        velocity = math.cos(fraction * 5.7 * math.pi) * 0.35 + normal(rng, sigma=0.045)
        slices.append(
            {
                "index": index,
                "x_fraction": rounded(fraction, 6),
                "density_contrast_toy": rounded(density, 7),
                "velocity_proxy_toy": rounded(velocity, 7),
                "is_synthetic": True,
            }
        )
    grid_size = (
        128
        if scenario.scenario_id == "high_resolution"
        else 16
        if scenario.scenario_id == "sparse_data"
        else 64
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "field_kind": "one-dimensional illustration slice",
        "slices": slices,
        "grid_shape": [grid_size, grid_size, grid_size],
        "values_stored": False,
        "is_synthetic": True,
        "scientific_warning": "A summary display pattern, not a volumetric simulation or N-body output.",
    }


def generate_import_error_examples() -> dict[str, Any]:
    """Return descriptions that drive importer error-state walkthroughs."""
    return {
        "schema_version": SCHEMA_VERSION,
        "is_synthetic": True,
        "examples": [
            {
                "filename": "example-malformed.json",
                "raw_text": '{"dataset_name": "truncated",',
                "expected_result": "parse-error",
                "description": "Malformed JSON text; for importer tests only.",
            },
            {
                "filename": "example-unknown-shape.json",
                "content": {"hello": "world", "rows": [1, 2, 3]},
                "expected_result": "unclassified-extra",
                "description": "Valid JSON without a recognized artifact signature.",
            },
            {
                "filename": "example-array-root.json",
                "content": [{"name": "array-root"}],
                "expected_result": "shape-error",
                "description": "Top-level array to exercise importer validation.",
            },
            {
                "filename": "example-deep-nesting.json",
                "content": {"a": {"b": {"c": {"d": "..."}}}},
                "expected_result": "depth-test",
                "description": "Small proxy for a deeply nested JSON test.",
            },
        ],
        "scientific_warning": "Import test cases are fixtures, not research data.",
    }


def build_dataset_metadata(
    scenario: DemoScenario,
    *,
    seed: int,
    sample_count: int,
    observation_count: int,
    grid_size: int,
    run_id: str,
) -> dict[str, Any]:
    """Create metadata accepted by the current frontend's JSON importer."""
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": f"NUFIELD synthetic demo — {scenario.label}",
        "dataset_id": f"synthetic-{run_id}",
        "dataset_tag": "MOCK_DEMO_NOT_FOR_SCIENCE",
        "physics_level": "toy",
        "is_synthetic": True,
        "source_kind": SOURCE_KIND,
        "scenario": scenario.scenario_id,
        "random_seed": seed,
        "n_simulations": sample_count,
        "n_observations": observation_count,
        "grid_size": grid_size,
        "feature_names": [
            "field_mean",
            "field_variance",
            "power_slope",
            "density_proxy",
            "noise_level",
            "mask_fraction",
        ],
        "parameter_names": list(PARAMETER_KEYS),
        "sampling": split_counts(sample_count),
        "summary_statistics": {
            "expected_parameter_centers": scenario.resolved_center(),
            "parameter_count": len(PARAMETER_KEYS),
        },
        "scientific_warning": SCIENCE_WARNING,
        "caveats": [
            "Rows are generated from display-oriented toy distributions, not a Boltzmann solver.",
            "Parameter associations are configured by the demo generator, not inferred.",
            "No measurement, prior, likelihood, posterior, or physical constraint is validated.",
        ],
    }


def build_manifest(
    dataset: dict[str, Any],
    *,
    seed: int,
    scenario: DemoScenario,
    run_id: str,
    options: dict[str, Any],
    artifact_names: list[str],
) -> dict[str, Any]:
    """Create the machine-readable provenance sidecar for all exported files."""
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "synthetic-demo-manifest",
        "dataset_id": dataset["dataset_id"],
        "dataset_name": dataset["dataset_name"],
        "physics_level": "toy",
        "is_synthetic": True,
        "source_kind": SOURCE_KIND,
        "generator_version": GENERATOR_VERSION,
        "random_seed": seed,
        "scenario": scenario.scenario_id,
        "requested_sizes": options,
        "artifact_names": artifact_names,
        "generated_from_real_observations": False,
        "generated_from_cosmology_solver": False,
        "posterior_is_scientific": False,
        "deterministic_rows": True,
        "deterministic_exclusion": "Creation timestamp metadata is generated separately by the output writer.",
        "scientific_warning": SCIENCE_WARNING,
    }


def build_provenance(
    scenario: DemoScenario, *, seed: int, options: dict[str, Any], run_id: str
) -> dict[str, Any]:
    """Create a stable top-level provenance record for audit and demo screenshots."""
    return {
        "provenance_id": f"provenance-{run_id}",
        "source_kind": SOURCE_KIND,
        "generator_version": GENERATOR_VERSION,
        "code_path": "src/nufield/demo_data/",
        "generated_from_real_observations": False,
        "generated_from_cosmology_solver": False,
        "posterior_is_scientific": False,
        "calibrated": False,
        "physical_units_verified": False,
        "scenario": scenario.scenario_id,
        "seed": seed,
        "parameters": options,
        "warning": SCIENCE_WARNING,
    }


def make_check(check_id: str, label: str, status: str, detail: str) -> dict[str, Any]:
    """Construct a single fixture-validation check with transparent semantics."""
    return {
        "id": check_id,
        "label": label,
        "status": status,
        "detail": detail,
        "fixture_contract_check": True,
        "is_synthetic": True,
    }
