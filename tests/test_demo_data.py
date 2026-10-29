"""Property and integration tests for synthetic demo data only.

These tests verify packaging, reproducibility, metadata contracts, and deliberate
UI stress cases. They do not validate the project's cosmology or inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import pytest

from nufield.demo_data.bundle import (
    DemoOptions,
    build_demo_bundle,
    list_artifacts,
    normalize_options,
    summarize_bundle,
    write_demo_bundle,
)
from nufield.demo_data.generators import (
    generate_expansion_history,
    generate_parameter_rows,
    generate_posterior_chains,
    generate_power_spectrum,
    generate_survey_catalog,
)
from nufield.demo_data.profiles import (
    DEMO_SCENARIOS,
    PARAMETER_DEFINITIONS,
    SCIENCE_WARNING,
    get_scenario,
    scenario_catalog,
)
from nufield.demo_data.randoms import (
    clamp,
    descriptive_stats,
    make_rng,
    quantile,
    split_counts,
    stable_digest,
)


def small_options(**overrides):
    options = {
        "scenario": "baseline",
        "seed": 7123,
        "sample_count": 60,
        "observation_count": 43,
        "chain_count": 2,
        "samples_per_chain": 55,
        "spectrum_points": 40,
        "history_count": 10,
        "field_slices": 25,
    }
    options.update(overrides)
    return options


def test_scenario_catalog_is_complete_and_labels_synthetic_profiles():
    assert len(DEMO_SCENARIOS) >= 10
    assert len(scenario_catalog()) == len(DEMO_SCENARIOS)
    assert "baseline" in DEMO_SCENARIOS
    for scenario_id, scenario in DEMO_SCENARIOS.items():
        assert scenario.scenario_id == scenario_id
        assert scenario.label
        assert scenario.description
        assert scenario.to_dict()["is_synthetic_profile"] is True
        assert scenario.to_dict()["scientific_warning"] == SCIENCE_WARNING


def test_unknown_profile_falls_back_to_baseline():
    assert get_scenario("missing-profile").scenario_id == "baseline"
    assert get_scenario(None).scenario_id == "baseline"


def test_options_normalize_and_cap_all_sizes():
    options = normalize_options(
        {
            "scenario": "bad-name",
            "seed": -5,
            "sample_count": 2_000_000,
            "observation_count": 0,
            "chain_count": 100,
            "samples_per_chain": 2,
            "spectrum_points": 10_000,
            "history_count": -1,
            "field_slices": 1,
        }
    )
    assert options.scenario == "baseline"
    assert options.seed == 0xFFFFFFFB
    assert options.sample_count == 100_000
    assert options.observation_count == 20
    assert options.chain_count == 8
    assert options.samples_per_chain == 50
    assert options.spectrum_points == 2_000
    assert options.history_count == 8
    assert options.field_slices == 20


def test_options_accept_dataclass_inputs():
    original = DemoOptions(scenario="baseline", seed=11, sample_count=40, observation_count=40)
    normalized = normalize_options(original)
    assert normalized.seed == 11
    assert normalized.sample_count == 40


def test_normalization_does_not_mutate_the_callers_dictionary():
    original = small_options()
    snapshot = dict(original)
    normalize_options(original)
    assert original == snapshot


def test_split_counts_do_not_drop_rows():
    for total in [0, 1, 2, 19, 20, 21, 100, 999]:
        sizes = split_counts(total)
        assert sum(sizes.values()) == total
        assert min(sizes.values()) >= 0


def test_split_counts_reject_bad_fractions():
    with pytest.raises(ValueError):
        split_counts(10, 0.8, 0.3)
    with pytest.raises(ValueError):
        split_counts(10, -0.1, 0.2)


def test_random_stream_is_local_and_repeatable():
    first = make_rng(123)
    second = make_rng(123)
    assert [first.random() for _ in range(20)] == [second.random() for _ in range(20)]
    assert make_rng(1).random() != make_rng(2).random()


def test_random_stream_rejects_bool_and_non_integer_seeds():
    with pytest.raises(TypeError):
        make_rng(True)
    with pytest.raises(TypeError):
        make_rng(1.2)  # type: ignore[arg-type]


def test_clamp_rejects_invalid_ranges_and_non_finite_values():
    assert clamp(2, 0, 1) == 1
    assert clamp(-2, 0, 1) == 0
    with pytest.raises(ValueError):
        clamp(1, 2, 0)
    with pytest.raises(ValueError):
        clamp(float("nan"), 0, 1)


def test_quantile_interpolates_and_handles_edges():
    assert quantile([0, 10], 0) == 0
    assert quantile([0, 10], 0.5) == 5
    assert quantile([0, 10], 1) == 10
    assert math.isnan(quantile([], 0.5))
    with pytest.raises(ValueError):
        quantile([0, 1], 1.1)


def test_descriptive_stats_empty_and_nonempty_inputs():
    empty = descriptive_stats([])
    assert empty["count"] == 0
    assert empty["mean"] is None
    stats = descriptive_stats([1, 2, 3, 4, float("nan")])
    assert stats["count"] == 4
    assert stats["mean"] == 2.5
    assert stats["median"] == 2.5
    assert stats["min"] == 1
    assert stats["max"] == 4


def test_stable_digest_is_repeatable_and_short():
    assert stable_digest("nufield") == stable_digest("nufield")
    assert len(stable_digest("nufield")) == 16
    assert stable_digest("nufield") != stable_digest("other")


def test_bundle_artifacts_have_required_top_level_provenance():
    bundle = build_demo_bundle(small_options())
    assert bundle["bundle_kind"] == "nufield-synthetic-demo-bundle"
    assert bundle["dataset"]["is_synthetic"] is True
    assert bundle["dataset"]["physics_level"] == "toy"
    assert bundle["manifest"]["generated_from_real_observations"] is False
    assert bundle["provenance"]["generated_from_cosmology_solver"] is False
    assert bundle["provenance"]["posterior_is_scientific"] is False
    assert SCIENCE_WARNING in bundle["dataset"]["scientific_warning"]


def test_bundle_is_deterministic_at_data_row_level():
    first = build_demo_bundle(small_options())
    second = build_demo_bundle(small_options())
    keys = [
        "parameter_samples",
        "posterior_chains",
        "power_spectrum",
        "survey_catalog",
        "cmb_summary",
        "lensing_catalog",
        "experiment_history",
    ]
    for key in keys:
        assert first[key] == second[key], key
    assert first["run_id"] == second["run_id"]
    # created_at is deliberately wall-clock metadata and is not used for reproducibility checks.


def test_parameter_rows_are_bounded_and_split_complete():
    bundle = build_demo_bundle(small_options(sample_count=237))
    rows = bundle["parameter_samples"]["rows"]
    assert len(rows) == 237
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    for row in rows:
        assert row["is_synthetic"] is True
        for key, definition in PARAMETER_DEFINITIONS.items():
            assert math.isfinite(row[key])
            assert definition["min"] <= row[key] <= definition["max"]


def test_parameter_row_ids_are_unique_except_intentional_realization_overlap():
    bundle = build_demo_bundle(small_options(sample_count=111))
    rows = bundle["parameter_samples"]["rows"]
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["realization_id"] for row in rows}) == len(rows)
    leakage = build_demo_bundle(small_options(scenario="split_leakage", sample_count=111))
    leak_rows = leakage["parameter_samples"]["rows"]
    assert len({row["id"] for row in leak_rows}) == len(leak_rows)
    assert len({row["realization_id"] for row in leak_rows}) == len(leak_rows) - 1


def test_posterior_trace_rows_are_marked_toy_and_respect_requested_size():
    bundle = build_demo_bundle(small_options(chain_count=3, samples_per_chain=72))
    chains = bundle["posterior_chains"]["chains"]
    assert len(chains) == 3
    assert all(len(chain["samples"]) == 72 for chain in chains)
    assert all(sample["is_synthetic"] for chain in chains for sample in chain["samples"])
    assert "not MCMC posterior samples" in bundle["posterior_chains"]["warning"]


@pytest.mark.parametrize("scenario", list(DEMO_SCENARIOS))
def test_all_scenarios_generate_parseable_bundle(scenario):
    bundle = build_demo_bundle(small_options(scenario=scenario))
    encoded = json.dumps(bundle, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["scenario"]["scenario_id"] == scenario
    assert decoded["dataset"]["is_synthetic"] is True
    assert decoded["summary"]["parameter_rows"] == 60
    assert decoded["summary"]["all_synthetic"] is True


def test_split_leakage_profile_fails_only_the_intended_split_contract():
    bundle = build_demo_bundle(small_options(scenario="split_leakage"))
    assert bundle["benchmark_report"]["split_disjoint"] is False
    assert bundle["validation_report"]["overall_status"] == "FAIL"
    check = next(
        item for item in bundle["validation_report"]["checks"] if item["id"] == "split_disjointness"
    )
    assert check["status"] == "FAIL"
    assert check["fixture_contract_check"] is True


def test_coverage_failure_has_artificially_low_coverage():
    bundle = build_demo_bundle(small_options(scenario="coverage_failure"))
    rows = bundle["benchmark_report"]["coverage_diagnostic"]
    assert any(row["observed"] < row["nominal"] - 0.06 for row in rows)
    assert bundle["validation_report"]["overall_status"] in {"WARN", "FAIL"}


def test_domain_shift_profile_adds_shift_warning():
    bundle = build_demo_bundle(small_options(scenario="domain_shift"))
    checks = bundle["validation_report"]["checks"]
    assert any(check["id"] == "domain_shift" and check["status"] == "WARN" for check in checks)
    assert any(
        "heldout-domain-shift-demo" in row["warnings"]
        for row in bundle["experiment_history"]["runs"]
    )


def test_calibration_profile_adds_calibration_warning():
    bundle = build_demo_bundle(small_options(scenario="calibration_bias"))
    assert any(
        check["id"] == "calibration_offset" for check in bundle["validation_report"]["checks"]
    )


def test_noisy_survey_quality_metric_marks_noise_stress():
    bundle = build_demo_bundle(small_options(scenario="noisy_survey"))
    metric = next(row for row in bundle["data_quality_report"]["metrics"] if row["key"] == "noise")
    assert metric["value"] > 1.5
    assert metric["status"] == "WARN"


def test_expansion_scale_factor_is_consistent_with_redshift():
    bundle = build_demo_bundle(small_options())
    for point in bundle["expansion_history"]["points"]:
        assert abs(point["scale_factor"] - 1 / (1 + point["redshift"])) < 1e-7


def test_power_spectrum_wavenumbers_are_monotonic_and_positive():
    bundle = build_demo_bundle(small_options(spectrum_points=100))
    points = bundle["power_spectrum"]["points"]
    assert len(points) == 100
    assert all(row["k"] > 0 and row["p_toy"] > 0 for row in points)
    assert all(left["k"] < right["k"] for left, right in zip(points, points[1:]))
    assert bundle["power_spectrum"]["is_synthetic"] is True


def test_catalog_coordinates_are_in_display_bounds():
    bundle = build_demo_bundle(small_options(observation_count=260))
    rows = bundle["survey_catalog"]["rows"]
    assert len(rows) == 260
    assert all(0 <= row["ra_deg"] <= 360 for row in rows)
    assert all(-90 <= row["dec_deg"] <= 90 for row in rows)
    assert all(row["is_synthetic"] is True for row in rows)


def test_lensing_catalog_has_schematic_model_disclaimer():
    bundle = build_demo_bundle(small_options(observation_count=120))
    lens = bundle["lensing_catalog"]
    assert len(lens["rows"]) >= 12
    assert "arbitrary angular units" in lens["lens_model"]
    assert "not real lens masses" in lens["scientific_warning"]


def test_ablation_report_has_seven_rows_and_nonphysical_conclusion():
    bundle = build_demo_bundle(small_options())
    results = bundle["ablation_report"]["results"]
    assert len(results) == 7
    assert all(row["is_synthetic"] for row in results)
    assert all("does not establish predictive utility" in row["conclusion"] for row in results)


def test_data_quality_counts_match_generated_rows():
    bundle = build_demo_bundle(small_options(observation_count=91))
    metrics = {metric["key"]: metric for metric in bundle["data_quality_report"]["metrics"]}
    assert metrics["row-count"]["value"] == 91
    assert 0 <= metrics["missingness"]["value"] <= 1
    assert 0 <= metrics["completeness"]["value"] <= 1


def test_malformed_profile_has_importer_test_descriptors():
    bundle = build_demo_bundle(small_options(scenario="malformed_artifacts"))
    examples = bundle["import_error_examples"]["examples"]
    assert len(examples) == 4
    assert examples[0]["expected_result"] == "parse-error"
    assert examples[1]["expected_result"] == "unclassified-extra"


def test_artifact_registry_is_stably_ordered_and_complete():
    bundle = build_demo_bundle(small_options())
    artifacts = list_artifacts(bundle)
    names = [name for name, _ in artifacts]
    assert names[:3] == ["dataset.json", "dataset.manifest.json", "benchmark_report.json"]
    assert len(names) == 15
    assert len(names) == len(set(names))


def test_bundle_summary_matches_data_rows():
    bundle = build_demo_bundle(
        small_options(
            sample_count=80,
            observation_count=61,
            chain_count=2,
            samples_per_chain=52,
            history_count=14,
        )
    )
    summary = summarize_bundle(bundle)
    assert summary["parameter_rows"] == 80
    assert summary["survey_rows"] == 61
    assert summary["posterior_trace_rows"] == 104
    assert summary["history_rows"] == 14
    assert summary["artifact_count"] == 15
    assert summary["all_synthetic"] is True


def test_write_bundle_creates_json_csv_checksums_and_zip(tmp_path: Path):
    output = tmp_path / "mock-output"
    result = write_demo_bundle(output, small_options(), include_csv=True, include_zip=True)
    assert Path(result["bundle_path"]).exists()
    assert Path(result["zip_path"]).exists()
    assert result["artifact_count"] == 15
    assert (output / "parameter_samples.csv").exists()
    assert (output / "checksums.json").exists()
    with (output / "bundle.json").open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    assert bundle["dataset"]["is_synthetic"] is True
    with (output / "checksums.json").open(encoding="utf-8") as handle:
        checksum_manifest = json.load(handle)
    assert checksum_manifest["algorithm"] == "SHA-256"
    assert checksum_manifest["files"]
    with zipfile.ZipFile(output / "nufield-demo-artifacts.zip") as archive:
        assert "bundle.json" in archive.namelist()
        assert "dataset.json" in archive.namelist()
        assert archive.testzip() is None


def test_written_csv_rows_match_bundle_rows(tmp_path: Path):
    output = tmp_path / "csv-test"
    write_demo_bundle(
        output, small_options(sample_count=44, observation_count=39), include_csv=True
    )
    with (output / "bundle.json").open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    with (output / "parameter_samples.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(bundle["parameter_samples"]["rows"]) == 44
    with (output / "survey_catalog.csv").open(encoding="utf-8", newline="") as handle:
        catalog_rows = list(csv.DictReader(handle))
    assert len(catalog_rows) == len(bundle["survey_catalog"]["rows"]) == 39


def test_write_bundle_refuses_existing_directory_when_requested(tmp_path: Path):
    output = tmp_path / "already-here"
    write_demo_bundle(output, small_options(), include_csv=False, include_zip=False)
    with pytest.raises(FileExistsError):
        write_demo_bundle(output, small_options(), include_csv=False, overwrite=False)


def test_write_bundle_can_omit_csv_and_zip(tmp_path: Path):
    output = tmp_path / "lightweight"
    result = write_demo_bundle(output, small_options(), include_csv=False, include_zip=False)
    assert result["zip_path"] is None
    assert not (output / "parameter_samples.csv").exists()
    assert (output / "bundle.json").exists()


def test_checksum_matches_exported_file(tmp_path: Path):
    output = tmp_path / "integrity"
    write_demo_bundle(output, small_options(), include_csv=False, include_zip=False)
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    expected = checksums["files"]["dataset.json"]
    actual = hashlib.sha256((output / "dataset.json").read_bytes()).hexdigest()
    assert actual == expected


def test_synthetic_bundle_json_has_no_nan_or_infinity():
    bundle = build_demo_bundle(small_options())
    json.dumps(bundle, allow_nan=False)


def test_seed_difference_changes_rows_but_not_schema():
    first = build_demo_bundle(small_options(seed=1))
    second = build_demo_bundle(small_options(seed=2))
    assert first["parameter_samples"]["rows"] != second["parameter_samples"]["rows"]
    assert first["dataset"]["feature_names"] == second["dataset"]["feature_names"]
    assert first["dataset"]["parameter_names"] == second["dataset"]["parameter_names"]


def test_extreme_valid_counts_are_supported_without_crossing_limits():
    normalized = normalize_options(
        {
            "sample_count": 100_000,
            "observation_count": 100_000,
            "chain_count": 8,
            "samples_per_chain": 10_000,
            "spectrum_points": 2_000,
            "history_count": 5_000,
            "field_slices": 2_000,
        }
    )
    assert normalized.sample_count == 100_000
    assert normalized.observation_count == 100_000
    assert normalized.chain_count == 8
    assert normalized.samples_per_chain == 10_000
    assert normalized.spectrum_points == 2_000
    assert normalized.history_count == 5_000
    assert normalized.field_slices == 2_000


def test_direct_generators_accept_private_random_streams():
    scenario = get_scenario("baseline")
    rows_a = generate_parameter_rows(make_rng(8), scenario, sample_count=30, seed=8, run_id="run-a")
    rows_b = generate_parameter_rows(make_rng(8), scenario, sample_count=30, seed=8, run_id="run-a")
    assert rows_a == rows_b
    spectrum_a = generate_power_spectrum(make_rng(3), scenario, count=35)
    spectrum_b = generate_power_spectrum(make_rng(3), scenario, count=35)
    assert spectrum_a == spectrum_b
    catalog = generate_survey_catalog(make_rng(9), scenario, count=20, run_id="test-run")
    assert len(catalog["rows"]) == 20
    chains = generate_posterior_chains(
        make_rng(10), scenario, chain_count=1, samples_per_chain=50, run_id="test-run"
    )
    assert len(chains["chains"][0]["samples"]) == 50
    expansion = generate_expansion_history(make_rng(11), scenario, count=50)
    assert len(expansion["points"]) == 50
