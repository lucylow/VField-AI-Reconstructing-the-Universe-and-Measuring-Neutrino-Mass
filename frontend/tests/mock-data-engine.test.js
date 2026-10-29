import test from "node:test";
import assert from "node:assert/strict";
import {
  MOCK_ENGINE_VERSION,
  MOCK_PARAMETER_DEFINITIONS,
  MOCK_SCENARIOS,
  MOCK_SCIENTIFIC_WARNING,
  generateMockBundle,
  listMockArtifacts,
  normalizeMockOptions,
  summarizeMockBundle,
  toCSV,
  toPrettyJSON,
} from "../src/data/mock-data-engine.js";

const REQUIRED_ARTIFACTS = [
  "dataset.json", "dataset.manifest.json", "benchmark_report.json", "parameter_samples.json",
  "posterior_chains.json", "power_spectrum.json", "expansion_history.json", "survey_catalog.json",
  "cmb_summary.json", "lensing_catalog.json", "validation_report.json", "experiment_history.json",
  "ablation_report.json", "data_quality_report.json", "field_summary.json",
];

test("scenario catalog contains stable IDs and descriptions", () => {
  assert.ok(Object.keys(MOCK_SCENARIOS).length >= 10);
  for (const [id, scenario] of Object.entries(MOCK_SCENARIOS)) {
    assert.equal(scenario.id, id);
    assert.ok(scenario.label.length > 2);
    assert.ok(scenario.description.length > 8);
    assert.ok(typeof scenario.observationNoise === "number");
  }
});

test("option normalization clamps every browser-controlled size", () => {
  const options = normalizeMockOptions({ scenario: "does-not-exist", seed: -1, sampleCount: 999999, observationCount: 0, chainCount: 500, samplesPerChain: 1, spectrumPoints: 5000, historyCount: -20 });
  assert.equal(options.scenario, "baseline");
  assert.equal(options.seed, 4294967295);
  assert.equal(options.sampleCount, 2000);
  assert.equal(options.observationCount, 20);
  assert.equal(options.chainCount, 8);
  assert.equal(options.samplesPerChain, 50);
  assert.equal(options.spectrumPoints, 800);
  assert.equal(options.historyCount, 8);
});

test("same seed and options yield identical scientific sample rows", () => {
  const options = { scenario: "baseline", seed: 123456, sampleCount: 80, observationCount: 40, chainCount: 2, samplesPerChain: 60, spectrumPoints: 60, historyCount: 8 };
  const a = generateMockBundle(options);
  const b = generateMockBundle(options);
  assert.deepEqual(a.parameter_samples.rows, b.parameter_samples.rows);
  assert.deepEqual(a.power_spectrum.points, b.power_spectrum.points);
  assert.deepEqual(a.survey_catalog.rows, b.survey_catalog.rows);
  assert.deepEqual(a.lensing_catalog.rows, b.lensing_catalog.rows);
  assert.deepEqual(a.experiment_history.runs, b.experiment_history.runs);
});

test("each key artifact explicitly labels synthetic provenance", () => {
  const bundle = generateMockBundle({ sampleCount: 30, observationCount: 25, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  assert.equal(bundle.dataset.is_synthetic, true);
  assert.equal(bundle.manifest.is_synthetic, true);
  assert.equal(bundle.benchmark_report.is_synthetic, true);
  assert.match(bundle.dataset.scientific_warning, /SYNTHETIC DEMO DATA ONLY/);
  assert.equal(bundle.provenance.generated_from_real_observations, false);
  assert.equal(bundle.provenance.generated_from_cosmology_solver, false);
  assert.equal(bundle.provenance.posterior_is_scientific, false);
});

test("dataset artifact matches the frontend importer signature", () => {
  const bundle = generateMockBundle({ sampleCount: 40, observationCount: 30, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  assert.ok(Array.isArray(bundle.dataset.feature_names));
  assert.ok(Array.isArray(bundle.dataset.parameter_names));
  assert.equal(bundle.dataset.physics_level, "toy");
  assert.equal(bundle.manifest.schema_version, "1.0.0");
  assert.equal(bundle.manifest.dataset_id, bundle.dataset.dataset_id);
  assert.ok(Array.isArray(bundle.benchmark_report.parameter_points));
  assert.ok(bundle.benchmark_report.parameter_points.length > 0);
  assert.ok(bundle.benchmark_report.parameter_points.length <= 600);
});

test("parameter samples are finite and stay inside display ranges", () => {
  const bundle = generateMockBundle({ sampleCount: 650, observationCount: 25, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30, seed: 810 });
  for (const row of bundle.parameter_samples.rows) {
    for (const [key, definition] of Object.entries(MOCK_PARAMETER_DEFINITIONS)) {
      assert.ok(Number.isFinite(row[key]), `${key} should be finite`);
      assert.ok(row[key] >= definition.min && row[key] <= definition.max, `${key} should be in display bounds`);
    }
    assert.equal(row.is_synthetic, true);
    assert.ok(["train", "validation", "test"].includes(row.split));
  }
});

test("train validation test proportions sum to sample count", () => {
  for (const sampleCount of [20, 21, 80, 501, 1999]) {
    const bundle = generateMockBundle({ sampleCount, observationCount: 20, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
    const counts = Object.groupBy ? Object.groupBy(bundle.parameter_samples.rows, (row) => row.split) : bundle.parameter_samples.rows.reduce((acc, row) => { (acc[row.split] ??= []).push(row); return acc; }, {});
    const total = Object.values(counts).reduce((sum, rows) => sum + rows.length, 0);
    assert.equal(total, sampleCount);
    assert.equal(bundle.benchmark_report.n_train + bundle.benchmark_report.n_validation_held_out + bundle.benchmark_report.n_test, sampleCount);
  }
});

test("default and edge-case scenarios exercise expected validation statuses", () => {
  const small = { sampleCount: 60, observationCount: 30, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30, seed: 987 };
  const baseline = generateMockBundle({ ...small, scenario: "baseline" });
  assert.equal(baseline.benchmark_report.split_disjoint, true);
  assert.ok(["PASS", "WARN"].includes(baseline.validation_report.overall_status));
  const leakage = generateMockBundle({ ...small, scenario: "split_leakage" });
  assert.equal(leakage.benchmark_report.split_disjoint, false);
  assert.equal(leakage.validation_report.overall_status, "FAIL");
  assert.ok(leakage.validation_report.checks.some((check) => check.id === "split_disjointness" && check.status === "FAIL"));
});

test("coverage failure profile generates under-coverage stress values", () => {
  const bundle = generateMockBundle({ scenario: "coverage_failure", seed: 22, sampleCount: 100, observationCount: 35, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  assert.ok(bundle.benchmark_report.coverage_diagnostic.some((item) => item.observed < item.nominal - 0.06));
  assert.ok(["WARN", "FAIL"].includes(bundle.validation_report.overall_status));
});

test("domain-shift scenario reports shifted held-out stress metadata", () => {
  const bundle = generateMockBundle({ scenario: "domain_shift", seed: 1001, sampleCount: 100, observationCount: 35, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  assert.equal(bundle.scenario.failure_mode, "domain-shift");
  assert.ok(bundle.validation_report.checks.some((check) => check.id === "domain_shift" && check.status === "WARN"));
  assert.ok(bundle.experiment_history.runs.some((run) => run.warnings.includes("heldout-domain-shift-demo")));
});

test("all generated numeric arrays contain finite values", () => {
  const bundle = generateMockBundle({ seed: 765, sampleCount: 80, observationCount: 55, samplesPerChain: 75, historyCount: 10, spectrumPoints: 55 });
  const numericValues = [
    ...bundle.power_spectrum.points.flatMap((row) => [row.k, row.p_toy, row.p_reference_toy, row.relative_noise]),
    ...bundle.expansion_history.points.flatMap((row) => [row.redshift, row.scale_factor, row.toy_distance_index]),
    ...bundle.cmb_summary.points.flatMap((row) => [row.ell, row.cl_toy, row.uncertainty_toy]),
    ...bundle.survey_catalog.rows.flatMap((row) => [row.ra_deg, row.dec_deg, row.redshift_toy, row.density_proxy_observed_toy, row.uncertainty_toy]),
    ...bundle.lensing_catalog.rows.flatMap((row) => [row.source_beta_arcsec_toy, row.theta_e_arcsec_toy, row.image_plus_arcsec_toy, row.image_minus_arcsec_toy]),
  ];
  assert.ok(numericValues.every(Number.isFinite));
});

test("scale factor grid follows the redshift definition", () => {
  const bundle = generateMockBundle({ sampleCount: 20, observationCount: 20, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  for (const point of bundle.expansion_history.points) {
    assert.ok(Math.abs(point.scale_factor - 1 / (1 + point.redshift)) < 1e-6);
  }
});

test("posterior chains are bounded pseudo-chain records, not advertised as inference", () => {
  const bundle = generateMockBundle({ seed: 9, sampleCount: 25, observationCount: 20, chainCount: 3, samplesPerChain: 53, historyCount: 8, spectrumPoints: 30 });
  assert.equal(bundle.posterior_chains.chains.length, 3);
  assert.ok(bundle.posterior_chains.chains.every((chain) => chain.samples.length === 53));
  assert.match(bundle.posterior_chains.warning, /not MCMC posterior samples/i);
  assert.ok(bundle.posterior_chains.chains.flatMap((chain) => chain.samples).every((sample) => sample.is_synthetic === true));
});

test("catalog rows have stable shape and generated IDs", () => {
  const bundle = generateMockBundle({ seed: 18, sampleCount: 20, observationCount: 71, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  assert.equal(bundle.survey_catalog.rows.length, 71);
  assert.equal(new Set(bundle.survey_catalog.rows.map((row) => row.object_id)).size, 71);
  assert.ok(bundle.survey_catalog.rows.every((row) => row.ra_deg >= 0 && row.ra_deg <= 360 && row.dec_deg >= -90 && row.dec_deg <= 90));
  assert.ok(bundle.survey_catalog.rows.every((row) => row.is_synthetic));
});

test("lensing fixtures satisfy point-mass image relation within display noise", () => {
  const bundle = generateMockBundle({ seed: 90, sampleCount: 20, observationCount: 80, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  assert.ok(bundle.lensing_catalog.rows.length > 0);
  assert.ok(bundle.lensing_catalog.rows.every((row) => Number.isFinite(row.image_plus_arcsec_toy) && Number.isFinite(row.image_minus_arcsec_toy)));
  assert.match(bundle.lensing_catalog.scientific_warning, /not estimates of real lens masses/i);
});

test("experiment history is sufficiently rich for table and timeline demos", () => {
  const bundle = generateMockBundle({ seed: 191, sampleCount: 20, observationCount: 20, samplesPerChain: 50, historyCount: 37, spectrumPoints: 30 });
  assert.equal(bundle.experiment_history.runs.length, 37);
  assert.ok(bundle.experiment_history.runs.some((run) => run.status === "running-demo"));
  assert.ok(bundle.experiment_history.runs.every((run) => Number.isFinite(run.toy_score) && run.is_synthetic));
});

test("quality report responds to generated survey volume and marks toy units", () => {
  const bundle = generateMockBundle({ seed: 18, scenario: "noisy_survey", sampleCount: 20, observationCount: 87, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  const rowCountMetric = bundle.data_quality_report.metrics.find((metric) => metric.key === "row-count");
  const noiseMetric = bundle.data_quality_report.metrics.find((metric) => metric.key === "noise");
  assert.equal(rowCountMetric.value, 87);
  assert.equal(noiseMetric.value, MOCK_SCENARIOS.noisy_survey.observationNoise);
  assert.ok(noiseMetric.status === "WARN");
});

test("artifact registry includes useful per-file exports", () => {
  const bundle = generateMockBundle({ sampleCount: 20, observationCount: 20, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  const artifacts = listMockArtifacts(bundle);
  assert.deepEqual(artifacts.map((artifact) => artifact.filename).slice(0, REQUIRED_ARTIFACTS.length), REQUIRED_ARTIFACTS);
  assert.ok(artifacts.every((artifact) => artifact.content && artifact.bytes > 0));
});

test("malformed-import profile includes valid raw test descriptions", () => {
  const bundle = generateMockBundle({ scenario: "malformed_artifacts", sampleCount: 20, observationCount: 20, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  const examples = bundle.import_error_examples.examples;
  assert.equal(examples.length, 4);
  assert.equal(examples[0].expected_result, "parse-error");
  assert.equal(examples[1].expected_result, "unclassified-extra");
  assert.ok(bundle.import_error_examples.is_synthetic);
});

test("summary counts artifacts, observations, posterior rows, and warnings", () => {
  const bundle = generateMockBundle({ sampleCount: 40, observationCount: 21, chainCount: 2, samplesPerChain: 53, historyCount: 9, spectrumPoints: 33 });
  const summary = summarizeMockBundle(bundle);
  assert.equal(summary.parameterRows, 40);
  assert.equal(summary.observations, 21);
  assert.equal(summary.posteriorRows, 106);
  assert.equal(summary.spectrumPoints, 33);
  assert.equal(summary.experiments, 9);
  assert.ok(summary.artifactCount >= 15);
  assert.equal(summary.allSynthetic, true);
});

test("CSV serialization quotes commas, quotes, line breaks, and nested objects", () => {
  const csv = toCSV([{ id: "a,b", text: 'say "hi"', value: 3, nested: { x: 1 }, blank: null }, { id: "simple", text: "plain", value: 7, nested: [1, 2], blank: undefined }]);
  assert.match(csv, /"a,b"/);
  assert.match(csv, /"say ""hi"""/);
  assert.match(csv, /"\{""x"":1\}"/);
  assert.ok(csv.endsWith("\r\n"));
});

test("JSON exports are parseable and retain the synthetic warning", () => {
  const bundle = generateMockBundle({ sampleCount: 20, observationCount: 20, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  const parsed = JSON.parse(toPrettyJSON(bundle));
  assert.equal(parsed.bundle_kind, "nufield-synthetic-demo-bundle");
  assert.match(parsed.dataset.scientific_warning, /SYNTHETIC DEMO DATA ONLY/);
});

test("scenario-specific parameters differ without pretending to infer them", () => {
  const options = { seed: 123, sampleCount: 100, observationCount: 20, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 };
  const low = generateMockBundle({ ...options, scenario: "low_neutrino" });
  const high = generateMockBundle({ ...options, scenario: "high_neutrino" });
  const lowMean = low.parameter_samples.rows.reduce((sum, row) => sum + row.sum_mnu_ev, 0) / low.parameter_samples.rows.length;
  const highMean = high.parameter_samples.rows.reduce((sum, row) => sum + row.sum_mnu_ev, 0) / high.parameter_samples.rows.length;
  assert.ok(highMean > lowMean + 0.1);
});

test("scientific warning is prominent and engine version is present", () => {
  assert.match(MOCK_SCIENTIFIC_WARNING, /not observations/i);
  assert.match(MOCK_SCIENTIFIC_WARNING, /not a calibrated cosmological prediction/i);
  const bundle = generateMockBundle({ sampleCount: 20, observationCount: 20, samplesPerChain: 50, historyCount: 8, spectrumPoints: 30 });
  assert.equal(bundle.manifest.generator_version, MOCK_ENGINE_VERSION);
});
