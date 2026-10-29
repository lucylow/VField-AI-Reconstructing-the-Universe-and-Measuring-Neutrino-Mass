/**
 * νField Mock Data Engine
 * ---------------------------------------------------------------------------
 * Deterministic, explicitly synthetic data for UI demos, screenshots, and
 * front-end workflow testing. Nothing returned from this module is an
 * observation, calibrated forecast, CLASS/CAMB output, or scientific result.
 *
 * Design goals:
 *   - Repeatable: same profile + seed + size gives the same scientific rows.
 *   - Bounded: browser-facing generation is capped to avoid accidental hangs.
 *   - Inspectable: every artifact carries synthetic provenance and caveats.
 *   - Composable: chart, table, diagnostics, and import-demo data share schemas.
 *   - Adversarial: profiles deliberately exercise warnings and failure states.
 */

export const MOCK_SCHEMA_VERSION = "1.0.0";
export const MOCK_ENGINE_VERSION = "0.4.0";
export const MOCK_SOURCE_KIND = "synthetic-demo-generator";
export const MOCK_SCIENTIFIC_WARNING =
  "SYNTHETIC DEMO DATA ONLY. Generated values are not observations, not a calibrated cosmological prediction, not a validated posterior, and not evidence about neutrino mass. Do not cite as scientific results.";

export const MOCK_PARAMETER_DEFINITIONS = Object.freeze({
  omega_m: Object.freeze({ label: "Ωm", name: "Matter density parameter", unit: "dimensionless", min: 0.18, max: 0.42, center: 0.30, sigma: 0.035 }),
  sigma8: Object.freeze({ label: "σ₈", name: "Fluctuation normalization", unit: "dimensionless", min: 0.65, max: 0.95, center: 0.81, sigma: 0.035 }),
  h: Object.freeze({ label: "h", name: "Scaled Hubble parameter", unit: "dimensionless", min: 0.58, max: 0.82, center: 0.68, sigma: 0.025 }),
  n_s: Object.freeze({ label: "nₛ", name: "Primordial spectral index", unit: "dimensionless", min: 0.88, max: 1.04, center: 0.965, sigma: 0.016 }),
  sum_mnu_ev: Object.freeze({ label: "Σmν", name: "Illustrative neutrino-mass sum", unit: "eV (toy input)", min: 0, max: 0.45, center: 0.12, sigma: 0.055 }),
  bias: Object.freeze({ label: "b", name: "Illustrative tracer bias", unit: "dimensionless", min: 0.65, max: 1.55, center: 1, sigma: 0.11 }),
});

const PARAMETER_KEYS = Object.freeze(Object.keys(MOCK_PARAMETER_DEFINITIONS));

/**
 * Demo profiles are UI scenarios, not competing physical cosmologies.
 * They are designed to demonstrate parameter shifts, noise sensitivity,
 * held-out evaluation, calibration problems, and validation failures.
 */
export const MOCK_SCENARIOS = Object.freeze({
  baseline: Object.freeze({
    id: "baseline",
    label: "Baseline walkthrough",
    group: "happy-path",
    description: "Balanced synthetic samples with a clean split and modest noise.",
    recommended: true,
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 1.0,
    observationNoise: 0.8,
    calibrationOffset: 0,
    domainShift: 0,
    coverageShift: 0,
    splitLeakage: false,
    failureMode: null,
    chartLabel: "Reference demo",
  }),
  low_neutrino: Object.freeze({
    id: "low_neutrino",
    label: "Low toy Σmν",
    group: "parameter-shift",
    description: "Moves the illustrative mass-sum input toward the lower edge of its demo range.",
    center: { omega_m: 0.30, sigma8: 0.815, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.035, bias: 1.0 },
    spread: 0.9, observationNoise: 0.8, calibrationOffset: 0, domainShift: 0, coverageShift: 0, splitLeakage: false, failureMode: null,
    chartLabel: "Low toy mass input",
  }),
  high_neutrino: Object.freeze({
    id: "high_neutrino",
    label: "High toy Σmν",
    group: "parameter-shift",
    description: "Moves the illustrative mass-sum input upward to exercise comparison charts.",
    center: { omega_m: 0.30, sigma8: 0.79, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.36, bias: 1.0 },
    spread: 0.95, observationNoise: 0.9, calibrationOffset: 0, domainShift: 0, coverageShift: 0, splitLeakage: false, failureMode: null,
    chartLabel: "High toy mass input",
  }),
  noisy_survey: Object.freeze({
    id: "noisy_survey",
    label: "Noisy survey",
    group: "observation-quality",
    description: "Adds heteroscedastic noise and lower data-quality flags to the synthetic catalog.",
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 1.25, observationNoise: 1.9, calibrationOffset: 0, domainShift: 0, coverageShift: -0.035, splitLeakage: false, failureMode: "noisy-observations",
    chartLabel: "Noise stress test",
  }),
  calibration_bias: Object.freeze({
    id: "calibration_bias",
    label: "Calibration offset",
    group: "observation-quality",
    description: "Adds a deliberate synthetic offset to mimic a mis-calibrated estimator.",
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 1.0, observationNoise: 0.9, calibrationOffset: 0.075, domainShift: 0, coverageShift: -0.055, splitLeakage: false, failureMode: "calibration-offset",
    chartLabel: "Calibration stress test",
  }),
  domain_shift: Object.freeze({
    id: "domain_shift",
    label: "Held-out domain shift",
    group: "validation-stress",
    description: "Keeps training rows near the reference center but shifts held-out rows and error metrics.",
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 1.0, observationNoise: 1.25, calibrationOffset: 0.015, domainShift: 0.06, coverageShift: -0.045, splitLeakage: false, failureMode: "domain-shift",
    chartLabel: "Domain-shift stress test",
  }),
  coverage_failure: Object.freeze({
    id: "coverage_failure",
    label: "Coverage failure",
    group: "validation-stress",
    description: "Demonstrates under-coverage warnings in a synthetic diagnostic report.",
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 1.0, observationNoise: 0.9, calibrationOffset: 0.025, domainShift: 0, coverageShift: -0.17, splitLeakage: false, failureMode: "coverage-underflow",
    chartLabel: "Coverage failure demo",
  }),
  split_leakage: Object.freeze({
    id: "split_leakage",
    label: "Split overlap warning",
    group: "validation-stress",
    description: "Intentionally shares synthetic realization IDs across splits to test leakage warnings.",
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 1.0, observationNoise: 0.8, calibrationOffset: 0, domainShift: 0, coverageShift: 0, splitLeakage: true, failureMode: "split-overlap",
    chartLabel: "Leakage warning demo",
  }),
  sparse_data: Object.freeze({
    id: "sparse_data",
    label: "Sparse dataset",
    group: "data-volume",
    description: "Small sample sizes and broad intervals exercise empty and low-count presentation states.",
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 1.7, observationNoise: 1.35, calibrationOffset: 0, domainShift: 0, coverageShift: -0.025, splitLeakage: false, failureMode: "small-sample",
    chartLabel: "Low-count demo",
  }),
  high_resolution: Object.freeze({
    id: "high_resolution",
    label: "High-volume dashboard",
    group: "data-volume",
    description: "Uses larger default tables and longer series for pagination and chart performance tests.",
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 0.95, observationNoise: 0.8, calibrationOffset: 0, domainShift: 0, coverageShift: 0, splitLeakage: false, failureMode: null,
    chartLabel: "High-volume demo",
  }),
  malformed_artifacts: Object.freeze({
    id: "malformed_artifacts",
    label: "Import error states",
    group: "ui-test-cases",
    description: "Provides separate malformed and unknown-shape examples for import-dialog testing.",
    center: { omega_m: 0.30, sigma8: 0.81, h: 0.68, n_s: 0.965, sum_mnu_ev: 0.12, bias: 1.0 },
    spread: 1.0, observationNoise: 0.8, calibrationOffset: 0, domainShift: 0, coverageShift: 0, splitLeakage: false, failureMode: "malformed-imports",
    chartLabel: "Importer test fixtures",
  }),
});

const SCENARIO_IDS = Object.freeze(Object.keys(MOCK_SCENARIOS));

/** Parse and clamp user controls into an explicit, stable generation contract. */
export function normalizeMockOptions(options = {}) {
  const scenario = SCENARIO_IDS.includes(options.scenario) ? options.scenario : "baseline";
  const seedRaw = Number(options.seed ?? 90721);
  const seed = Number.isFinite(seedRaw) ? (Math.trunc(seedRaw) >>> 0) : 90721;
  const sampleCount = clampInteger(options.sampleCount ?? 500, 20, 2000, 500);
  const observationCount = clampInteger(options.observationCount ?? 300, 20, 5000, 300);
  const chainCount = clampInteger(options.chainCount ?? 4, 1, 8, 4);
  const samplesPerChain = clampInteger(options.samplesPerChain ?? 250, 50, 1500, 250);
  const spectrumPoints = clampInteger(options.spectrumPoints ?? 180, 30, 800, 180);
  const historyCount = clampInteger(options.historyCount ?? 48, 8, 300, 48);
  return Object.freeze({ scenario, seed, sampleCount, observationCount, chainCount, samplesPerChain, spectrumPoints, historyCount });
}

/** Create all chart tables and report artifacts for one synthetic demo run. */
export function generateMockBundle(inputOptions = {}) {
  const options = normalizeMockOptions(inputOptions);
  const scenario = MOCK_SCENARIOS[options.scenario];
  const random = createRandom(options.seed);
  const createdAt = new Date().toISOString();
  const runId = `mock-${options.scenario}-${options.seed.toString(16).padStart(8, "0")}`;
  const profileNotes = createProfileNotes(scenario);
  const parameterSamples = generateParameterSamples({ random, options, scenario, runId });
  const posteriorChains = generatePosteriorChains({ random, options, scenario, runId });
  const powerSpectrum = generatePowerSpectrum({ random, options, scenario });
  const expansionHistory = generateExpansionHistory({ random, options, scenario });
  const surveyCatalog = generateSurveyCatalog({ random, options, scenario, runId });
  const cmbSummary = generateCmbSummary({ random, options, scenario });
  const lensingCatalog = generateLensingCatalog({ random, options, scenario, runId });
  const benchmarkReport = generateBenchmarkReport({ random, options, scenario, runId, parameterSamples });
  const validationReport = generateValidationReport({ random, options, scenario, benchmarkReport, runId });
  const experimentHistory = generateExperimentHistory({ random, options, scenario, runId });
  const ablationReport = generateAblationReport({ random, scenario, options });
  const dataQualityReport = generateDataQualityReport({ random, options, scenario, surveyCatalog });
  const fieldSummary = generateFieldSummary({ random, options, scenario });

  const dataset = {
    schema_version: MOCK_SCHEMA_VERSION,
    dataset_name: `NUFIELD synthetic demo — ${scenario.label}`,
    dataset_id: `synthetic-${runId}`,
    created_at: createdAt,
    physics_level: "toy",
    is_synthetic: true,
    source_kind: MOCK_SOURCE_KIND,
    scenario: scenario.id,
    random_seed: options.seed,
    n_simulations: options.sampleCount,
    n_observations: options.observationCount,
    grid_size: scenario.id === "high_resolution" ? 128 : scenario.id === "sparse_data" ? 16 : 64,
    feature_names: ["field_mean", "field_variance", "power_slope", "density_proxy", "noise_level", "mask_fraction"],
    parameter_names: [...PARAMETER_KEYS],
    sampling: { train: Math.floor(options.sampleCount * 0.65), validation: Math.floor(options.sampleCount * 0.15), test: options.sampleCount - Math.floor(options.sampleCount * 0.8) },
    scientific_warning: MOCK_SCIENTIFIC_WARNING,
    caveats: [
      "Rows are generated from display-oriented toy distributions, not from a Boltzmann solver.",
      "Parameter associations are configured for interface demonstration, not inferred from data.",
      "No measurement, prior, likelihood, posterior, or physical constraint is being validated.",
    ],
    notes: profileNotes,
  };

  const manifest = {
    schema_version: MOCK_SCHEMA_VERSION,
    manifest_kind: "synthetic-demo-manifest",
    dataset_id: dataset.dataset_id,
    dataset_name: dataset.dataset_name,
    physics_level: "toy",
    is_synthetic: true,
    source_kind: MOCK_SOURCE_KIND,
    generator_version: MOCK_ENGINE_VERSION,
    generated_at: createdAt,
    random_seed: options.seed,
    scenario: scenario.id,
    requested_sizes: { ...options },
    artifact_names: ["dataset.json", "benchmark_report.json", "posterior_samples.json", "power_spectrum.json", "expansion_history.json", "survey_catalog.json", "cmb_summary.json", "lensing_catalog.json", "validation_report.json", "experiment_history.json", "ablation_report.json", "data_quality_report.json", "field_summary.json"],
    deterministic_rows: true,
    deterministic_exclusion: "generated_at is a wall-clock timestamp and therefore differs between runs",
    scientific_warning: MOCK_SCIENTIFIC_WARNING,
  };

  const bundle = {
    schema_version: MOCK_SCHEMA_VERSION,
    bundle_kind: "nufield-synthetic-demo-bundle",
    generated_at: createdAt,
    run_id: runId,
    options,
    scenario: { id: scenario.id, label: scenario.label, group: scenario.group, description: scenario.description, failure_mode: scenario.failureMode },
    provenance: buildProvenance(options, scenario, runId),
    dataset,
    manifest,
    benchmark_report: benchmarkReport,
    parameter_samples: { schema_version: MOCK_SCHEMA_VERSION, is_synthetic: true, run_id: runId, rows: parameterSamples, parameter_names: [...PARAMETER_KEYS], scientific_warning: MOCK_SCIENTIFIC_WARNING },
    posterior_chains: posteriorChains,
    power_spectrum: powerSpectrum,
    expansion_history: expansionHistory,
    survey_catalog: surveyCatalog,
    cmb_summary: cmbSummary,
    lensing_catalog: lensingCatalog,
    validation_report: validationReport,
    experiment_history: experimentHistory,
    ablation_report: ablationReport,
    data_quality_report: dataQualityReport,
    field_summary: fieldSummary,
    import_error_examples: scenario.id === "malformed_artifacts" ? generateImportErrorExamples() : undefined,
  };
  return bundle;
}

export function listMockArtifacts(bundle) {
  if (!bundle || typeof bundle !== "object") return [];
  const artifactMap = {
    "dataset.json": bundle.dataset,
    "dataset.manifest.json": bundle.manifest,
    "benchmark_report.json": bundle.benchmark_report,
    "parameter_samples.json": bundle.parameter_samples,
    "posterior_chains.json": bundle.posterior_chains,
    "power_spectrum.json": bundle.power_spectrum,
    "expansion_history.json": bundle.expansion_history,
    "survey_catalog.json": bundle.survey_catalog,
    "cmb_summary.json": bundle.cmb_summary,
    "lensing_catalog.json": bundle.lensing_catalog,
    "validation_report.json": bundle.validation_report,
    "experiment_history.json": bundle.experiment_history,
    "ablation_report.json": bundle.ablation_report,
    "data_quality_report.json": bundle.data_quality_report,
    "field_summary.json": bundle.field_summary,
  };
  if (bundle.import_error_examples) artifactMap["import_error_examples.json"] = bundle.import_error_examples;
  return Object.entries(artifactMap).map(([filename, content]) => ({ filename, content, bytes: JSON.stringify(content).length }));
}

export function summarizeMockBundle(bundle) {
  const report = bundle?.benchmark_report;
  const rows = bundle?.parameter_samples?.rows ?? [];
  const vals = rows.map((row) => row.omega_m).filter(Number.isFinite);
  return {
    scenario: bundle?.scenario?.label ?? "Unknown synthetic scenario",
    seed: bundle?.options?.seed ?? null,
    parameterRows: rows.length,
    observations: bundle?.survey_catalog?.rows?.length ?? 0,
    posteriorRows: (bundle?.posterior_chains?.chains ?? []).reduce((sum, chain) => sum + chain.samples.length, 0),
    spectrumPoints: bundle?.power_spectrum?.points?.length ?? 0,
    experiments: bundle?.experiment_history?.runs?.length ?? 0,
    validationStatus: report?.split_disjoint === false || bundle?.validation_report?.overall_status === "FAIL" ? "ATTENTION" : bundle?.validation_report?.overall_status ?? "SYNTHETIC",
    omegaM: vals.length ? { mean: mean(vals), min: Math.min(...vals), max: Math.max(...vals) } : null,
    artifactCount: listMockArtifacts(bundle).length,
    allSynthetic: bundle?.dataset?.is_synthetic === true && bundle?.manifest?.is_synthetic === true,
  };
}

export function toPrettyJSON(value) {
  return JSON.stringify(value, null, 2) + "\n";
}

export function toCSV(rows, columns = null) {
  if (!Array.isArray(rows) || rows.length === 0) return "";
  const keys = columns ?? [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const escape = (value) => {
    if (value === null || value === undefined) return "";
    const text = typeof value === "object" ? JSON.stringify(value) : String(value);
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [keys.map(escape).join(","), ...rows.map((row) => keys.map((key) => escape(row[key])).join(","))].join("\r\n") + "\r\n";
}

function buildProvenance(options, scenario, runId) {
  return {
    provenance_id: `provenance-${runId}`,
    source_kind: MOCK_SOURCE_KIND,
    generator_version: MOCK_ENGINE_VERSION,
    code_path: "frontend/src/data/mock-data-engine.js",
    generated_from_real_observations: false,
    generated_from_cosmology_solver: false,
    posterior_is_scientific: false,
    calibrated: false,
    physical_units_verified: false,
    scenario: scenario.id,
    seed: options.seed,
    parameters: { sample_count: options.sampleCount, observation_count: options.observationCount, chain_count: options.chainCount, samples_per_chain: options.samplesPerChain },
    warning: MOCK_SCIENTIFIC_WARNING,
  };
}

function createProfileNotes(scenario) {
  const notes = [
    `Scenario "${scenario.id}" is a manually designed UI test profile.`,
    `Expected stress behavior: ${scenario.failureMode ?? "standard happy-path rendering"}.`,
    "Parameter-space patterns are manufactured by the generator, not scientifically inferred.",
  ];
  if (scenario.splitLeakage) notes.push("Intentional shared realization IDs are injected to test leakage warnings.");
  if (scenario.domainShift) notes.push("Held-out rows are shifted for a domain-shift illustration.");
  if (scenario.coverageShift < 0) notes.push("Coverage values are artificially depressed to demonstrate under-coverage states.");
  return notes;
}

function generateParameterSamples({ random, options, scenario, runId }) {
  const rows = [];
  const trainCount = Math.floor(options.sampleCount * 0.65);
  const validationCount = Math.floor(options.sampleCount * 0.15);
  for (let index = 0; index < options.sampleCount; index += 1) {
    const split = index < trainCount ? "train" : index < trainCount + validationCount ? "validation" : "test";
    const heldOutOffset = split === "test" ? scenario.domainShift : 0;
    const latent = randomNormal(random);
    const correlated = randomNormal(random);
    const parameterValues = {};
    PARAMETER_KEYS.forEach((key) => {
      const definition = MOCK_PARAMETER_DEFINITIONS[key];
      const center = scenario.center[key] ?? definition.center;
      let normalized = randomNormal(random) * definition.sigma * scenario.spread;
      if (key === "sigma8") normalized += -latent * definition.sigma * 0.38;
      if (key === "omega_m") normalized += latent * definition.sigma * 0.44;
      if (key === "h") normalized += correlated * definition.sigma * 0.62;
      if (key === "sum_mnu_ev") normalized += scenario.calibrationOffset * 0.24;
      if (key === "bias") normalized += scenario.calibrationOffset * 0.15;
      if (key === "omega_m" || key === "sigma8") normalized += heldOutOffset * (key === "omega_m" ? 0.42 : -0.25);
      parameterValues[key] = round(clamp(center + normalized, definition.min, definition.max), 6);
    });
    const qualityScore = clamp(0.94 - scenario.observationNoise * random() * 0.15 - (random() < 0.025 ? 0.35 : 0), 0.18, 0.995);
    rows.push({
      id: `${runId}-sample-${String(index + 1).padStart(5, "0")}`,
      realization_id: `realization-${String(index + 1).padStart(5, "0")}`,
      split,
      ...parameterValues,
      synthetic_score: round(qualityScore, 4),
      quality_flag: qualityScore < 0.65 ? "review" : "synthetic-ok",
      synthetic_noise_scale: round(scenario.observationNoise * (0.75 + random() * 0.5), 4),
      is_synthetic: true,
    });
  }
  if (scenario.splitLeakage && rows.length > 4) {
    rows[rows.findIndex((row) => row.split === "test")].realization_id = rows.find((row) => row.split === "train").realization_id;
  }
  return rows;
}

function generatePosteriorChains({ random, options, scenario, runId }) {
  const chains = [];
  for (let chainIndex = 0; chainIndex < options.chainCount; chainIndex += 1) {
    const samples = [];
    let previous = { ...scenario.center };
    for (let step = 0; step < options.samplesPerChain; step += 1) {
      const row = {};
      PARAMETER_KEYS.forEach((key) => {
        const definition = MOCK_PARAMETER_DEFINITIONS[key];
        const proposal = previous[key] + randomNormal(random) * definition.sigma * (0.24 + 0.12 * scenario.spread);
        const bounded = clamp(proposal, definition.min, definition.max);
        row[key] = round(bounded, 6);
        previous[key] = bounded;
      });
      const burnIn = step < Math.floor(options.samplesPerChain * 0.1);
      samples.push({ iteration: step, ...row, log_density_toy: round(-0.5 * PARAMETER_KEYS.reduce((acc, key) => acc + ((row[key] - scenario.center[key]) / MOCK_PARAMETER_DEFINITIONS[key].sigma) ** 2, 0), 5), accepted: random() > 0.22, burn_in: burnIn, is_synthetic: true });
    }
    chains.push({ chain_id: `chain-${chainIndex + 1}`, seed_offset: chainIndex * 100003, samples });
  }
  return { schema_version: MOCK_SCHEMA_VERSION, is_synthetic: true, sampler_label: "random-walk illustration (not a production sampler)", run_id: runId, chains, warning: "These are generated pseudo-chains for layout and diagnostics testing only. They are not MCMC posterior samples and have no convergence guarantee." };
}

function generatePowerSpectrum({ random, options, scenario }) {
  const points = [];
  for (let index = 0; index < options.spectrumPoints; index += 1) {
    const u = index / (options.spectrumPoints - 1);
    const k = 10 ** (-2 + u * 3.2);
    const shape = (k / 0.08) ** 0.82 / (1 + (k / 0.8) ** 1.1);
    const turnover = 0.66 + 0.34 / (1 + (k / 0.055) ** 1.55);
    const wiggle = 1 + 0.05 * Math.sin(7.1 * Math.log(k + 0.003));
    const toySuppression = Math.exp(-(0.25 + scenario.center.sum_mnu_ev * 0.5) * (k / 1.9) ** 0.62);
    const noise = 1 + randomNormal(random) * 0.025 * scenario.observationNoise;
    const p = Math.max(1e-7, shape * turnover * wiggle * toySuppression * noise);
    points.push({ k: round(k, 7), p_toy: round(p, 9), p_reference_toy: round(shape * turnover * wiggle, 9), relative_noise: round(noise - 1, 6), is_synthetic: true });
  }
  return { schema_version: MOCK_SCHEMA_VERSION, x_name: "k", x_unit: "h/Mpc (display label only)", y_name: "P_toy(k)", y_unit: "arbitrary units", transform: "hand-designed display curve; not a physical transfer function", points, is_synthetic: true, scientific_warning: MOCK_SCIENTIFIC_WARNING };
}

function generateExpansionHistory({ random, options, scenario }) {
  const points = [];
  const count = Math.max(50, Math.min(160, options.spectrumPoints));
  for (let index = 0; index < count; index += 1) {
    const u = index / (count - 1);
    const redshift = 10 * (1 - u);
    const scaleFactor = 1 / (1 + redshift);
    // The scale factor relation is exact as a definition; the remaining fields are visual scaffolding.
    const toyDistance = 1.25 * (1 - Math.exp(-redshift / 2.8)) + randomNormal(random) * 0.006 * scenario.observationNoise;
    points.push({ redshift: round(redshift, 5), scale_factor: round(1 / (1 + round(redshift, 5)), 7), toy_distance_index: round(Math.max(0, toyDistance), 7), epoch_label: redshift < 0.1 ? "late" : redshift < 2 ? "intermediate" : "early", is_synthetic: true });
  }
  return { schema_version: MOCK_SCHEMA_VERSION, x_name: "redshift", y_names: ["scale_factor", "toy_distance_index"], points, is_synthetic: true, scientific_warning: "Only scale_factor = 1/(1+z) is a definitional relation here. toy_distance_index is arbitrary UI data, not a distance calculation." };
}

function generateSurveyCatalog({ random, options, scenario, runId }) {
  const rows = [];
  for (let index = 0; index < options.observationCount; index += 1) {
    const redshift = clamp(-Math.log(Math.max(0.0001, random())) * 0.58, 0.005, 3.25);
    const ra = random() * 360;
    const dec = Math.asin(random() * 2 - 1) * (180 / Math.PI);
    const latentDensity = Math.sin((ra + 13) * Math.PI / 180) * Math.cos(dec * Math.PI / 180) + randomNormal(random) * 0.35;
    const noise = randomNormal(random) * 0.08 * scenario.observationNoise * (1 + redshift * 0.24);
    const completeness = clamp(0.97 - redshift * 0.12 - random() * 0.12 * scenario.observationNoise, 0.12, 0.995);
    rows.push({
      object_id: `${runId}-obj-${String(index + 1).padStart(6, "0")}`,
      ra_deg: round(ra, 6),
      dec_deg: round(dec, 6),
      redshift_toy: round(redshift, 6),
      density_proxy_true_toy: round(latentDensity, 6),
      density_proxy_observed_toy: round(latentDensity + noise + scenario.calibrationOffset, 6),
      uncertainty_toy: round(Math.abs(noise) + 0.035 * scenario.observationNoise, 6),
      completeness_fraction_toy: round(completeness, 5),
      mask_flag: random() < (scenario.observationNoise > 1.5 ? 0.12 : 0.045) ? "masked-demo" : "unmasked-demo",
      survey_region: ["north-demo", "equatorial-demo", "south-demo"][Math.floor(random() * 3)],
      is_synthetic: true,
    });
  }
  return { schema_version: MOCK_SCHEMA_VERSION, catalog_label: "Synthetic angular catalog", row_count: rows.length, columns: Object.keys(rows[0] ?? {}), rows, is_synthetic: true, scientific_warning: "Coordinates and redshift-like values are generated for UI performance testing; they are not a real survey catalog." };
}

function generateCmbSummary({ random, options, scenario }) {
  const points = [];
  const count = Math.min(500, Math.max(40, Math.round(options.spectrumPoints * 1.4)));
  for (let ell = 2; ell < count + 2; ell += 1) {
    const acoustic = 1 + 0.32 * Math.sin(ell / 16) ** 2 + 0.18 * Math.sin(ell / 42);
    const damping = Math.exp(-ell / 850);
    const value = (0.18 + acoustic * damping) * (1 + randomNormal(random) * 0.018 * scenario.observationNoise);
    points.push({ ell, cl_toy: round(Math.max(0.0001, value), 8), uncertainty_toy: round(0.015 + random() * 0.025 * scenario.observationNoise, 8), is_synthetic: true });
  }
  return { schema_version: MOCK_SCHEMA_VERSION, quantity: "C_ell toy summary", points, generated_texture: true, pixelization: "not HEALPix", is_synthetic: true, scientific_warning: "This table is a hand-shaped demo curve and does not represent Planck/WMAP data or a Boltzmann calculation." };
}

function generateLensingCatalog({ random, options, scenario, runId }) {
  const count = Math.min(250, Math.max(12, Math.round(options.observationCount * 0.14)));
  const rows = [];
  for (let index = 0; index < count; index += 1) {
    const beta = (random() * 1.8 - 0.9) * 1.1;
    const thetaE = 0.35 + random() * 0.55;
    const root = Math.sqrt(beta * beta + 4 * thetaE * thetaE);
    const thetaPlus = (beta + root) / 2;
    const thetaMinus = (beta - root) / 2;
    rows.push({
      system_id: `${runId}-lens-${String(index + 1).padStart(4, "0")}`,
      source_beta_arcsec_toy: round(beta, 5),
      theta_e_arcsec_toy: round(thetaE, 5),
      image_plus_arcsec_toy: round(thetaPlus + randomNormal(random) * scenario.observationNoise * 0.01, 5),
      image_minus_arcsec_toy: round(thetaMinus + randomNormal(random) * scenario.observationNoise * 0.01, 5),
      magnification_proxy_toy: round(1 + thetaE / Math.sqrt(beta * beta + 0.04), 5),
      fit_residual_toy: round(Math.abs(randomNormal(random)) * 0.02 * scenario.observationNoise, 6),
      is_synthetic: true,
    });
  }
  return { schema_version: MOCK_SCHEMA_VERSION, lens_model: "point-mass equation in arbitrary angular units", rows, is_synthetic: true, scientific_warning: "The catalog is a UI fixture. Values are not estimates of real lens masses or cosmological distances." };
}

function generateBenchmarkReport({ random, options, scenario, runId, parameterSamples }) {
  const medianAbsoluteError = {};
  const maeByParameter = {};
  PARAMETER_KEYS.forEach((key) => {
    const base = MOCK_PARAMETER_DEFINITIONS[key].sigma * (0.07 + scenario.observationNoise * 0.075);
    maeByParameter[key] = round(Math.max(0.0001, base * (1 + Math.abs(randomNormal(random)) * 0.2) + Math.abs(scenario.calibrationOffset) * 0.2 + scenario.domainShift * 0.18), 6);
    medianAbsoluteError[key] = maeByParameter[key];
  });
  const expectedCoverage = [0.5, 0.68, 0.9, 0.95, 0.99];
  const coverageDiagnostic = expectedCoverage.map((nominal, index) => {
    const observed = clamp(nominal + scenario.coverageShift + randomNormal(random) * 0.012, 0.02, 0.995);
    return { nominal, observed: round(observed, 4), n: Math.max(20, Math.floor(options.sampleCount * (0.5 + nominal * 0.5))), interval_level: nominal, is_synthetic: true, status: observed < nominal - 0.06 ? "under-coverage-demo" : observed > nominal + 0.06 ? "over-coverage-demo" : "within-demo-band" };
  });
  const trainIds = parameterSamples.filter((row) => row.split === "train").map((row) => row.realization_id);
  const validationIds = parameterSamples.filter((row) => row.split === "validation").map((row) => row.realization_id);
  const testIds = parameterSamples.filter((row) => row.split === "test").map((row) => row.realization_id);
  const overlapExists = hasOverlap(trainIds, validationIds) || hasOverlap(trainIds, testIds) || hasOverlap(validationIds, testIds);
  return {
    schema_version: MOCK_SCHEMA_VERSION,
    report_id: `${runId}-benchmark` ,
    created_at: new Date().toISOString(),
    method: { algorithm: "synthetic-demo-regressor", version: MOCK_ENGINE_VERSION, note: "Label is a visualization placeholder, not a fitted predictive model." },
    parameter_names: [...PARAMETER_KEYS],
    parameter_points: parameterSamples.slice(0, 600),
    median_absolute_error: medianAbsoluteError,
    mae_by_parameter: maeByParameter,
    coverage_diagnostic: coverageDiagnostic,
    n_train: trainIds.length,
    n_validation_held_out: validationIds.length,
    n_test: testIds.length,
    train_ids: unique(trainIds),
    validation_ids: unique(validationIds),
    test_ids: unique(testIds),
    split_disjoint: !overlapExists,
    physics_level: "toy",
    is_synthetic: true,
    scenario: scenario.id,
    scientific_warning: MOCK_SCIENTIFIC_WARNING,
  };
}

function generateValidationReport({ random, options, scenario, benchmarkReport, runId }) {
  const checks = [
    makeCheck("schema_shape", "Artifact shape", "PASS", "Required keys exist in this generated fixture."),
    makeCheck("finite_values", "Finite display values", "PASS", "Generator clamps all generated numeric fields to finite ranges."),
    makeCheck("scenario_provenance", "Synthetic provenance label", "PASS", "Synthetic label is attached to the mock bundle."),
    makeCheck("split_disjointness", "Realization split disjointness", benchmarkReport.split_disjoint ? "PASS" : "FAIL", benchmarkReport.split_disjoint ? "No shared realization IDs in these generated splits." : "The selected UI stress profile deliberately injects a shared split ID."),
  ];
  const coverageBad = benchmarkReport.coverage_diagnostic.some((item) => item.observed < item.nominal - 0.06);
  checks.push(makeCheck("coverage_demo", "Coverage diagnostic demo", coverageBad ? "WARN" : "PASS", coverageBad ? "Synthetic coverage is intentionally below the nominal value in this profile." : "Illustrative coverage values sit within the broad demo tolerance."));
  if (scenario.failureMode === "calibration-offset") checks.push(makeCheck("calibration_offset", "Estimator calibration", "WARN", "A synthetic calibration offset was injected; no real estimator was fit."));
  if (scenario.failureMode === "domain-shift") checks.push(makeCheck("domain_shift", "Held-out distribution", "WARN", "Test rows are deliberately shifted away from train rows."));
  if (scenario.failureMode === "noisy-observations") checks.push(makeCheck("noise_stress", "Noise stress", "WARN", "Observation noise is intentionally increased for the demo."));
  const status = checks.some((check) => check.status === "FAIL") ? "FAIL" : checks.some((check) => check.status === "WARN") ? "WARN" : "PASS";
  return { schema_version: MOCK_SCHEMA_VERSION, report_id: `${runId}-validation`, created_at: new Date().toISOString(), overall_status: status, checks, summary: { total: checks.length, pass: checks.filter((check) => check.status === "PASS").length, warn: checks.filter((check) => check.status === "WARN").length, fail: checks.filter((check) => check.status === "FAIL").length }, is_synthetic: true, scientific_warning: "This is a software-demo validation report. A PASS checks generated fixture properties only; it does not validate scientific accuracy." };
}

function generateExperimentHistory({ random, options, scenario, runId }) {
  const phases = ["queued", "preparing", "simulating", "extracting-features", "fitting-demo", "validating", "complete"];
  const names = ["baseline-a", "noise-ablation", "mask-stress", "heldout-check", "parameter-sweep", "seed-repeat", "coverage-pass", "catalog-refresh"];
  const runs = [];
  for (let index = 0; index < options.historyCount; index += 1) {
    const failureIndex = scenario.failureMode ? Math.floor(options.historyCount * 0.68) : -1;
    const isFailureRun = index === failureIndex;
    const id = `${runId}-history-${String(index + 1).padStart(3, "0")}`;
    const runtime = Math.max(2, Math.round(8 + random() * 72 * scenario.observationNoise));
    const startedOffsetMinutes = (options.historyCount - index) * (7 + Math.round(random() * 21));
    runs.push({
      run_id: id,
      label: `${names[index % names.length]}-${String(index + 1).padStart(2, "0")}`,
      scenario: scenario.id,
      status: isFailureRun && scenario.failureMode !== "small-sample" ? "warning" : index === options.historyCount - 1 ? "running-demo" : "complete-demo",
      phase: phases[Math.min(phases.length - 1, Math.floor(random() * phases.length))],
      started_at_offset_minutes: startedOffsetMinutes,
      duration_seconds_toy: runtime,
      seed: (options.seed + index * 997) >>> 0,
      simulations: Math.max(20, Math.round(options.sampleCount * (0.45 + random() * 0.9))),
      observations: Math.max(20, Math.round(options.observationCount * (0.4 + random() * 0.9))),
      toy_score: round(clamp(0.9 - scenario.observationNoise * 0.07 - Math.abs(randomNormal(random)) * 0.05 - (isFailureRun ? 0.16 : 0), 0.1, 0.99), 4),
      warnings: isFailureRun ? [scenario.failureMode] : scenario.domainShift && index % 6 === 0 ? ["heldout-domain-shift-demo"] : [],
      is_synthetic: true,
    });
  }
  runs.sort((a, b) => b.started_at_offset_minutes - a.started_at_offset_minutes);
  return { schema_version: MOCK_SCHEMA_VERSION, history_kind: "synthetic-run-history", run_count: runs.length, runs, is_synthetic: true, scientific_warning: "Run history is fabricated to demonstrate interface states, not a record of executed model runs." };
}

function generateAblationReport({ random, scenario, options }) {
  const components = ["field-summary", "power-features", "angular-catalog", "cmb-summary", "lensing-proxy", "late-fusion", "all-inputs"];
  const results = components.map((component, index) => {
    const numberOfInputs = index === components.length - 1 ? 5 : index + 1;
    const score = clamp(0.58 + index * 0.035 - scenario.observationNoise * 0.025 + randomNormal(random) * 0.014, 0.35, 0.94);
    return { component, inputs_enabled: numberOfInputs, toy_score: round(score, 5), toy_score_error: round(0.015 + random() * 0.025, 5), sample_count: Math.max(20, Math.floor(options.sampleCount * (0.72 + random() * 0.25))), conclusion: "Synthetic comparison value; does not establish predictive utility.", is_synthetic: true };
  });
  return { schema_version: MOCK_SCHEMA_VERSION, ablation_kind: "fabricated UI comparison", results, is_synthetic: true, scientific_warning: MOCK_SCIENTIFIC_WARNING };
}

function generateDataQualityReport({ random, options, scenario, surveyCatalog }) {
  const rows = surveyCatalog.rows;
  const masked = rows.filter((row) => row.mask_flag !== "unmasked-demo").length;
  const completeness = rows.map((row) => row.completeness_fraction_toy);
  const avgCompleteness = completeness.length ? mean(completeness) : 0;
  const metrics = [
    { key: "row-count", label: "Catalog rows", value: rows.length, unit: "rows", status: rows.length < 50 ? "WARN" : "PASS" },
    { key: "missingness", label: "Masked fraction", value: round(masked / Math.max(1, rows.length), 5), unit: "fraction", status: masked / Math.max(1, rows.length) > 0.12 ? "WARN" : "PASS" },
    { key: "completeness", label: "Mean completeness", value: round(avgCompleteness, 5), unit: "fraction", status: avgCompleteness < 0.72 ? "WARN" : "PASS" },
    { key: "noise", label: "Configured noise multiplier", value: scenario.observationNoise, unit: "relative toy scale", status: scenario.observationNoise > 1.5 ? "WARN" : "PASS" },
    { key: "seed", label: "Reproducibility seed", value: options.seed, unit: "integer", status: "PASS" },
  ];
  return { schema_version: MOCK_SCHEMA_VERSION, report_kind: "synthetic quality summary", metrics, generated_checks: metrics.length, is_synthetic: true, scientific_warning: "Quality metrics characterize generated demo rows only; they say nothing about any actual survey." };
}

function generateFieldSummary({ random, options, scenario }) {
  const slices = [];
  const sliceCount = 40;
  for (let index = 0; index < sliceCount; index += 1) {
    const x = index / (sliceCount - 1);
    const wave = Math.sin((x * 4.8 + 0.1) * Math.PI) * 0.42 + Math.sin((x * 11 + 0.17) * Math.PI) * 0.17;
    const envelope = 0.55 + 0.45 * Math.cos((x - 0.5) * Math.PI);
    const density = wave * envelope + randomNormal(random) * 0.06 * scenario.observationNoise;
    const velocity = Math.cos(x * 5.7 * Math.PI) * 0.35 + randomNormal(random) * 0.045;
    slices.push({ index, x_fraction: round(x, 5), density_contrast_toy: round(density, 6), velocity_proxy_toy: round(velocity, 6), is_synthetic: true });
  }
  return { schema_version: MOCK_SCHEMA_VERSION, field_kind: "one-dimensional illustration slice", slices, grid_shape: scenario.id === "high_resolution" ? [128, 128, 128] : [64, 64, 64], values_stored: false, is_synthetic: true, scientific_warning: "This summary is not a simulation volume. It is a one-dimensional display pattern for chart and UI demonstrations." };
}

function generateImportErrorExamples() {
  return {
    schema_version: MOCK_SCHEMA_VERSION,
    is_synthetic: true,
    examples: [
      { filename: "example-malformed.json", raw_text: "{\"dataset_name\": \"truncated\",", expected_result: "parse-error", description: "Malformed JSON text. This file must be imported as raw text by a dedicated test harness; it is not a valid JSON artifact." },
      { filename: "example-unknown-shape.json", content: { hello: "world", rows: [1, 2, 3] }, expected_result: "unclassified-extra", description: "Valid JSON, but no recognized NUFIELD artifact signature." },
      { filename: "example-array-root.json", content: [{ name: "array-root" }], expected_result: "shape-error", description: "A top-level array to exercise importer validation." },
      { filename: "example-deep-nesting.json", content: { a: { b: { c: { d: "..." } } } }, expected_result: "depth-test", description: "A shallow stand-in for a too-deep JSON nesting test." },
    ],
    scientific_warning: "Import test cases are not research data and should never be interpreted as model output.",
  };
}

function makeCheck(id, label, status, detail) {
  return { id, label, status, detail, generated_fixture_check: true, is_synthetic: true };
}

function createRandom(seed) {
  let state = seed >>> 0;
  return function random() {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function randomNormal(random) {
  const u1 = Math.max(random(), 1e-12);
  const u2 = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clampInteger(value, min, max, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, Math.trunc(n))) : fallback;
}

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function round(value, digits = 5) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function unique(values) {
  return [...new Set(values)];
}

function hasOverlap(first, second) {
  const candidates = new Set(first);
  return second.some((value) => candidates.has(value));
}
