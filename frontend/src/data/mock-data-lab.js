import { $, $$, announce, escapeHTML, formatNumber, setText } from "../core/dom.js";
import { updateImportState } from "../core/state.js";
import { navigateTo } from "../components/navigation.js";
import {
  generateMockBundle,
  listMockArtifacts,
  MOCK_SCENARIOS,
  MOCK_PARAMETER_DEFINITIONS,
  normalizeMockOptions,
  summarizeMockBundle,
  toCSV,
  toPrettyJSON,
} from "./mock-data-engine.js";

const DEFAULTS = Object.freeze({ scenario: "baseline", seed: 90721, sampleCount: 500, observationCount: 300, chainCount: 4, samplesPerChain: 250, spectrumPoints: 180, historyCount: 48 });
const VISIBLE_ROWS = 18;
let activeBundle = null;
let filteredSampleRows = [];
let tableFilter = "";
let chosenParameter = "omega_m";
let chosenSplit = "all";

export function initMockDataLab() {
  bindGenerationControls();
  bindScenarioDescription();
  bindArtifactExports();
  bindTableControls();
  bindBundleActions();
  buildScenarioOptions();
  generateAndRender(DEFAULTS, { announceChange: false });
}

export function getActiveMockBundle() {
  return activeBundle;
}

function bindGenerationControls() {
  $("#mock-generate-button")?.addEventListener("click", () => generateAndRender(readOptions(), { announceChange: true }));
  $("#mock-random-seed")?.addEventListener("click", () => {
    const nextSeed = Math.floor(Math.random() * 2 ** 32) >>> 0;
    const input = $("#mock-seed-input");
    if (input) input.value = String(nextSeed);
    generateAndRender(readOptions({ seed: nextSeed }), { announceChange: true });
  });
  $("#mock-reset-button")?.addEventListener("click", () => {
    setInputValues(DEFAULTS);
    tableFilter = "";
    chosenParameter = "omega_m";
    chosenSplit = "all";
    if ($("#mock-table-search")) $("#mock-table-search").value = "";
    if ($("#mock-table-parameter")) $("#mock-table-parameter").value = chosenParameter;
    if ($("#mock-table-split")) $("#mock-table-split").value = chosenSplit;
    generateAndRender(DEFAULTS, { announceChange: true });
  });
  $("#mock-scenario-select")?.addEventListener("change", renderScenarioDescription);
}

function bindScenarioDescription() {
  renderScenarioDescription();
}

function bindArtifactExports() {
  $("#mock-artifact-list")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-export-artifact]");
    if (!button || !activeBundle) return;
    const fileName = button.dataset.exportArtifact;
    const artifact = listMockArtifacts(activeBundle).find((entry) => entry.filename === fileName);
    if (!artifact) return;
    downloadText(fileName, toPrettyJSON(artifact.content), "application/json;charset=utf-8");
    announce(`Exported ${fileName}. It is labeled synthetic.`, "success");
  });
}

function bindTableControls() {
  $("#mock-table-search")?.addEventListener("input", (event) => {
    tableFilter = event.currentTarget.value.trim().toLowerCase();
    renderSampleTable();
  });
  $("#mock-table-parameter")?.addEventListener("change", (event) => {
    chosenParameter = event.currentTarget.value;
    renderDistribution();
  });
  $("#mock-table-split")?.addEventListener("change", (event) => {
    chosenSplit = event.currentTarget.value;
    renderSampleTable();
    renderDistribution();
  });
  $("#mock-export-csv")?.addEventListener("click", () => {
    if (!activeBundle) return;
    const rows = activeBundle.parameter_samples.rows;
    downloadText("nufield-synthetic-parameter-samples.csv", toCSV(rows), "text/csv;charset=utf-8");
    announce("Exported synthetic parameter samples as CSV.", "success");
  });
}

function bindBundleActions() {
  $("#mock-export-bundle")?.addEventListener("click", () => {
    if (!activeBundle) return;
    downloadText("nufield-synthetic-demo-bundle.json", toPrettyJSON(activeBundle), "application/json;charset=utf-8");
    announce("Exported the complete synthetic demo bundle as JSON.", "success");
  });
  $("#mock-load-benchmark")?.addEventListener("click", () => {
    if (!activeBundle) return;
    const now = new Date().toISOString();
    updateImportState({
      dataset: wrappedArtifact("dataset.json", activeBundle.dataset, now),
      manifest: wrappedArtifact("dataset.manifest.json", activeBundle.manifest, now),
      benchmark: wrappedArtifact("benchmark_report.json", activeBundle.benchmark_report, now),
      extras: [
        wrappedArtifact("mock_validation_report.json", activeBundle.validation_report, now),
        wrappedArtifact("mock_data_quality_report.json", activeBundle.data_quality_report, now),
      ],
    });
    announce("Loaded synthetic demo metadata into Overview and Run diagnostics. These are not real results.", "info");
    navigateTo("diagnostics");
    window.dispatchEvent(new CustomEvent("nufield:mockimport", { detail: { scenario: activeBundle.scenario.id } }));
  });
}

function buildScenarioOptions() {
  const select = $("#mock-scenario-select");
  if (!select || select.options.length > 1) return;
  const groups = new Map();
  Object.values(MOCK_SCENARIOS).forEach((scenario) => {
    const group = scenario.group || "other";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(scenario);
  });
  select.replaceChildren();
  const groupLabels = {
    "happy-path": "Happy path",
    "parameter-shift": "Parameter variations",
    "observation-quality": "Observation quality",
    "validation-stress": "Validation stress tests",
    "data-volume": "Data volume",
    "ui-test-cases": "Importer test cases",
  };
  groups.forEach((scenarios, groupName) => {
    const group = document.createElement("optgroup");
    group.label = groupLabels[groupName] ?? groupName;
    scenarios.forEach((scenario) => {
      const option = document.createElement("option");
      option.value = scenario.id;
      option.textContent = scenario.label;
      group.append(option);
    });
    select.append(group);
  });
  select.value = DEFAULTS.scenario;
}

function renderScenarioDescription() {
  const scenario = MOCK_SCENARIOS[$("#mock-scenario-select")?.value ?? "baseline"];
  if (!scenario) return;
  setText("#mock-scenario-name", scenario.label);
  setText("#mock-scenario-description", scenario.description);
  setText("#mock-scenario-group", scenario.group.replaceAll("-", " ").toUpperCase());
  setText("#mock-scenario-failure", scenario.failureMode ? `Stress behavior: ${scenario.failureMode}` : "Expected behavior: standard demo flow");
  const flag = $("#mock-scenario-flag");
  if (flag) {
    flag.textContent = scenario.failureMode ? "STRESS PROFILE" : "SYNTHETIC PROFILE";
    flag.classList.toggle("mock-flag-stress", Boolean(scenario.failureMode));
  }
}

function readOptions(overrides = {}) {
  return normalizeMockOptions({
    scenario: $("#mock-scenario-select")?.value ?? DEFAULTS.scenario,
    seed: $("#mock-seed-input")?.value ?? DEFAULTS.seed,
    sampleCount: $("#mock-sample-count")?.value ?? DEFAULTS.sampleCount,
    observationCount: $("#mock-observation-count")?.value ?? DEFAULTS.observationCount,
    chainCount: $("#mock-chain-count")?.value ?? DEFAULTS.chainCount,
    samplesPerChain: $("#mock-chain-length")?.value ?? DEFAULTS.samplesPerChain,
    spectrumPoints: $("#mock-spectrum-count")?.value ?? DEFAULTS.spectrumPoints,
    historyCount: $("#mock-history-count")?.value ?? DEFAULTS.historyCount,
    ...overrides,
  });
}

function setInputValues(options) {
  const mapping = {
    "#mock-scenario-select": options.scenario,
    "#mock-seed-input": options.seed,
    "#mock-sample-count": options.sampleCount,
    "#mock-observation-count": options.observationCount,
    "#mock-chain-count": options.chainCount,
    "#mock-chain-length": options.samplesPerChain,
    "#mock-spectrum-count": options.spectrumPoints,
    "#mock-history-count": options.historyCount,
  };
  Object.entries(mapping).forEach(([selector, value]) => { if ($(selector)) $(selector).value = String(value); });
  renderScenarioDescription();
}

function generateAndRender(inputOptions, { announceChange = true } = {}) {
  const generateButton = $("#mock-generate-button");
  if (generateButton) { generateButton.disabled = true; generateButton.textContent = "Generating…"; }
  try {
    const options = normalizeMockOptions(inputOptions);
    setInputValues(options);
    activeBundle = generateMockBundle(options);
    const summary = summarizeMockBundle(activeBundle);
    renderSummary(summary);
    renderScenarioDescription();
    renderDistribution();
    renderSampleTable();
    renderArtifactList();
    renderRunHistory();
    renderValidationChecks();
    renderPowerPreview();
    renderDataQuality();
    const size = new Blob([toPrettyJSON(activeBundle)]).size;
    setText("#mock-bundle-size", formatBytes(size));
    setText("#mock-bundle-id", activeBundle.run_id);
    setText("#mock-created-at", new Date(activeBundle.generated_at).toLocaleString());
    setText("#mock-seed-echo", String(activeBundle.options.seed));
    if (announceChange) announce(`Generated ${summary.parameterRows.toLocaleString()} parameter rows and ${summary.artifactCount} synthetic artifacts.`, "success");
  } catch (error) {
    console.error("Mock data generation failed", error);
    announce(error?.message ?? "Could not generate synthetic demo data.", "error");
  } finally {
    if (generateButton) { generateButton.disabled = false; generateButton.textContent = "Generate demo data"; }
  }
}

function renderSummary(summary) {
  const metrics = [
    ["mock-metric-rows", summary.parameterRows.toLocaleString()],
    ["mock-metric-observations", summary.observations.toLocaleString()],
    ["mock-metric-posterior", summary.posteriorRows.toLocaleString()],
    ["mock-metric-artifacts", summary.artifactCount.toLocaleString()],
  ];
  metrics.forEach(([selector, value]) => setText(`#${selector}`, value));
  setText("#mock-metric-validation", summary.validationStatus);
  setText("#mock-metric-spectrum", summary.spectrumPoints.toLocaleString());
  const quality = $("#mock-metric-validation-card");
  if (quality) quality.dataset.status = summary.validationStatus.toLowerCase();
  const label = $("#mock-data-state-label");
  if (label) label.textContent = "LOCAL SYNTHETIC DATA";
  setText("#mock-data-status", "DEMO PROFILE READY");
  setText("#mock-scenario-output", summary.scenario);
}

function renderDistribution() {
  if (!activeBundle) return;
  const definition = MOCK_PARAMETER_DEFINITIONS[chosenParameter] ?? MOCK_PARAMETER_DEFINITIONS.omega_m;
  let values = activeBundle.parameter_samples.rows;
  if (chosenSplit !== "all") values = values.filter((row) => row.split === chosenSplit);
  const numbers = values.map((row) => row[chosenParameter]).filter(Number.isFinite);
  const bins = new Array(18).fill(0);
  numbers.forEach((value) => {
    const position = (value - definition.min) / (definition.max - definition.min);
    const index = Math.min(bins.length - 1, Math.max(0, Math.floor(position * bins.length)));
    bins[index] += 1;
  });
  const maxBin = Math.max(1, ...bins);
  const host = $("#mock-distribution-chart");
  if (!host) return;
  const width = 680, height = 230, pad = { top: 16, right: 16, bottom: 35, left: 38 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const barWidth = plotWidth / bins.length - 3;
  const bars = bins.map((count, index) => {
    const x = pad.left + index * plotWidth / bins.length + 1;
    const barHeight = count / maxBin * plotHeight;
    const y = pad.top + plotHeight - barHeight;
    return `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${Math.max(1, barWidth).toFixed(2)}" height="${Math.max(0, barHeight).toFixed(2)}" rx="3" fill="url(#mockBarFill)" opacity="${count ? 0.92 : 0.25}"><title>${count} synthetic rows in bin ${index + 1}</title></rect>`;
  }).join("");
  const axis = `<line x1="${pad.left}" y1="${pad.top + plotHeight}" x2="${width - pad.right}" y2="${pad.top + plotHeight}" stroke="rgba(166,190,232,.35)"/><line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotHeight}" stroke="rgba(166,190,232,.35)"/><text x="${pad.left}" y="${height - 8}" fill="#91a4c5" font-size="11">${definition.min}</text><text x="${width - pad.right}" y="${height - 8}" text-anchor="end" fill="#91a4c5" font-size="11">${definition.max}</text><text x="${pad.left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" fill="#91a4c5" font-size="11">${definition.label} (${definition.unit})</text><text x="12" y="${pad.top + plotHeight / 2}" transform="rotate(-90 12 ${pad.top + plotHeight / 2})" fill="#91a4c5" font-size="10">Rows per bin</text>`;
  host.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Synthetic histogram for ${escapeHTML(definition.name)}" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="mockBarFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#8fa4ff"/><stop offset="1" stop-color="#4d83e8"/></linearGradient></defs>${axis}${bars}</svg><div class="mock-chart-caption"><span>${escapeHTML(definition.name)}</span><span>${numbers.length.toLocaleString()} synthetic rows · seed ${activeBundle.options.seed}</span></div>`;
  const sorted = [...numbers].sort((a, b) => a - b);
  setText("#mock-dist-mean", formatNumber(numbers.length ? numbers.reduce((sum, value) => sum + value, 0) / numbers.length : 0, 5));
  setText("#mock-dist-median", formatNumber(quantile(sorted, 0.5), 5));
  setText("#mock-dist-interval", `${formatNumber(quantile(sorted, 0.16), 4)} – ${formatNumber(quantile(sorted, 0.84), 4)}`);
}

function renderSampleTable() {
  if (!activeBundle) return;
  const host = $("#mock-sample-table");
  if (!host) return;
  const rows = activeBundle.parameter_samples.rows.filter((row) => chosenSplit === "all" || row.split === chosenSplit).filter((row) => {
    if (!tableFilter) return true;
    return [row.id, row.realization_id, row.split, row.quality_flag, row.omega_m, row.sigma8, row.sum_mnu_ev].some((value) => String(value).toLowerCase().includes(tableFilter));
  });
  filteredSampleRows = rows;
  const visible = rows.slice(0, VISIBLE_ROWS);
  if (!visible.length) {
    host.innerHTML = `<div class="mock-empty-table">No synthetic rows match this filter.</div>`;
    setText("#mock-table-count", "0 rows");
    return;
  }
  const headers = ["Sample ID", "Split", "Ωm", "σ₈", "h", "nₛ", "Σmν (eV)", "Flag"];
  const body = visible.map((row) => `<tr><td><code>${escapeHTML(row.id)}</code></td><td><span class="mock-split-pill split-${row.split}">${escapeHTML(row.split)}</span></td><td>${row.omega_m.toFixed(4)}</td><td>${row.sigma8.toFixed(4)}</td><td>${row.h.toFixed(4)}</td><td>${row.n_s.toFixed(4)}</td><td>${row.sum_mnu_ev.toFixed(4)}</td><td><span class="mock-quality-pill ${row.quality_flag === "review" ? "quality-review" : ""}">${escapeHTML(row.quality_flag)}</span></td></tr>`).join("");
  host.innerHTML = `<div class="mock-table-scroll"><table class="mock-data-table"><thead><tr>${headers.map((header) => `<th scope="col">${header}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;
  setText("#mock-table-count", `${rows.length.toLocaleString()} matching rows · showing ${visible.length}`);
  const exportFiltered = $("#mock-export-filtered-csv");
  if (exportFiltered) exportFiltered.disabled = rows.length === 0;
}

function renderArtifactList() {
  if (!activeBundle) return;
  const host = $("#mock-artifact-list");
  if (!host) return;
  const artifacts = listMockArtifacts(activeBundle);
  host.innerHTML = artifacts.map((artifact) => {
    const title = artifact.filename.replaceAll("_", " ").replaceAll(".json", "");
    return `<div class="mock-artifact-row"><div class="mock-artifact-icon">{ }</div><div class="mock-artifact-info"><strong>${escapeHTML(title)}</strong><span>${escapeHTML(formatBytes(artifact.bytes))} · JSON artifact · synthetic</span></div><button class="small-button" data-export-artifact="${escapeHTML(artifact.filename)}">Export</button></div>`;
  }).join("");
  setText("#mock-artifact-count", `${artifacts.length} artifacts available`);
}

function renderRunHistory() {
  if (!activeBundle) return;
  const host = $("#mock-run-history");
  if (!host) return;
  const runs = activeBundle.experiment_history.runs.slice(0, 12);
  host.innerHTML = runs.map((run) => {
    const stateClass = run.status === "warning" ? "mock-run-warning" : run.status === "running-demo" ? "mock-run-active" : "mock-run-complete";
    return `<div class="mock-run-row"><span class="mock-run-dot ${stateClass}"></span><div class="mock-run-main"><strong>${escapeHTML(run.label)}</strong><span>${escapeHTML(run.phase)} · seed ${run.seed}</span></div><div class="mock-run-metrics"><b>${run.toy_score.toFixed(3)}</b><span>toy score</span></div><span class="mock-run-state ${stateClass}">${escapeHTML(run.status.replaceAll("-demo", ""))}</span></div>`;
  }).join("");
}

function renderValidationChecks() {
  if (!activeBundle) return;
  const host = $("#mock-validation-checks");
  if (!host) return;
  const report = activeBundle.validation_report;
  host.innerHTML = report.checks.map((check) => {
    const icon = check.status === "PASS" ? "✓" : check.status === "FAIL" ? "×" : "!";
    return `<div class="mock-validation-row"><span class="mock-validation-icon validation-${check.status.toLowerCase()}">${icon}</span><div><strong>${escapeHTML(check.label)}</strong><p>${escapeHTML(check.detail)}</p></div><span class="mock-validation-status validation-${check.status.toLowerCase()}">${check.status}</span></div>`;
  }).join("");
  setText("#mock-validation-overall", report.overall_status);
  const overall = $("#mock-validation-overall");
  if (overall) overall.className = `mock-validation-overall status-${report.overall_status.toLowerCase()}`;
}

function renderPowerPreview() {
  if (!activeBundle) return;
  const host = $("#mock-spectrum-preview");
  if (!host) return;
  const points = activeBundle.power_spectrum.points;
  const width = 600, height = 190, pad = 16;
  const minY = Math.min(...points.map((point) => point.p_toy));
  const maxY = Math.max(...points.map((point) => point.p_toy));
  const logMin = Math.log10(points[0].k), logMax = Math.log10(points.at(-1).k);
  const xy = points.map((point) => {
    const x = pad + (Math.log10(point.k) - logMin) / (logMax - logMin) * (width - 2 * pad);
    const y = height - pad - (point.p_toy - minY) / Math.max(1e-10, maxY - minY) * (height - 2 * pad);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  host.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Illustrative synthetic power spectrum curve"><defs><linearGradient id="mockSpectrumStroke"><stop offset="0" stop-color="#6cb5ff"/><stop offset="1" stop-color="#a49aff"/></linearGradient></defs><path d="M ${xy.replaceAll(" ", " L ")}" fill="none" stroke="url(#mockSpectrumStroke)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/><line x1="${pad}" x2="${width - pad}" y1="${height - pad}" y2="${height - pad}" stroke="rgba(166,190,232,.25)"/></svg>`;
  setText("#mock-spectrum-note", `${points.length} synthetic points · log-spaced k · arbitrary units`);
}

function renderDataQuality() {
  if (!activeBundle) return;
  const host = $("#mock-quality-grid");
  if (!host) return;
  host.innerHTML = activeBundle.data_quality_report.metrics.map((metric) => `<div class="mock-quality-card"><span>${escapeHTML(metric.label)}</span><strong>${typeof metric.value === "number" ? metric.value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : escapeHTML(metric.value)}</strong><small>${escapeHTML(metric.unit)} · ${escapeHTML(metric.status)}</small></div>`).join("");
}

function wrappedArtifact(name, content, importedAt) {
  return { name, size: JSON.stringify(content).length, importedAt, content };
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const index = (sorted.length - 1) * q;
  const low = Math.floor(index), high = Math.ceil(index);
  return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function downloadText(filename, text, contentType) {
  const blob = new Blob([text], { type: contentType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.hidden = true;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

$("#mock-export-filtered-csv")?.addEventListener("click", () => {
  if (!filteredSampleRows.length) return;
  downloadText("nufield-filtered-synthetic-parameter-samples.csv", toCSV(filteredSampleRows), "text/csv;charset=utf-8");
  announce(`Exported ${filteredSampleRows.length.toLocaleString()} filtered rows.`, "success");
});
