import { $, $$, announce, escapeHTML, formatNumber, setText, prefersReducedMotion } from "./core/dom.js";
import { getState, updateState, updateNestedState, updateImportState, subscribe } from "./core/state.js";
import { initNavigation, navigateTo } from "./components/navigation.js";
import { initModals, openImportModal, closeTopModal } from "./components/modals.js";
import { initModelAdvisor } from "./components/model-advisor.js";
import { parseJSONFiles, summarizeImported } from "./data/importers.js";
import { DEMO_PARAMETERS, buildIllustrativeParameterPoints, buildIllustrativeSpectrum, DEMO_CAVEAT } from "./data/demo-data.js";
import { initExpansionVisuals, setGlobalMotionPaused, getRedshift } from "./visualizations/expansion.js";
import { initCosmicWeb, setCosmicWebMotionPaused } from "./visualizations/cosmic-web.js";
import { initCmbSky, setCmbMotionPaused } from "./visualizations/cmb-sky.js";
import { initLensing, setLensingMotionPaused } from "./visualizations/lensing.js";
import { renderSpectrumChart, renderScatterChart, renderCoverageChart, renderBarChart, exportSvg } from "./visualizations/svg-charts.js";

const illustrativePoints = buildIllustrativeParameterPoints();
let selectedFiles = [];
let chartResizeTimer = 0;

function init() {
  initNavigation(navigateTo);
  initModals();
  initModelAdvisor();
  initExpansionVisuals();
  initCosmicWeb();
  initCmbSky();
  initLensing();
  bindGlobalControls();
  bindImportFlow();
  bindSpectrumControls();
  bindParameterControls();
  bindDismissibleNotices();
  subscribe(renderImportedState);
  renderSpectrum();
  renderParameterSpace();
  renderImportedState(getState());
  setText("#data-source-label", "Illustration mode");
  const motionPreference = window.matchMedia?.("(prefers-reduced-motion: reduce)");
  setAllMotionPaused(motionPreference?.matches ?? prefersReducedMotion());
  motionPreference?.addEventListener?.("change", (event) => {
    setAllMotionPaused(event.matches);
    if (event.matches) announce("Reduced-motion preference detected; all animations are paused.", "info");
  });
  window.addEventListener("resize", () => {
    window.clearTimeout(chartResizeTimer);
    chartResizeTimer = window.setTimeout(() => { renderSpectrum(); renderParameterSpace(); renderCoverageChart($("#coverage-chart"), getState().import.benchmark?.content); }, 100);
  }, { passive: true });
  window.addEventListener("nufield:viewchange", (event) => {
    if (event.detail.view === "spectra") renderSpectrum();
    if (event.detail.view === "parameters") renderParameterSpace();
    if (event.detail.view === "diagnostics") renderCoverageChart($("#coverage-chart"), getState().import.benchmark?.content);
  });
  console.info("νField Observatory frontend ready. All unimported visual values are illustrative.");
}

function bindGlobalControls() {
  $("#motion-toggle")?.addEventListener("click", (event) => {
    const paused = event.currentTarget.getAttribute("aria-pressed") === "true";
    setAllMotionPaused(paused);
    updateState({ motionPaused: paused });
  });
  $$('[data-dismiss]').forEach((button) => {
    button.addEventListener("click", () => {
      const target = $("#" + button.dataset.dismiss);
      target?.classList.add("notice-dismissed");
    });
  });
}

function setAllMotionPaused(paused) {
  setGlobalMotionPaused(paused);
  setCosmicWebMotionPaused(paused);
  setCmbMotionPaused(paused);
  setLensingMotionPaused(paused);
}

function bindImportFlow() {
  const fileInput = $("#json-file-input");
  const hiddenInput = $("#hidden-json-input");
  const dropZone = $("#import-drop-zone");
  fileInput?.addEventListener("change", () => {
    selectedFiles = [...(fileInput.files ?? [])];
    renderSelectedFiles();
  });
  hiddenInput?.addEventListener("change", () => {
    selectedFiles = [...(hiddenInput.files ?? [])];
    renderSelectedFiles();
    if (selectedFiles.length) openImportModal();
  });
  dropZone?.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("drop-zone-active"); });
  dropZone?.addEventListener("dragleave", () => dropZone.classList.remove("drop-zone-active"));
  dropZone?.addEventListener("drop", (event) => {
    event.preventDefault(); dropZone.classList.remove("drop-zone-active");
    selectedFiles = [...(event.dataTransfer?.files ?? [])];
    renderSelectedFiles();
  });
  $("#apply-import")?.addEventListener("click", applyImport);
}

function renderSelectedFiles() {
  const host = $("#import-file-list");
  const apply = $("#apply-import");
  if (!host) return;
  host.replaceChildren();
  if (!selectedFiles.length) {
    host.innerHTML = '<span class="muted-text">No files selected yet.</span>';
    if (apply) apply.disabled = true;
    return;
  }
  selectedFiles.forEach((file) => {
    const row = document.createElement("div");
    row.className = "import-file-row";
    const icon = document.createElement("span"); icon.className = "import-file-icon"; icon.textContent = "JSON";
    const name = document.createElement("span"); name.className = "import-file-name"; name.textContent = file.name;
    const size = document.createElement("span"); size.className = "import-file-size"; size.textContent = `${Math.max(1, Math.round(file.size / 1024))} KB`;
    row.append(icon, name, size); host.append(row);
  });
  if (apply) apply.disabled = false;
  $("#import-error")?.setAttribute("hidden", "");
}

async function applyImport() {
  const button = $("#apply-import");
  const errorBox = $("#import-error");
  if (!selectedFiles.length || !button) return;
  button.disabled = true;
  button.textContent = "Reading files…";
  if (errorBox) { errorBox.hidden = true; errorBox.textContent = ""; }
  try {
    const result = await parseJSONFiles(selectedFiles);
    const previous = getState().import;
    const merged = {
      dataset: result.imported.dataset ?? previous.dataset,
      manifest: result.imported.manifest ?? previous.manifest,
      benchmark: result.imported.benchmark ?? previous.benchmark,
      extras: [...previous.extras, ...result.imported.extras],
    };
    if (result.successCount === 0) {
      const message = result.errors.join("\n") || "No recognized JSON files were imported.";
      if (errorBox) { errorBox.textContent = message; errorBox.hidden = false; }
      announce("No JSON artifacts were imported. Review the file errors.", "error");
      return;
    }
    updateImportState(merged);
    announce(`Imported ${result.successCount} JSON artifact${result.successCount === 1 ? "" : "s"} locally.`, "success");
    if (result.errors.length) announce(`${result.errors.length} file(s) skipped. Check the import dialog for details.`, "error");
    if (errorBox && result.errors.length) { errorBox.textContent = result.errors.join("\n"); errorBox.hidden = false; }
    else closeTopModal();
    selectedFiles = [];
    $("#json-file-input").value = "";
    renderSelectedFiles();
    navigateTo(merged.benchmark ? "diagnostics" : "parameters");
  } catch (error) {
    if (errorBox) { errorBox.textContent = error?.message ?? "Unexpected error while importing files."; errorBox.hidden = false; }
    announce("Could not import the selected JSON files.", "error");
  } finally {
    button.disabled = selectedFiles.length === 0;
    button.textContent = "Import selected files";
  }
}

function bindSpectrumControls() {
  const inputs = ["amplitude", "tilt", "cutoff"];
  inputs.forEach((key) => {
    $(`#spectrum-${key}`)?.addEventListener("input", (event) => {
      const mapKey = key === "cutoff" ? "suppression" : key;
      updateNestedState("spectrum", { [mapKey]: Number(event.currentTarget.value) });
      renderSpectrum();
    });
  });
  $("#spectrum-reset")?.addEventListener("click", () => {
    updateNestedState("spectrum", { amplitude: 1, tilt: 0, suppression: 0.3 });
    $("#spectrum-amplitude").value = "1";
    $("#spectrum-tilt").value = "0";
    $("#spectrum-cutoff").value = "0.3";
    renderSpectrum();
    announce("Schematic curve controls reset.");
  });
  $("#spectrum-export")?.addEventListener("click", () => {
    const ok = exportSvg($("#power-spectrum-chart"), "nufield-illustrative-power-spectrum.svg");
    announce(ok ? "SVG chart exported." : "SVG export failed. Check that your browser allows file downloads.", ok ? "success" : "error");
  });
}

function renderSpectrum() {
  const state = getState();
  const base = buildIllustrativeSpectrum({ amplitude: 1, tilt: 0, suppression: 0.3 });
  const adjusted = buildIllustrativeSpectrum(state.spectrum);
  renderSpectrumChart($("#power-spectrum-chart"), base, adjusted);
  setText("#spectrum-amplitude-out", `${Number(state.spectrum.amplitude).toFixed(2)}×`);
  setText("#spectrum-tilt-out", Number(state.spectrum.tilt).toFixed(2));
  setText("#spectrum-cutoff-out", Number(state.spectrum.suppression).toFixed(2));
}

function bindParameterControls() {
  ["#parameter-x-select", "#parameter-y-select"].forEach((selector) => $(selector)?.addEventListener("change", renderParameterSpace));
}

function renderParameterSpace() {
  const xKey = $("#parameter-x-select")?.value ?? "omega_m";
  const yKey = $("#parameter-y-select")?.value ?? "sigma8";
  const imported = getState().import.benchmark?.content;
  const points = readParameterPoints(imported) ?? illustrativePoints;
  renderScatterChart($("#parameter-space-chart"), points, xKey, yKey, DEMO_PARAMETERS);
  setText("#parameter-data-status", readParameterPoints(imported) ? "IMPORTED POINTS" : "SAMPLE PLACEHOLDER");
  const errors = imported?.median_absolute_error;
  renderBarChart($("#error-summary-table"), errors, { label: "parameter", valueLabel: "Median absolute error" });
  if (!errors) {
    const host = $("#error-summary-table");
    if (host) host.innerHTML = `<div class="empty-state small-empty"><span>↥</span><strong>No benchmark report loaded</strong><p>Import a benchmark_report.json to populate actual reported metrics.</p><button class="secondary-button" id="table-import-button">Import report</button></div>`;
    $("#table-import-button")?.addEventListener("click", openImportModal);
  }
}

function readParameterPoints(report) {
  const candidates = [report?.parameter_points, report?.parameter_samples, report?.parameters_table];
  for (const value of candidates) {
    if (Array.isArray(value) && value.length && value.every((row) => row && typeof row === "object")) {
      const valid = value.filter((row) => Object.keys(DEMO_PARAMETERS).some((key) => Number.isFinite(Number(row[key]))));
      if (valid.length) return valid.map((row, index) => ({ id: String(row.id ?? `import-${index + 1}`), ...row }));
    }
  }
  return null;
}

function renderImportedState(state) {
  const summary = summarizeImported(state.import);
  const dataset = summary.dataset;
  const benchmark = summary.benchmark;
  const manifest = summary.manifest;
  const hasAny = Boolean(dataset || benchmark || manifest || state.import.extras.length);
  setText("#data-source-label", hasAny ? "Imported local files" : "Illustration mode");
  setText("#import-status", hasAny ? "LOADED" : "LOCAL");
  setText("#mode-label", hasAny ? "MIXED / IMPORTED" : "ILLUSTRATIVE MODE");
  setText("#metric-sims", Number.isFinite(Number(dataset?.n_simulations)) ? dataset.n_simulations : "—");
  setText("#metric-sims-note", dataset ? `${dataset.dataset_name ?? "Imported dataset"} · ${dataset.grid_size ? `grid ${dataset.grid_size}³` : "metadata loaded"}` : "Import dataset metadata to populate");
  const validation = benchmark ? (benchmark.split_disjoint === true ? "Split disjoint" : "Report loaded") : "Not loaded";
  setText("#metric-validation", validation);
  setText("#metric-validation-note", benchmark ? `${benchmark.n_train ?? "—"} train · ${benchmark.n_validation_held_out ?? "—"} validation · ${benchmark.n_test ?? "—"} test` : "No imported benchmark report");
  setText("#metric-physics", summary.physicsLevel ?? "Illustrative");
  setText("#metric-physics-note", summary.physicsLevel === "toy" ? "Toy forward model · not a physical constraint" : summary.physicsLevel ? "Level stated by imported artifact" : "Conceptual visuals only");
  renderParameterSpace();
  renderCoverageChart($("#coverage-chart"), benchmark);
  renderSplitSummary(benchmark);
  renderProvenance({ dataset, manifest, benchmark, extras: state.import.extras, hasAny, warning: summary.warning });
}

function renderSplitSummary(report) {
  const host = $("#split-summary");
  if (!host) return;
  if (!report) {
    host.innerHTML = `<div class="empty-state small-empty"><span>◇</span><strong>No split metadata</strong><p>Load a benchmark report to inspect realization IDs and disjointness.</p></div>`;
    setText("#split-status", "UNKNOWN");
    return;
  }
  const status = report.split_disjoint === true ? "PASS" : report.split_disjoint === false ? "REVIEW" : "UNKNOWN";
  setText("#split-status", status);
  const ids = [
    ["Training IDs", report.train_ids],
    ["Validation IDs", report.validation_ids],
    ["Test IDs", report.test_ids],
  ];
  host.replaceChildren();
  ids.forEach(([label, values]) => {
    const row = document.createElement("div"); row.className = "split-row";
    const left = document.createElement("span"); left.textContent = label;
    const right = document.createElement("strong"); right.textContent = Array.isArray(values) ? `${values.length} realizations` : "Not provided";
    row.append(left, right); host.append(row);
  });
  const disjoint = document.createElement("div"); disjoint.className = `split-callout ${report.split_disjoint === true ? "split-callout-pass" : "split-callout-review"}`;
  disjoint.textContent = report.split_disjoint === true ? "Report marks the realization splits as disjoint." : report.split_disjoint === false ? "Report indicates overlap; inspect split construction." : "The report does not state whether splits are disjoint.";
  host.append(disjoint);
}

function renderProvenance({ dataset, manifest, benchmark, extras, hasAny, warning }) {
  const host = $("#provenance-content");
  if (!host) return;
  setText("#provenance-source-label", hasAny ? "IMPORTED JSON" : "NO FILE");
  if (!hasAny) {
    host.innerHTML = `<div class="empty-state"><span>ⓘ</span><strong>Import a νField JSON artifact</strong><p>Dataset metadata, manifests, or benchmark reports can be opened locally. Files stay in this browser session.</p><button class="secondary-button" data-open-import>Choose JSON files</button></div>`;
    host.querySelector("[data-open-import]")?.addEventListener("click", openImportModal);
    return;
  }
  const items = [
    ["Dataset metadata", dataset?.dataset_name ?? "Not imported", dataset?.physics_level ?? "—", dataset?.scientific_warning ?? ""],
    ["Dataset manifest", manifest?.schema_version ? `Schema ${manifest.schema_version}` : (manifest ? "Manifest loaded" : "Not imported"), manifest?.physics_level ?? "—", manifest?.scientific_warning ?? ""],
    ["Benchmark report", benchmark?.method?.algorithm ?? (benchmark ? "Benchmark loaded" : "Not imported"), benchmark?.physics_level ?? "—", benchmark?.scientific_warning ?? ""],
    ...extras.map((item) => ["Additional JSON", item.name, "Unclassified", ""]),
  ];
  host.replaceChildren();
  if (warning) {
    const warningBox = document.createElement("div"); warningBox.className = "scientific-warning-box";
    const icon = document.createElement("span"); icon.className = "warning-box-icon"; icon.textContent = "!";
    const copy = document.createElement("div"); const strong = document.createElement("strong"); strong.textContent = "Scientific warning from artifact";
    const paragraph = document.createElement("p"); paragraph.textContent = warning;
    copy.append(strong, paragraph); warningBox.append(icon, copy); host.append(warningBox);
  }
  const table = document.createElement("div"); table.className = "provenance-list";
  items.forEach(([label, name, level, caveat]) => {
    const row = document.createElement("div"); row.className = "provenance-row";
    const type = document.createElement("span"); type.className = "provenance-type"; type.textContent = label;
    const info = document.createElement("div"); info.className = "provenance-info";
    const title = document.createElement("strong"); title.textContent = name;
    const sub = document.createElement("span"); sub.textContent = `Physics level: ${level}`;
    info.append(title, sub);
    const warningText = document.createElement("p"); warningText.className = "provenance-caveat"; warningText.textContent = caveat || (level === "toy" ? DEMO_CAVEAT : "Values displayed as provided; frontend has not independently validated this artifact.");
    row.append(type, info, warningText); table.append(row);
  });
  host.append(table);
}

function bindDismissibleNotices() {
  $$('[data-dismiss]').forEach((button) => button.addEventListener("click", () => $("#" + button.dataset.dismiss)?.classList.add("notice-dismissed")));
}

init();
