import { escapeHTML } from "../core/dom.js";

const NS = "http://www.w3.org/2000/svg";

export function renderSpectrumChart(host, baseline, adjusted) {
  if (!host) return;
  const width = Math.max(host.clientWidth || 640, 320);
  const height = 340;
  const margin = { top: 18, right: 26, bottom: 52, left: 62 };
  const all = [...baseline, ...adjusted];
  const xMin = Math.min(...all.map((d) => d.k));
  const xMax = Math.max(...all.map((d) => d.k));
  const yMin = Math.min(...all.map((d) => d.p));
  const yMax = Math.max(...all.map((d) => d.p));
  const x = (value) => margin.left + (Math.log10(value) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - (Math.log10(value) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)) * (height - margin.top - margin.bottom);
  const svg = createSvg(width, height, "Power spectrum comparison");
  addChartBackground(svg, width, height, margin);
  addLogGrid(svg, { width, height, margin, xMin, xMax, yMin, yMax, x, y });
  addPath(svg, baseline.map((d) => [x(d.k), y(d.p)]), "spectrum-path spectrum-baseline");
  addPath(svg, adjusted.map((d) => [x(d.k), y(d.p)]), "spectrum-path spectrum-adjusted");
  addAxisLabels(svg, width, height, margin, "Wave number k (arbitrary scale)", "Power P(k) · normalized");
  host.replaceChildren(svg);
}

export function renderScatterChart(host, points, xKey, yKey, parameterMeta) {
  if (!host) return;
  const width = Math.max(host.clientWidth || 560, 320);
  const height = 360;
  const margin = { top: 18, right: 28, bottom: 52, left: 64 };
  const xMeta = parameterMeta[xKey];
  const yMeta = parameterMeta[yKey];
  const xValues = points.map((point) => Number(point[xKey])).filter(Number.isFinite);
  const yValues = points.map((point) => Number(point[yKey])).filter(Number.isFinite);
  const xMin = Math.min(...xValues); const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues); const yMax = Math.max(...yValues);
  const xPad = (xMax - xMin || 1) * 0.08;
  const yPad = (yMax - yMin || 1) * 0.08;
  const sx = (value) => margin.left + (value - (xMin - xPad)) / ((xMax + xPad) - (xMin - xPad)) * (width - margin.left - margin.right);
  const sy = (value) => height - margin.bottom - (value - (yMin - yPad)) / ((yMax + yPad) - (yMin - yPad)) * (height - margin.top - margin.bottom);
  const svg = createSvg(width, height, `Scatter plot of ${xMeta?.label ?? xKey} versus ${yMeta?.label ?? yKey}`);
  addChartBackground(svg, width, height, margin);
  addLinearGrid(svg, { width, height, margin, xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad, sx, sy });
  const group = svgEl("g", { class: "scatter-points" });
  points.forEach((point, index) => {
    const cx = sx(Number(point[xKey]));
    const cy = sy(Number(point[yKey]));
    if (![cx, cy].every(Number.isFinite)) return;
    const circle = svgEl("circle", { cx, cy, r: index % 9 === 0 ? 3.5 : 2.5, class: "scatter-point", tabindex: 0, role: "img", "aria-label": `${point.id}: ${xMeta?.label ?? xKey} ${Number(point[xKey]).toFixed(3)}, ${yMeta?.label ?? yKey} ${Number(point[yKey]).toFixed(3)}` });
    const title = svgEl("title"); title.textContent = `${point.id}\n${xMeta?.label ?? xKey}: ${Number(point[xKey]).toFixed(4)}\n${yMeta?.label ?? yKey}: ${Number(point[yKey]).toFixed(4)}`;
    circle.append(title); group.append(circle);
  });
  svg.append(group);
  addAxisLabels(svg, width, height, margin, xMeta?.label ?? xKey, yMeta?.label ?? yKey);
  host.replaceChildren(svg);
}

export function renderCoverageChart(host, report) {
  if (!host) return;
  const diagnostic = report?.coverage_diagnostic;
  const parameters = report?.parameter_names ?? [];
  const levels = ["0.50", "0.68", "0.95"];
  if (!diagnostic || !parameters.length) {
    host.innerHTML = `<div class="empty-state small-empty"><span>▥</span><strong>No coverage results</strong><p>Import a benchmark report with coverage_diagnostic.</p></div>`;
    return;
  }
  const series = levels.filter((level) => Array.isArray(diagnostic[`coverage_${level}`]));
  if (!series.length) {
    host.innerHTML = `<div class="empty-state small-empty"><span>▥</span><strong>Coverage arrays unavailable</strong><p>The selected report does not include recognized coverage levels.</p></div>`;
    return;
  }
  const width = Math.max(host.clientWidth || 640, 320);
  const rowH = 42;
  const height = Math.max(260, parameters.length * rowH + 72);
  const margin = { top: 22, right: 32, bottom: 34, left: 110 };
  const plotWidth = width - margin.left - margin.right;
  const svg = createSvg(width, height, "Reported empirical coverage by parameter");
  addChartBackground(svg, width, height, margin);
  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    const x = margin.left + tick * plotWidth;
    svg.append(svgEl("line", { x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, class: "chart-grid-line" }));
    const label = svgEl("text", { x, y: height - margin.bottom + 18, class: "chart-tick-label", "text-anchor": "middle" }); label.textContent = `${Math.round(tick * 100)}%`; svg.append(label);
  });
  parameters.forEach((name, row) => {
    const y = margin.top + row * rowH + 11;
    const label = svgEl("text", { x: margin.left - 10, y: y + 4, class: "chart-tick-label", "text-anchor": "end" }); label.textContent = name.replaceAll("_", " "); svg.append(label);
    series.forEach((level, seriesIndex) => {
      const values = diagnostic[`coverage_${level}`];
      const value = Number(values[row]);
      if (!Number.isFinite(value)) return;
      const groupY = y + (seriesIndex - (series.length - 1) / 2) * 8;
      svg.append(svgEl("line", { x1: margin.left, y1: groupY, x2: margin.left + plotWidth, y2: groupY, class: "coverage-track" }));
      svg.append(svgEl("circle", { cx: margin.left + Math.max(0, Math.min(1, value)) * plotWidth, cy: groupY, r: 4.2, class: `coverage-dot coverage-level-${level.replace(".", "")}`, tabindex: 0, role: "img", "aria-label": `${name}, nominal ${level}, reported coverage ${value}` }));
      const title = svgEl("title"); title.textContent = `${name}: nominal ${level}, reported ${Number(value).toFixed(3)}`; svg.lastChild.append(title);
    });
  });
  const legend = svgEl("g", { class: "coverage-legend" });
  series.forEach((level, index) => {
    const y = height - 9;
    const x = margin.left + index * 115;
    legend.append(svgEl("circle", { cx: x, cy: y - 1, r: 4, class: `coverage-dot coverage-level-${level.replace(".", "")}` }));
    const label = svgEl("text", { x: x + 8, y: y + 3, class: "chart-legend-text" }); label.textContent = `Nominal ${Math.round(Number(level) * 100)}%`; legend.append(label);
  });
  svg.append(legend);
  host.replaceChildren(svg);
}

export function renderBarChart(host, data, { label = "Parameter", valueLabel = "Metric" } = {}) {
  if (!host) return;
  const entries = Object.entries(data ?? {}).filter(([, value]) => Number.isFinite(Number(value)));
  if (!entries.length) {
    host.innerHTML = `<div class="empty-state small-empty"><span>▥</span><strong>No numeric metrics available</strong><p>Import a report with a numeric summary.</p></div>`;
    return;
  }
  const width = Math.max(host.clientWidth || 480, 320);
  const rowHeight = 35;
  const height = Math.max(190, entries.length * rowHeight + 58);
  const margin = { top: 16, right: 68, bottom: 26, left: 132 };
  const plotWidth = width - margin.left - margin.right;
  const maxValue = Math.max(...entries.map(([, value]) => Math.abs(Number(value))), 1e-9);
  const svg = createSvg(width, height, `${valueLabel} by ${label}`);
  addChartBackground(svg, width, height, margin);
  entries.forEach(([name, value], index) => {
    const y = margin.top + index * rowHeight + 5;
    const numeric = Number(value);
    const barWidth = Math.abs(numeric) / maxValue * plotWidth;
    const label = svgEl("text", { x: margin.left - 9, y: y + 12, class: "chart-tick-label", "text-anchor": "end" }); label.textContent = name.replaceAll("_", " "); svg.append(label);
    svg.append(svgEl("rect", { x: margin.left, y, width: barWidth, height: 17, rx: 4, class: numeric < 0 ? "bar-negative" : "bar-positive" }));
    const valueText = svgEl("text", { x: Math.min(width - 4, margin.left + barWidth + 7), y: y + 12, class: "chart-value-label", "text-anchor": margin.left + barWidth + 60 > width ? "end" : "start" }); valueText.textContent = formatMetric(numeric); svg.append(valueText);
  });
  host.replaceChildren(svg);
}

export function exportSvg(host, fileName = "nufield-chart.svg") {
  const svg = host?.querySelector("svg");
  if (!svg) return false;
  let url = "";
  let anchor = null;
  try {
    if (typeof XMLSerializer !== "function" || typeof URL?.createObjectURL !== "function") return false;
    const copy = svg.cloneNode(true);
    copy.setAttribute("xmlns", NS);
    const source = new XMLSerializer().serializeToString(copy);
    const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
    url = URL.createObjectURL(blob);
    anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = String(fileName).replace(/[\\/\0]/g, "-");
    anchor.hidden = true;
    document.body.append(anchor);
    anchor.click();
    return true;
  } catch {
    return false;
  } finally {
    anchor?.remove?.();
    if (url) {
      const release = () => { try { URL.revokeObjectURL(url); } catch { /* best-effort cleanup */ } };
      try { window.setTimeout(release, 1000); } catch { release(); }
    }
  }
}

function createSvg(width, height, titleText) {
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, role: "img", "aria-label": titleText, preserveAspectRatio: "xMidYMid meet" });
  const title = svgEl("title"); title.textContent = titleText; svg.append(title);
  return svg;
}

function addChartBackground(svg, width, height, margin) {
  svg.append(svgEl("rect", { x: 0, y: 0, width, height, rx: 10, class: "chart-background" }));
  svg.append(svgEl("line", { x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom, class: "chart-axis-line" }));
  svg.append(svgEl("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, class: "chart-axis-line" }));
}

function addLogGrid(svg, config) {
  const { width, height, margin, xMin, xMax, yMin, yMax, x, y } = config;
  const xStart = Math.ceil(Math.log10(xMin));
  const xEnd = Math.floor(Math.log10(xMax));
  for (let exp = xStart; exp <= xEnd; exp += 1) {
    const val = 10 ** exp;
    const px = x(val);
    svg.append(svgEl("line", { x1: px, y1: margin.top, x2: px, y2: height - margin.bottom, class: "chart-grid-line" }));
    const label = svgEl("text", { x: px, y: height - margin.bottom + 18, class: "chart-tick-label", "text-anchor": "middle" }); label.textContent = `10${superscript(exp)}`; svg.append(label);
  }
  const yTicks = 5;
  const logMin = Math.log10(yMin); const logMax = Math.log10(yMax);
  for (let index = 0; index <= yTicks; index += 1) {
    const val = 10 ** (logMin + (logMax - logMin) * index / yTicks);
    const py = y(val);
    svg.append(svgEl("line", { x1: margin.left, y1: py, x2: width - margin.right, y2: py, class: "chart-grid-line" }));
    const label = svgEl("text", { x: margin.left - 8, y: py + 4, class: "chart-tick-label", "text-anchor": "end" }); label.textContent = formatMetric(val); svg.append(label);
  }
}

function addLinearGrid(svg, config) {
  const { width, height, margin, xMin, xMax, yMin, yMax, sx, sy } = config;
  for (let index = 0; index <= 4; index += 1) {
    const xVal = xMin + (xMax - xMin) * index / 4;
    const x = sx(xVal);
    svg.append(svgEl("line", { x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, class: "chart-grid-line" }));
    const label = svgEl("text", { x, y: height - margin.bottom + 18, class: "chart-tick-label", "text-anchor": "middle" }); label.textContent = formatMetric(xVal); svg.append(label);
    const yVal = yMin + (yMax - yMin) * index / 4;
    const y = sy(yVal);
    svg.append(svgEl("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "chart-grid-line" }));
    const yLabel = svgEl("text", { x: margin.left - 8, y: y + 4, class: "chart-tick-label", "text-anchor": "end" }); yLabel.textContent = formatMetric(yVal); svg.append(yLabel);
  }
}

function addAxisLabels(svg, width, height, margin, xLabel, yLabel) {
  const x = svgEl("text", { x: margin.left + (width - margin.left - margin.right) / 2, y: height - 9, class: "chart-axis-label", "text-anchor": "middle" }); x.textContent = xLabel; svg.append(x);
  const y = svgEl("text", { x: 16, y: margin.top + (height - margin.top - margin.bottom) / 2, class: "chart-axis-label", "text-anchor": "middle", transform: `rotate(-90 16 ${margin.top + (height - margin.top - margin.bottom) / 2})` }); y.textContent = yLabel; svg.append(y);
}

function addPath(svg, points, className) {
  const d = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  svg.append(svgEl("path", { d, class: className, fill: "none", "vector-effect": "non-scaling-stroke" }));
}

function svgEl(tag, attributes = {}) {
  const element = document.createElementNS(NS, tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  return element;
}

function formatMetric(value) {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 100 || (abs > 0 && abs < 0.001)) return value.toExponential(1);
  return value.toFixed(2).replace(/\.00$/, "");
}

function superscript(value) {
  const map = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  return String(value).split("").map((digit) => map[digit] ?? digit).join("");
}
