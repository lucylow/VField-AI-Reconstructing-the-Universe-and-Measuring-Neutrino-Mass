import { $, setText, formatNumber } from "../core/dom.js";
import { scaleFactorFromRedshift, formatEpoch } from "../core/cosmology.js";

const canvases = new Map();
let raf = 0;
let frameTime = 0;
let logicalTime = 0;
let currentZ = 0;
let paused = false;
let playing = false;

export function initExpansionVisuals() {
  ["#overview-expansion-canvas", "#detail-expansion-canvas"].forEach((selector) => {
    const canvas = $(selector);
    if (!canvas) return;
    canvases.set(selector, { canvas, ctx: canvas.getContext("2d"), width: 1, height: 1 });
  });
  resizeCanvases();
  window.addEventListener("resize", resizeCanvases, { passive: true });
  const overviewSlider = $("#overview-redshift-slider");
  const detailSlider = $("#detail-redshift-slider");
  overviewSlider?.addEventListener("input", () => setRedshift(Number(overviewSlider.value), "overview"));
  detailSlider?.addEventListener("input", () => setRedshift(Number(detailSlider.value), "detail"));
  document.querySelectorAll("[data-redshift-preset]").forEach((button) => {
    button.addEventListener("click", () => setRedshift(Number(button.dataset.redshiftPreset), "preset"));
  });
  $("#expansion-reset")?.addEventListener("click", () => { stopPlayback(); setRedshift(0, "detail"); });
  $("#expansion-play")?.addEventListener("click", togglePlayback);
  $("#motion-toggle")?.addEventListener("click", (event) => {
    paused = !paused;
    event.currentTarget.setAttribute("aria-pressed", String(paused));
    event.currentTarget.textContent = paused ? "▶" : "Ⅱ";
    event.currentTarget.title = paused ? "Resume animations" : "Pause animations";
  });
  setRedshift(0, "overview");
  window.addEventListener("nufield:viewchange", drawAll);
  tick();
}

export function setGlobalMotionPaused(value) {
  paused = Boolean(value);
  const button = $("#motion-toggle");
  button?.setAttribute("aria-pressed", String(paused));
  if (button) {
    button.textContent = paused ? "▶" : "Ⅱ";
    button.setAttribute("aria-label", paused ? "Resume animations" : "Pause animations");
    button.title = paused ? "Resume animations" : "Pause animations";
  }
}

export function getRedshift() { return currentZ; }

function setRedshift(z, origin = "overview") {
  currentZ = Math.min(10, Math.max(0, Number.isFinite(z) ? z : 0));
  const a = scaleFactorFromRedshift(currentZ);
  $("#overview-redshift-slider") && ($("#overview-redshift-slider").value = String(currentZ));
  $("#detail-redshift-slider") && ($("#detail-redshift-slider").value = String(currentZ));
  setText("#overview-redshift-value", `z ${currentZ.toFixed(1)}`);
  setText("#overview-redshift", `z = ${currentZ.toFixed(1)}`);
  setText("#overview-scale-factor", `a = ${a.toFixed(3)}`);
  setText("#detail-redshift-label", currentZ.toFixed(1));
  setText("#detail-redshift-readout", currentZ.toFixed(1));
  setText("#detail-scale-readout", a.toFixed(3));
  setText("#overview-epoch", `z = ${currentZ.toFixed(1)} · ${formatEpoch(currentZ)}`);
  setText("#metric-scale", a.toFixed(3));
  setText("#equation-z", currentZ.toFixed(1));
  setText("#equation-a", a.toFixed(3));
  setText("#equation-spacing", `${Math.round(a * 100)}%`);
  document.querySelectorAll("[data-redshift-preset]").forEach((button) => {
    const selected = Math.abs(Number(button.dataset.redshiftPreset) - currentZ) < 0.05;
    button.setAttribute("aria-pressed", String(selected));
  });
  const scaleText = $("#metric-scale");
  if (scaleText) scaleText.setAttribute("aria-label", `Illustrative scale factor ${formatNumber(a)}`);
  if (origin === "detail") $("#detail-redshift-slider")?.focus({ preventScroll: true });
  drawAll();
}

function togglePlayback() {
  playing = !playing;
  const button = $("#expansion-play");
  button?.setAttribute("aria-pressed", String(playing));
  if (button) button.textContent = playing ? "Ⅱ Pause" : "▶ Play";
  if (!playing) return;
  if (currentZ >= 10) setRedshift(0, "detail");
}

function stopPlayback() {
  playing = false;
  const button = $("#expansion-play");
  button?.setAttribute("aria-pressed", "false");
  if (button) button.textContent = "▶ Play";
}

function resizeCanvases() {
  for (const item of canvases.values()) {
    const rect = item.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    item.width = Math.max(1, Math.floor(rect.width * dpr));
    item.height = Math.max(1, Math.floor(rect.height * dpr));
    item.canvas.width = item.width;
    item.canvas.height = item.height;
    item.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  drawAll();
}

function drawAll() {
  for (const [selector, item] of canvases.entries()) {
    const view = item.canvas.closest(".view");
    if (view && !view.classList.contains("active")) continue;
    drawExpansion(item, selector.includes("overview"));
  }
}

function drawExpansion(item, compact) {
  const { canvas, ctx } = item;
  if (!ctx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  if (width < 2 || height < 2) return;
  ctx.clearRect(0, 0, width, height);
  const background = ctx.createLinearGradient(0, 0, width, height);
  background.addColorStop(0, "#071024");
  background.addColorStop(0.56, "#0c1931");
  background.addColorStop(1, "#101b36");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);
  drawStars(ctx, width, height, compact ? 54 : 100);

  const centerX = width / 2;
  const centerY = height / 2;
  const scaleFactor = scaleFactorFromRedshift(currentZ);
  const displayScale = 0.36 + (1 - scaleFactor) * 1.28;
  const baseRadius = Math.min(width, height) * (compact ? 0.19 : 0.23);
  const orbitRadius = baseRadius * displayScale;

  ctx.save();
  ctx.translate(centerX, centerY);
  ctx.strokeStyle = "rgba(90, 162, 255, 0.10)";
  ctx.lineWidth = 1;
  for (let i = 1; i <= 4; i += 1) {
    ctx.beginPath(); ctx.arc(0, 0, orbitRadius * (i / 4), 0, Math.PI * 2); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(120, 165, 241, 0.12)";
  ctx.setLineDash([3, 8]);
  for (let i = 0; i < 8; i += 1) {
    const a = i * Math.PI / 4;
    ctx.beginPath();
    ctx.moveTo(Math.cos(a) * baseRadius * 0.2, Math.sin(a) * baseRadius * 0.2);
    ctx.lineTo(Math.cos(a) * orbitRadius * 1.43, Math.sin(a) * orbitRadius * 1.43);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  const points = markerAngles.map((angle, index) => {
    const drift = paused ? 0 : Math.sin(logicalTime * 0.00035 + index) * 2.4;
    return { x: Math.cos(angle) * orbitRadius * 0.78 + drift, y: Math.sin(angle) * orbitRadius * 0.78 + drift, angle };
  });
  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    ctx.strokeStyle = `rgba(111, 174, 255, ${0.18 + i % 3 * 0.04})`;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(point.x, point.y); ctx.stroke();
    const glow = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, compact ? 13 : 18);
    glow.addColorStop(0, "rgba(136, 194, 255, .32)");
    glow.addColorStop(1, "rgba(136, 194, 255, 0)");
    ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(point.x, point.y, compact ? 13 : 18, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = i % 4 === 0 ? "#a6dcff" : "#668fff";
    ctx.beginPath(); ctx.arc(point.x, point.y, compact ? 2.5 : 3.2, 0, Math.PI * 2); ctx.fill();
    if (!compact && i % 2 === 0) {
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillStyle = "rgba(202, 218, 255, .60)";
      ctx.fillText(`G${String(i + 1).padStart(2, "0")}`, point.x + 8, point.y - 7);
    }
  }
  const centerGlow = ctx.createRadialGradient(0, 0, 0, 0, 0, baseRadius * 0.9);
  centerGlow.addColorStop(0, "rgba(107, 93, 246, .28)");
  centerGlow.addColorStop(1, "rgba(107, 93, 246, 0)");
  ctx.fillStyle = centerGlow; ctx.beginPath(); ctx.arc(0, 0, baseRadius * 0.9, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "rgba(202, 213, 255, .84)";
  ctx.beginPath(); ctx.arc(0, 0, compact ? 2.2 : 3, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  if (!compact) drawScaleRuler(ctx, width, height, scaleFactor);
}

const markerAngles = Array.from({ length: 16 }, (_, i) => (i / 16) * Math.PI * 2 + 0.08);

function drawStars(ctx, width, height, count) {
  for (let i = 0; i < count; i += 1) {
    const x = seeded(i * 17 + 13) * width;
    const y = seeded(i * 29 + 5) * height;
    const radius = seeded(i * 41 + 7) > 0.92 ? 1.25 : 0.55;
    const alpha = 0.20 + seeded(i * 11 + 3) * 0.55;
    ctx.fillStyle = `rgba(190, 210, 255, ${alpha})`;
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();
  }
}

function drawScaleRuler(ctx, width, height, scaleFactor) {
  const left = 26;
  const y = height - 26;
  const trackWidth = Math.min(210, width * 0.38);
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "rgba(205, 218, 245, .54)";
  ctx.fillText("RELATIVE SCALE", left, y - 12);
  ctx.fillStyle = "rgba(140, 164, 218, .22)";
  ctx.fillRect(left, y, trackWidth, 3);
  ctx.fillStyle = "#72b6ff";
  ctx.fillRect(left, y, trackWidth * scaleFactor, 3);
  ctx.fillStyle = "rgba(205, 218, 245, .75)";
  ctx.fillText(`${Math.round(scaleFactor * 100)}%`, left + trackWidth + 9, y + 4);
}

function seeded(value) {
  const x = Math.sin(value * 78.233) * 43758.5453123;
  return x - Math.floor(x);
}

function tick(time = 0) {
  const delta = frameTime ? Math.min(50, time - frameTime) : 0;
  frameTime = time;
  const pageVisible = document.visibilityState !== "hidden";
  if (!paused && pageVisible) logicalTime += delta;
  if (playing && !paused && pageVisible) {
    const next = currentZ + delta * 0.0019;
    if (next >= 10) stopPlayback();
    else setRedshift(next, "playback");
  } else if (!paused && pageVisible) {
    // Only visible views are drawn, avoiding unnecessary canvas work in hidden tabs.
    drawAll();
  }
  raf = window.requestAnimationFrame(tick);
}

export function disposeExpansion() {
  if (raf) window.cancelAnimationFrame(raf);
  window.removeEventListener("resize", resizeCanvases);
  window.removeEventListener("nufield:viewchange", drawAll);
}
