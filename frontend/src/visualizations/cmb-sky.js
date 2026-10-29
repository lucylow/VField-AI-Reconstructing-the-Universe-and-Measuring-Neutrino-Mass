import { $, setText } from "../core/dom.js";
import { clamp, mulberry32 } from "../data/demo-data.js";

const TEXTURE_WIDTH = 480;
const TEXTURE_HEIGHT = 240;
let canvas;
let ctx;
let histogramCanvas;
let histogramCtx;
let textureCanvas;
let textureCtx;
let width = 1;
let height = 1;
let seed = 71421;
let contrast = 1;
let showGraticule = true;
let scanning = false;
let globalPaused = false;
let scanPhase = 0;
let lastFrame = 0;
let raf = 0;
let resizeObserver;
let pixels = [];
let textureStats = { mean: 0, spread: 0, count: 0, histogram: [] };

export function initCmbSky() {
  canvas = $("#cmb-sky-canvas");
  histogramCanvas = $("#cmb-histogram-canvas");
  if (!canvas || !histogramCanvas) return;
  ctx = canvas.getContext("2d");
  histogramCtx = histogramCanvas.getContext("2d");
  textureCanvas = document.createElement("canvas");
  textureCanvas.width = TEXTURE_WIDTH;
  textureCanvas.height = TEXTURE_HEIGHT;
  textureCtx = textureCanvas.getContext("2d", { willReadFrequently: true });
  $("#cmb-regenerate")?.addEventListener("click", regenerate);
  $("#cmb-scan-toggle")?.addEventListener("click", toggleScan);
  $("#cmb-contrast")?.addEventListener("input", (event) => {
    contrast = Number(event.currentTarget.value);
    setText("#cmb-contrast-output", `${contrast.toFixed(1)}×`);
    setText("#cmb-current-contrast", `${contrast.toFixed(1)}×`);
    paintTexture();
    renderStats();
  });
  $("#cmb-graticule")?.addEventListener("change", (event) => { showGraticule = event.currentTarget.checked; draw(); });
  if ("ResizeObserver" in window) {
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resizeObserver.observe(histogramCanvas);
  } else window.addEventListener("resize", resize, { passive: true });
  regenerate(false);
  window.addEventListener("nufield:viewchange", onViewChange);
  tick();
}

function regenerate(changeSeed = true) {
  if (changeSeed) seed = Math.floor(Math.random() * 90000) + 10000;
  setText("#cmb-seed-label", `SEED ${seed}`);
  const random = mulberry32(seed);
  const data = new Float32Array(TEXTURE_WIDTH * TEXTURE_HEIGHT);
  const imageData = textureCtx.createImageData(TEXTURE_WIDTH, TEXTURE_HEIGHT);
  const histogram = new Array(31).fill(0);
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let py = 0; py < TEXTURE_HEIGHT; py += 1) {
    const v = ((py / (TEXTURE_HEIGHT - 1)) * 2) - 1;
    for (let px = 0; px < TEXTURE_WIDTH; px += 1) {
      const u = ((px / (TEXTURE_WIDTH - 1)) * 2) - 1;
      const index = py * TEXTURE_WIDTH + px;
      const inMap = u * u + v * v <= 1.0;
      if (!inMap) { data[index] = Number.NaN; continue; }
      // Multi-scale smooth texture + seeded local texture, solely for explanatory UI.
      const low = Math.sin(u * 8.4 + Math.cos(v * 5.2)) * 0.52;
      const medium = Math.cos(u * 20.1 - v * 11.4) * 0.24 + Math.sin(v * 25.7 + u * 4.5) * 0.17;
      const fine = Math.sin(u * 58.2 + v * 37.1) * 0.085 + Math.cos(v * 70.3 - u * 22.2) * 0.045;
      const local = (random() - 0.5) * 0.12;
      const value = low + medium + fine + local;
      data[index] = value;
      sum += value;
      sumSq += value * value;
      count += 1;
      const bucket = clamp(Math.floor((value + 1.25) / 2.5 * histogram.length), 0, histogram.length - 1);
      histogram[bucket] += 1;
    }
  }
  pixels = Array.from(data);
  textureStats = { mean: count ? sum / count : 0, spread: count ? Math.sqrt(Math.max(0, sumSq / count - (sum / count) ** 2)) : 0, count, histogram };
  paintTexture();
  renderStats();
  draw();
}

function paintTexture() {
  if (!textureCtx) return;
  const image = textureCtx.createImageData(TEXTURE_WIDTH, TEXTURE_HEIGHT);
  const { mean, spread } = textureStats;
  const scale = Math.max(0.15, spread || 0.6);
  for (let index = 0; index < pixels.length; index += 1) {
    const value = pixels[index];
    const offset = index * 4;
    if (!Number.isFinite(value)) {
      image.data[offset] = 4; image.data[offset + 1] = 9; image.data[offset + 2] = 19; image.data[offset + 3] = 0;
      continue;
    }
    const normalized = clamp(((value - mean) / scale) * contrast, -1.45, 1.45);
    const color = divergingColor(normalized);
    image.data[offset] = color[0];
    image.data[offset + 1] = color[1];
    image.data[offset + 2] = color[2];
    image.data[offset + 3] = 255;
  }
  textureCtx.putImageData(image, 0, 0);
}

function divergingColor(value) {
  // Cobalt -> cool gray -> warm coral; no calibrated temperature mapping is implied.
  const negative = [20, 53, 126];
  const neutral = [222, 225, 224];
  const positive = [174, 37, 69];
  const t = Math.min(1, Math.abs(value));
  const start = value < 0 ? neutral : neutral;
  const end = value < 0 ? negative : positive;
  const eased = t ** 0.72;
  return start.map((channel, index) => Math.round(channel + (end[index] - channel) * eased));
}

function renderStats() {
  setText("#cmb-pixel-count", textureStats.count.toLocaleString());
  setText("#cmb-mean-offset", textureStats.mean.toFixed(3));
  setText("#cmb-spread", textureStats.spread.toFixed(3));
  setText("#cmb-current-contrast", `${contrast.toFixed(1)}×`);
  drawHistogram();
}

function resize() {
  if (!canvas || !ctx || !histogramCanvas || !histogramCtx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  width = Math.max(1, rect.width);
  height = Math.max(1, rect.height);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const histRect = histogramCanvas.getBoundingClientRect();
  histogramCanvas.width = Math.max(1, Math.round(histRect.width * dpr));
  histogramCanvas.height = Math.max(1, Math.round(histRect.height * dpr));
  histogramCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
  drawHistogram();
}

function draw() {
  if (!ctx || width < 2 || height < 2) return;
  ctx.clearRect(0, 0, width, height);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rectWidth = Math.min(width * 0.9, height * 2.0);
  const rectHeight = rectWidth / 2;
  const x = (width - rectWidth) / 2;
  const y = (height - rectHeight) / 2;
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(width / 2, height / 2, rectWidth * 0.5, rectHeight * 0.5, 0, 0, Math.PI * 2);
  ctx.clip();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(textureCanvas, x, y, rectWidth, rectHeight);
  if (showGraticule) drawGraticule(x, y, rectWidth, rectHeight);
  ctx.restore();
  ctx.beginPath(); ctx.ellipse(width / 2, height / 2, rectWidth * 0.5, rectHeight * 0.5, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(194, 216, 255, .43)"; ctx.lineWidth = 1; ctx.stroke();
  // Slight edge shading helps separate the map from the dark stage.
  const shade = ctx.createRadialGradient(width / 2, height / 2, rectHeight * 0.3, width / 2, height / 2, Math.max(width, height) * 0.68);
  shade.addColorStop(0, "rgba(0, 0, 0, 0)"); shade.addColorStop(1, "rgba(0, 0, 0, .36)");
  ctx.fillStyle = shade; ctx.fillRect(0, 0, width, height);
  void dpr;
}

function drawGraticule(x, y, w, h) {
  ctx.save();
  ctx.strokeStyle = "rgba(240, 243, 255, .23)";
  ctx.lineWidth = 0.65;
  ctx.setLineDash([3, 4]);
  [-0.65, -0.32, 0, 0.32, 0.65].forEach((fraction) => {
    const yy = y + h * (0.5 + fraction * 0.5);
    const normalized = (yy - (y + h / 2)) / (h / 2);
    const extent = Math.sqrt(Math.max(0, 1 - normalized * normalized)) * w / 2;
    ctx.beginPath(); ctx.moveTo(x + w / 2 - extent, yy); ctx.lineTo(x + w / 2 + extent, yy); ctx.stroke();
  });
  [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75].forEach((fraction) => {
    ctx.beginPath();
    for (let i = 0; i <= 45; i += 1) {
      const v = -1 + i / 45 * 2;
      const boundary = Math.sqrt(Math.max(0, 1 - v * v));
      const px = x + w / 2 + fraction * boundary * w / 2;
      const py = y + h / 2 + v * h / 2;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  });
  ctx.restore();
}

function drawHistogram() {
  if (!histogramCtx || !histogramCanvas) return;
  const w = histogramCanvas.width / Math.min(window.devicePixelRatio || 1, 2);
  const h = histogramCanvas.height / Math.min(window.devicePixelRatio || 1, 2);
  if (w < 2 || h < 2) return;
  histogramCtx.clearRect(0, 0, w, h);
  const margin = { left: 3, right: 3, top: 12, bottom: 19 };
  const max = Math.max(...textureStats.histogram, 1);
  const plotW = w - margin.left - margin.right;
  const plotH = h - margin.top - margin.bottom;
  textureStats.histogram.forEach((value, index) => {
    const barW = plotW / textureStats.histogram.length - 2;
    const barH = (value / max) * plotH;
    const ratio = index / (textureStats.histogram.length - 1);
    histogramCtx.fillStyle = ratio < 0.45 ? "rgba(73, 143, 213, .78)" : ratio > 0.55 ? "rgba(206, 94, 117, .78)" : "rgba(177, 194, 212, .8)";
    histogramCtx.fillRect(margin.left + index * plotW / textureStats.histogram.length, margin.top + plotH - barH, Math.max(1, barW), barH);
  });
  histogramCtx.strokeStyle = "rgba(132, 160, 212, .22)";
  histogramCtx.beginPath(); histogramCtx.moveTo(margin.left, margin.top + plotH); histogramCtx.lineTo(w - margin.right, margin.top + plotH); histogramCtx.stroke();
  histogramCtx.fillStyle = "rgba(132, 155, 190, .74)";
  histogramCtx.font = "9px ui-monospace, monospace";
  histogramCtx.textAlign = "left"; histogramCtx.fillText("LOW", margin.left, h - 4);
  histogramCtx.textAlign = "center"; histogramCtx.fillText("CENTER", w / 2, h - 4);
  histogramCtx.textAlign = "right"; histogramCtx.fillText("HIGH", w - margin.right, h - 4);
}

function toggleScan() {
  scanning = !scanning;
  const button = $("#cmb-scan-toggle");
  button?.setAttribute("aria-pressed", String(scanning));
  if (button) button.textContent = scanning ? "Ⅱ Stop scan" : "▶ Scan animation";
}

function onViewChange(event) {
  if (event.detail.view === "cmb") { draw(); drawHistogram(); }
}

function tick(timestamp = 0) {
  const delta = lastFrame ? Math.min(40, timestamp - lastFrame) : 0;
  lastFrame = timestamp;
  const active = canvas?.closest(".view")?.classList.contains("active") ?? false;
  const pageVisible = document.visibilityState !== "hidden";
  if (scanning && !globalPaused && active && pageVisible) {
    scanPhase = (scanPhase + delta * 0.000045) % 1;
    const line = $("#cmb-scan-line");
    if (line) { line.style.left = `${10 + scanPhase * 80}%`; line.style.opacity = String(0.3 + 0.5 * Math.sin(scanPhase * Math.PI)); }
  }
  raf = requestAnimationFrame(tick);
}

export function setCmbMotionPaused(paused) {
  globalPaused = Boolean(paused);
}

export function disposeCmbSky() {
  if (raf) cancelAnimationFrame(raf);
  resizeObserver?.disconnect();
  window.removeEventListener("resize", resize);
  window.removeEventListener("nufield:viewchange", onViewChange);
}
