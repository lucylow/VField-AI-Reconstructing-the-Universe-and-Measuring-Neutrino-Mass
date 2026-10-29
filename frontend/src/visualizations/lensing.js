import { $, setText } from "../core/dom.js";
import { mulberry32, clamp } from "../data/demo-data.js";
import { pointMassLensImages } from "../core/lensing.js";

const SOURCE_COUNT = 48;
let canvas;
let ctx;
let width = 1;
let height = 1;
let strength = 0.65;
let showSources = true;
let animating = false;
let paused = false;
let rayPhase = 0;
let lastTime = 0;
let raf = 0;
let resizeObserver;
let sources = [];

export function initLensing() {
  canvas = $("#lensing-canvas");
  if (!canvas) return;
  ctx = canvas.getContext("2d");
  $("#lensing-strength")?.addEventListener("input", (event) => {
    strength = Number(event.currentTarget.value);
    setText("#lensing-strength-output", `${strength.toFixed(2)}×`);
    setText("#lensing-strength-readout", `${strength.toFixed(2)}×`);
    draw();
  });
  $("#lensing-show-sources")?.addEventListener("change", (event) => { showSources = event.currentTarget.checked; draw(); });
  $("#lensing-animation")?.addEventListener("click", toggleAnimation);
  $("#lensing-reset")?.addEventListener("click", resetLensing);
  if ("ResizeObserver" in window) {
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
  } else window.addEventListener("resize", resize, { passive: true });
  makeSources();
  resize();
  window.addEventListener("nufield:viewchange", onViewChange);
  tick();
}

function makeSources() {
  const random = mulberry32(42013);
  sources = Array.from({ length: SOURCE_COUNT }, (_, index) => {
    const angle = random() * Math.PI * 2;
    const radius = 0.10 + Math.sqrt(random()) * 0.82;
    return {
      id: index + 1,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      size: 1.5 + random() * 1.5,
      aspect: 0.65 + random() * 0.7,
      angle: random() * Math.PI,
      hue: random() > 0.5 ? "blue" : "gold",
      phase: random() * Math.PI * 2,
    };
  });
  setText("#lensing-source-count", String(sources.length));
}

function resetLensing() {
  strength = 0.65;
  showSources = true;
  $("#lensing-strength").value = "0.65";
  $("#lensing-show-sources").checked = true;
  setText("#lensing-strength-output", "0.65×");
  setText("#lensing-strength-readout", "0.65×");
  stopAnimation();
  draw();
}

function toggleAnimation() {
  animating = !animating;
  syncAnimationControl();
  draw();
}

function stopAnimation() {
  animating = false;
  syncAnimationControl();
}

function syncAnimationControl() {
  const button = $("#lensing-animation");
  button?.setAttribute("aria-pressed", String(animating));
  if (button) button.textContent = animating ? "Ⅱ Stop rays" : "▶ Animate rays";
  setText("#lensing-animation-state", animating ? "Running" : "Paused");
}

function resize() {
  if (!canvas || !ctx) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  width = Math.max(1, rect.width);
  height = Math.max(1, rect.height);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function draw() {
  if (!ctx || width < 2 || height < 2) return;
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "#070d19"); bg.addColorStop(0.5, "#0b1830"); bg.addColorStop(1, "#15142a");
  ctx.fillStyle = bg; ctx.fillRect(0, 0, width, height);
  drawStars();
  const cx = width * 0.5;
  const cy = height * 0.5;
  const unit = Math.min(width, height) * 0.42;
  const thetaE = strength * Math.min(width, height) * 0.105;
  drawGuideGeometry(cx, cy, thetaE, unit);

  for (const source of sources) {
    const bx = source.x * unit;
    const by = source.y * unit;
    const beta = Math.hypot(bx, by);
    const directionX = beta > 0 ? bx / beta : 1;
    const directionY = beta > 0 ? by / beta : 0;
    const sourceX = cx + bx;
    const sourceY = cy + by;
    if (showSources) drawSource(source, sourceX, sourceY, true);
    if (strength < 0.001 || beta < 0.0001) {
      drawImage(source, sourceX, sourceY, 1, 0.65);
      continue;
    }
    const images = pointMassLensImages(beta, thetaE);
    const posX = cx + directionX * images.positive;
    const posY = cy + directionY * images.positive;
    const negX = cx + directionX * images.negative;
    const negY = cy + directionY * images.negative;
    if (showSources) {
      ctx.strokeStyle = "rgba(155, 179, 226, .13)";
      ctx.lineWidth = .65;
      ctx.setLineDash([2, 5]);
      ctx.beginPath(); ctx.moveTo(sourceX, sourceY); ctx.lineTo(posX, posY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sourceX, sourceY); ctx.lineTo(negX, negY); ctx.stroke();
      ctx.setLineDash([]);
    }
    const magnificationPlus = Math.abs((images.positive ** 2) / Math.max(1e-6, images.positive ** 2 - images.negative ** 2));
    drawImage(source, posX, posY, clamp(magnificationPlus, 0.4, 1.65), 0.95);
    drawImage(source, negX, negY, 0.73, 0.72);
  }

  drawLensMass(cx, cy, thetaE);
  if (animating && !paused) drawRaySweep(cx, cy, unit, rayPhase);
  drawCaption();
}

function drawGuideGeometry(cx, cy, thetaE, unit) {
  ctx.save();
  ctx.strokeStyle = "rgba(118, 153, 218, .10)";
  ctx.lineWidth = 1;
  for (let ring = 1; ring <= 4; ring += 1) {
    ctx.beginPath(); ctx.arc(cx, cy, unit * ring / 4, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(136, 168, 223, .12)";
  ctx.setLineDash([3, 6]);
  ctx.beginPath(); ctx.moveTo(20, cy); ctx.lineTo(width - 20, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, 30); ctx.lineTo(cx, height - 30); ctx.stroke();
  ctx.setLineDash([]);
  if (thetaE > 0) {
    ctx.strokeStyle = "rgba(120, 177, 255, .22)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, thetaE, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.restore();
}

function drawSource(source, x, y, faint) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(source.angle);
  ctx.fillStyle = faint ? "rgba(139, 167, 212, .43)" : "rgba(139, 195, 255, .8)";
  ctx.shadowBlur = faint ? 0 : 6;
  ctx.shadowColor = "rgba(104, 168, 255, .6)";
  ctx.beginPath(); ctx.ellipse(0, 0, source.size * 1.5, source.size * source.aspect, 0, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawImage(source, x, y, magnification, opacity) {
  if (x < -12 || y < -12 || x > width + 12 || y > height + 12) return;
  const timePulse = animating && !paused ? 0.4 + 0.25 * Math.sin(rayPhase + source.phase) : 0.45;
  const tint = source.hue === "gold" ? [255, 205, 129] : [127, 183, 255];
  const size = source.size * clamp(magnification, 0.45, 1.8);
  const glow = ctx.createRadialGradient(x, y, 0, x, y, size * 5.5);
  glow.addColorStop(0, `rgba(${tint[0]}, ${tint[1]}, ${tint[2]}, ${0.26 * opacity + timePulse * 0.1})`);
  glow.addColorStop(1, `rgba(${tint[0]}, ${tint[1]}, ${tint[2]}, 0)`);
  ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x, y, size * 5.5, 0, Math.PI * 2); ctx.fill();
  ctx.save(); ctx.translate(x, y); ctx.rotate(source.angle + (magnification < 0.9 ? Math.PI / 2 : 0));
  ctx.globalAlpha = opacity;
  ctx.fillStyle = `rgba(${tint[0]}, ${tint[1]}, ${tint[2]}, .9)`;
  ctx.beginPath(); ctx.ellipse(0, 0, size * 1.5, Math.max(1, size * source.aspect), 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = `rgba(235, 244, 255, ${0.34 * opacity})`; ctx.lineWidth = .65;
  ctx.beginPath(); ctx.moveTo(-size * 1.8, 0); ctx.lineTo(size * 1.8, 0); ctx.stroke();
  ctx.restore();
}

function drawLensMass(cx, cy, thetaE) {
  const radius = Math.max(4, thetaE * 0.22);
  const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(36, thetaE * 2.2));
  glow.addColorStop(0, "rgba(10, 12, 25, .98)");
  glow.addColorStop(0.24, "rgba(39, 50, 91, .75)");
  glow.addColorStop(0.42, "rgba(87, 129, 213, .13)");
  glow.addColorStop(1, "rgba(87, 129, 213, 0)");
  ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(cx, cy, Math.max(36, thetaE * 2.2), 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "rgba(133, 183, 255, .48)"; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.arc(cx, cy, Math.max(5, thetaE * 0.55), 0, Math.PI * 2); ctx.stroke();
  ctx.fillStyle = "#111528"; ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "rgba(225, 237, 255, .82)"; ctx.lineWidth = .8; ctx.stroke();
  ctx.fillStyle = "#c9e3ff"; ctx.beginPath(); ctx.arc(cx, cy, 1.3, 0, Math.PI * 2); ctx.fill();
}

function drawRaySweep(cx, cy, unit, phase) {
  const angle = phase * Math.PI * 2;
  const endX = cx + Math.cos(angle) * unit;
  const endY = cy + Math.sin(angle) * unit;
  const gradient = ctx.createLinearGradient(cx, cy, endX, endY);
  gradient.addColorStop(0, "rgba(200, 229, 255, .85)");
  gradient.addColorStop(0.32, "rgba(116, 184, 255, .3)");
  gradient.addColorStop(1, "rgba(116, 184, 255, 0)");
  ctx.strokeStyle = gradient; ctx.lineWidth = 1.1;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(endX, endY); ctx.stroke();
}

function drawStars() {
  for (let index = 0; index < 95; index += 1) {
    const x = noise(index * 13 + 3) * width;
    const y = noise(index * 31 + 7) * height;
    ctx.fillStyle = `rgba(189, 212, 252, ${0.12 + noise(index * 17 + 6) * 0.38})`;
    ctx.beginPath(); ctx.arc(x, y, noise(index * 29 + 1) > 0.92 ? 1.1 : 0.5, 0, Math.PI * 2); ctx.fill();
  }
}

function drawCaption() {
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillStyle = "rgba(147, 169, 208, .55)";
  ctx.fillText("SCREEN-SPACE SCHEMATIC · NO PHYSICAL SCALE", 15, height - 14);
}

function noise(value) { const value2 = Math.sin(value * 78.233) * 43758.5453123; return value2 - Math.floor(value2); }
function onViewChange(event) { if (event.detail.view === "lensing") draw(); }

function tick(timestamp = 0) {
  const delta = lastTime ? Math.min(40, timestamp - lastTime) : 0;
  lastTime = timestamp;
  const active = canvas?.closest(".view")?.classList.contains("active") ?? false;
  const pageVisible = document.visibilityState !== "hidden";
  if (active && pageVisible && animating && !paused) {
    rayPhase = (rayPhase + delta * 0.0014) % (Math.PI * 2);
    draw();
  }
  raf = requestAnimationFrame(tick);
}

export function setLensingMotionPaused(value) {
  paused = Boolean(value);
  setText("#lensing-animation-state", animating ? (paused ? "Paused globally" : "Running") : "Paused");
  if (!paused && animating) draw();
}

export function disposeLensing() {
  if (raf) cancelAnimationFrame(raf);
  resizeObserver?.disconnect();
  window.removeEventListener("resize", resize);
  window.removeEventListener("nufield:viewchange", onViewChange);
}
