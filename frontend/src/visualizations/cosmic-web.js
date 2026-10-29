import { $, setText } from "../core/dom.js";
import { mulberry32 } from "../data/demo-data.js";

let canvas;
let ctx;
let width = 1;
let height = 1;
let nodes = [];
let edges = [];
let particles = [];
let seed = 2047;
let density = 2;
let playing = true;
let globalPaused = false;
let showLabels = true;
let lastTime = 0;
let time = 0;
let raf = 0;
let resizeObserver;

export function initCosmicWeb() {
  canvas = $("#cosmic-web-canvas");
  if (!canvas) return;
  ctx = canvas.getContext("2d");
  $("#field-regenerate")?.addEventListener("click", regenerate);
  $("#field-play")?.addEventListener("click", toggleFlow);
  $("#field-density")?.addEventListener("input", (event) => {
    density = Number(event.currentTarget.value);
    setText("#field-density-value", ["Sparse", "Medium", "Dense"][density - 1]);
    regenerate();
  });
  $("#field-show-labels")?.addEventListener("change", (event) => { showLabels = event.currentTarget.checked; });
  if ("ResizeObserver" in window) {
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
  } else window.addEventListener("resize", resize, { passive: true });
  regenerate();
  window.addEventListener("nufield:viewchange", onViewChange);
  tick();
}

function regenerate() {
  seed = Math.floor(Math.random() * 90000) + 10000;
  setText("#field-seed-label", String(seed));
  const random = mulberry32(seed);
  const count = [26, 42, 60][density - 1];
  nodes = [];
  edges = [];
  particles = [];
  for (let i = 0; i < count; i += 1) {
    const angle = random() * Math.PI * 2;
    const radius = Math.sqrt(random()) * 0.46;
    nodes.push({
      id: i,
      x: 0.5 + Math.cos(angle) * radius * (0.8 + random() * 0.5),
      y: 0.5 + Math.sin(angle) * radius * (0.7 + random() * 0.6),
      mass: 0.45 + random() * 0.95,
      cluster: random() > 0.78,
    });
  }
  // Link each node to its nearest few neighbours, then remove duplicate undirected edges.
  const edgeKeys = new Set();
  nodes.forEach((node) => {
    const nearest = nodes.filter((candidate) => candidate.id !== node.id)
      .map((candidate) => ({ candidate, distance: distance(node, candidate) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 2 + Math.floor(random() * 2));
    nearest.forEach(({ candidate }) => {
      const low = Math.min(node.id, candidate.id);
      const high = Math.max(node.id, candidate.id);
      const key = `${low}:${high}`;
      if (!edgeKeys.has(key)) {
        edgeKeys.add(key);
        edges.push({ from: low, to: high, phase: random() * Math.PI * 2, opacity: 0.18 + random() * 0.28 });
      }
    });
  });
  for (let index = 0; index < Math.min(count * 1.7, 100); index += 1) {
    const edge = edges[Math.floor(random() * edges.length)];
    if (edge) particles.push({ edge, progress: random(), speed: 0.025 + random() * 0.045, size: 0.8 + random() * 1.2 });
  }
  draw();
}

function toggleFlow() {
  playing = !playing;
  const button = $("#field-play");
  button?.setAttribute("aria-pressed", String(!playing));
  if (button) button.textContent = playing ? "Ⅱ Pause flow" : "▶ Resume flow";
}

function resize() {
  if (!canvas || !ctx) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  width = Math.max(1, rect.width);
  height = Math.max(1, rect.height);
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function draw() {
  if (!ctx || width < 2 || height < 2) return;
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "#070d1a"); bg.addColorStop(0.55, "#0b1930"); bg.addColorStop(1, "#10162b");
  ctx.fillStyle = bg; ctx.fillRect(0, 0, width, height);
  drawBackgroundStars();
  const point = (node) => ({ x: node.x * width, y: node.y * height });
  edges.forEach((edge) => {
    const a = point(nodes[edge.from]);
    const b = point(nodes[edge.to]);
    const pulse = playing ? 0.03 * Math.sin(time * 0.0015 + edge.phase) : 0;
    const gradient = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
    gradient.addColorStop(0, `rgba(74, 151, 247, ${edge.opacity + pulse})`);
    gradient.addColorStop(0.5, `rgba(115, 129, 237, ${edge.opacity + 0.1 + pulse})`);
    gradient.addColorStop(1, `rgba(89, 219, 218, ${edge.opacity + pulse})`);
    ctx.strokeStyle = gradient; ctx.lineWidth = 0.75 + (edge.opacity * 1.4);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  });
  particles.forEach((particle) => {
    const a = point(nodes[particle.edge.from]);
    const b = point(nodes[particle.edge.to]);
    const p = particle.progress;
    const x = a.x + (b.x - a.x) * p;
    const y = a.y + (b.y - a.y) * p;
    ctx.fillStyle = "rgba(179, 231, 255, 0.88)";
    ctx.shadowBlur = 8; ctx.shadowColor = "rgba(92, 181, 255, .62)";
    ctx.beginPath(); ctx.arc(x, y, particle.size, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  });
  nodes.forEach((node) => {
    const { x, y } = point(node);
    const r = 1.5 + node.mass * (node.cluster ? 2.8 : 1.35);
    const glow = ctx.createRadialGradient(x, y, 0, x, y, r * 5);
    glow.addColorStop(0, node.cluster ? "rgba(117, 173, 255, .33)" : "rgba(98, 210, 221, .21)");
    glow.addColorStop(1, "rgba(40, 80, 160, 0)");
    ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x, y, r * 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = node.cluster ? "#c2ddff" : "#6eb7ef";
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    if (showLabels && node.cluster) {
      ctx.fillStyle = "rgba(205, 222, 252, .76)";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(`N${String(node.id + 1).padStart(2, "0")}`, x + 7, y - 7);
    }
  });
  if (showLabels) drawVoidLabels();
}

function drawBackgroundStars() {
  for (let index = 0; index < 70; index += 1) {
    const x = stableNoise(index * 13 + 4) * width;
    const y = stableNoise(index * 29 + 5) * height;
    ctx.fillStyle = `rgba(178, 202, 248, ${0.15 + stableNoise(index * 7 + 8) * 0.35})`;
    ctx.beginPath(); ctx.arc(x, y, stableNoise(index * 5 + 10) > 0.94 ? 1 : 0.45, 0, Math.PI * 2); ctx.fill();
  }
}

function drawVoidLabels() {
  const labels = [{ x: 0.22, y: 0.30, text: "VOID REGION" }, { x: 0.71, y: 0.68, text: "UNDERDENSE" }];
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillStyle = "rgba(158, 178, 218, .4)";
  labels.forEach((label) => ctx.fillText(label.text, label.x * width, label.y * height));
}

function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function stableNoise(seedValue) { const value = Math.sin(seedValue * 12.9898) * 43758.5453; return value - Math.floor(value); }

function onViewChange(event) {
  if (event.detail.view === "field") draw();
}

function tick(timestamp = 0) {
  const delta = lastTime ? Math.min(40, timestamp - lastTime) : 0;
  lastTime = timestamp;
  const active = canvas?.closest(".view")?.classList.contains("active") ?? false;
  const pageVisible = document.visibilityState !== "hidden";
  if (playing && !globalPaused && active && pageVisible) {
    time += delta;
    particles.forEach((particle) => { particle.progress = (particle.progress + particle.speed * delta / 1000) % 1; });
    draw();
  }
  raf = requestAnimationFrame(tick);
}

export function setCosmicWebMotionPaused(paused) {
  globalPaused = Boolean(paused);
}

export function disposeCosmicWeb() {
  if (raf) cancelAnimationFrame(raf);
  resizeObserver?.disconnect();
  window.removeEventListener("resize", resize);
  window.removeEventListener("nufield:viewchange", onViewChange);
}
