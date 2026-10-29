/**
 * Reproducible, deliberately synthetic values for interface development.
 * These never masquerade as observations or a physical simulation.
 */
export const DEMO_PARAMETERS = {
  omega_m: { label: "Ωm", fullName: "Matter density parameter", min: 0.18, max: 0.42, center: 0.30 },
  sigma8: { label: "σ₈", fullName: "Fluctuation normalization", min: 0.65, max: 0.95, center: 0.81 },
  h: { label: "h", fullName: "Scaled Hubble parameter", min: 0.58, max: 0.82, center: 0.68 },
  n_s: { label: "nₛ", fullName: "Primordial spectral index", min: 0.88, max: 1.04, center: 0.965 },
  sum_mnu_ev: { label: "Σmν", fullName: "Sum of neutrino masses (toy parameter)", min: 0.0, max: 0.45, center: 0.12 },
  bias: { label: "Bias", fullName: "Tracer bias coefficient", min: 0.65, max: 1.55, center: 1.0 },
};

export const DEMO_CAVEAT = "Synthetic frontend values for interaction demonstration only. They are not observational data or posterior samples, and they are not a cosmological prediction.";

export function buildIllustrativeParameterPoints(count = 90, seed = 90721) {
  const rand = mulberry32(seed);
  return Array.from({ length: count }, (_, index) => {
    const shared = rand() * 2 - 1;
    const omega_m = clamp(0.30 + shared * 0.075 + (rand() - 0.5) * 0.06, 0.18, 0.42);
    const sigma8 = clamp(0.81 - shared * 0.055 + (rand() - 0.5) * 0.075, 0.65, 0.95);
    const h = clamp(0.68 + (rand() - 0.5) * 0.20, 0.58, 0.82);
    const n_s = clamp(0.965 + (rand() - 0.5) * 0.10, 0.88, 1.04);
    const sum_mnu_ev = clamp(0.12 + (rand() - 0.5) * 0.25, 0, 0.45);
    const bias = clamp(1 + (rand() - 0.5) * 0.56, 0.65, 1.55);
    return { id: `illustration-${String(index + 1).padStart(3, "0")}`, omega_m, sigma8, h, n_s, sum_mnu_ev, bias };
  });
}

export function buildIllustrativeSpectrum({ amplitude = 1, tilt = 0, suppression = 0.3, count = 120 } = {}) {
  const points = [];
  const minLog = -2;
  const maxLog = 1.2;
  for (let index = 0; index < count; index += 1) {
    const u = index / (count - 1);
    const k = 10 ** (minLog + u * (maxLog - minLog));
    // The function is aesthetic scaffolding only; no transfer function is implied.
    const turnover = 1 / (1 + (k / 0.055) ** 1.55);
    const broadShape = (k / 0.08) ** (0.85 + tilt * 0.32) / (1 + (k / 0.9) ** 1.05);
    const oscillation = 1 + 0.055 * Math.sin(8.3 * Math.log(k + 0.002));
    const damping = Math.exp(-suppression * (k / 1.8) ** 0.66);
    const p = Math.max(1e-5, amplitude * broadShape * (0.65 + 0.35 * turnover) * oscillation * damping);
    points.push({ k, p });
  }
  return points;
}

export function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
