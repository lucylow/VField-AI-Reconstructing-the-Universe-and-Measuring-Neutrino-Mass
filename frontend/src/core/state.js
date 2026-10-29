const initialState = {
  view: "overview",
  redshift: 0,
  motionPaused: false,
  expansionPlaying: false,
  fieldPlaying: true,
  fieldSeed: 2047,
  fieldDensity: 2,
  showFieldLabels: true,
  spectrum: { amplitude: 1, tilt: 0, suppression: 0.3 },
  import: { dataset: null, manifest: null, benchmark: null, extras: [] },
};

let state = structuredCloneSafe(initialState);
const listeners = new Set();

export function getState() {
  return state;
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function updateState(patch) {
  state = { ...state, ...patch };
  publish();
  return state;
}

export function updateNestedState(key, patch) {
  state = { ...state, [key]: { ...state[key], ...patch } };
  publish();
  return state;
}

export function updateImportState(patch) {
  state = { ...state, import: { ...state.import, ...patch } };
  publish();
  return state;
}

export function resetState() {
  state = structuredCloneSafe(initialState);
  publish();
}

function publish() {
  for (const listener of listeners) listener(state);
}

function structuredCloneSafe(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}
