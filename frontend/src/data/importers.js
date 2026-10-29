const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_JSON_DEPTH = 40;

export async function parseJSONFiles(fileList) {
  const imported = { dataset: null, manifest: null, benchmark: null, extras: [] };
  const errors = [];
  let files;
  try {
    files = Array.from(fileList ?? []);
  } catch {
    return { imported, errors: ["The selected files could not be read."], successCount: 0 };
  }
  if (!files.length) errors.push("No files were selected.");
  for (const file of files) {
    const name = typeof file?.name === "string" && file.name ? file.name : "(unnamed file)";
    if (name === "(unnamed file)" || typeof file?.text !== "function") {
      errors.push(`${name}: selected item is not a readable file.`);
      continue;
    }
    if (name.length > 240) {
      errors.push(`${name.slice(0, 64)}: filename is unexpectedly long.`);
      continue;
    }
    if (!name.toLowerCase().endsWith(".json")) {
      errors.push(`${name}: only JSON files are supported in this importer.`);
      continue;
    }
    if (!Number.isFinite(file.size) || file.size < 0) {
      errors.push(`${name}: file size is missing or invalid.`);
      continue;
    }
    if (file.size > MAX_FILE_BYTES) {
      errors.push(`${name}: file is larger than the 20 MB limit.`);
      continue;
    }
    try {
      const content = JSON.parse(await file.text());
      assertShallowJSON(content);
      const item = { name, size: file.size, importedAt: new Date().toISOString(), content };
      const type = classifyArtifact(name, content);
      if (type !== "extra" && imported[type]) {
        errors.push(`${name}: another ${type} artifact was already selected; the first one was kept.`);
        continue;
      }
      if (type === "dataset") imported.dataset = item;
      else if (type === "manifest") imported.manifest = item;
      else if (type === "benchmark") imported.benchmark = item;
      else imported.extras.push(item);
    } catch (error) {
      errors.push(`${name}: ${error?.message ?? "could not parse JSON"}`);
    }
  }
  return { imported, errors, successCount: Object.values(imported).reduce((sum, item) => sum + (Array.isArray(item) ? item.length : Number(Boolean(item))), 0) };
}

function classifyArtifact(fileName, json) {
  const name = fileName.toLowerCase();
  if (name.includes("benchmark_report") || (json?.median_absolute_error && json?.parameter_names)) return "benchmark";
  if (name.includes("manifest") || (json?.dataset?.sha256 && json?.schema_version)) return "manifest";
  if (name.includes("dataset") || (json?.feature_names && json?.parameter_names && json?.physics_level)) return "dataset";
  return "extra";
}

function assertShallowJSON(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected a JSON object at the top level.");
  let depth = 0;
  const stack = [{ value, level: 1 }];
  while (stack.length) {
    const current = stack.pop();
    depth = Math.max(depth, current.level);
    if (depth > MAX_JSON_DEPTH) throw new Error("JSON nesting is too deep to safely display.");
    if (!current.value || typeof current.value !== "object") continue;
    for (const entry of Object.values(current.value)) {
      if (typeof entry === "number" && !Number.isFinite(entry)) {
        throw new Error("JSON contains a non-finite number.");
      }
      if (entry && typeof entry === "object") stack.push({ value: entry, level: current.level + 1 });
    }
  }
}

export function summarizeImported(importState) {
  const benchmark = importState.benchmark?.content ?? null;
  const dataset = importState.dataset?.content ?? null;
  const manifest = importState.manifest?.content ?? null;
  return {
    benchmark,
    dataset,
    manifest,
    physicsLevel: benchmark?.physics_level ?? dataset?.physics_level ?? manifest?.physics_level ?? null,
    warning: benchmark?.scientific_warning ?? dataset?.scientific_warning ?? manifest?.scientific_warning ?? null,
  };
}
