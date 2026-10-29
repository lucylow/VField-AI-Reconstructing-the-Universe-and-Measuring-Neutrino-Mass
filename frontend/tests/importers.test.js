import test from "node:test";
import assert from "node:assert/strict";
import { parseJSONFiles, summarizeImported } from "../src/data/importers.js";

function fakeFile(name, value, size) {
  return { name, size: size ?? Buffer.byteLength(value), async text() { return value; } };
}

test("classifies dataset, manifest and benchmark JSON by schema or filename", async () => {
  const files = [
    fakeFile("dataset.json", JSON.stringify({ physics_level: "toy", parameter_names: ["omega_m"], feature_names: ["f"] })),
    fakeFile("dataset.manifest.json", JSON.stringify({ schema_version: "1.0", dataset: { sha256: "abc" } })),
    fakeFile("benchmark_report.json", JSON.stringify({ parameter_names: ["omega_m"], median_absolute_error: { omega_m: 0.02 } })),
  ];
  const result = await parseJSONFiles(files);
  assert.equal(result.successCount, 3);
  assert.equal(result.imported.dataset.content.physics_level, "toy");
  assert.equal(result.imported.manifest.content.schema_version, "1.0");
  assert.equal(result.imported.benchmark.content.median_absolute_error.omega_m, 0.02);
  assert.deepEqual(result.errors, []);
});

test("invalid JSON and unsupported formats return errors without throwing", async () => {
  const result = await parseJSONFiles([
    fakeFile("bad.json", "{ definitely not json"),
    fakeFile("arrays.npz", "not an npz file"),
  ]);
  assert.equal(result.successCount, 0);
  assert.equal(result.errors.length, 2);
});

test("large files and non-object JSON are rejected", async () => {
  const result = await parseJSONFiles([
    fakeFile("large.json", "{}", 20 * 1024 * 1024 + 1),
    fakeFile("array.json", "[1,2,3]"),
  ]);
  assert.equal(result.successCount, 0);
  assert.match(result.errors[0], /20 MB/i);
  assert.match(result.errors[1], /top level/i);
});

test("summary surfaces project physics label and scientific warning", () => {
  const summary = summarizeImported({
    dataset: { content: { physics_level: "toy", scientific_warning: "toy only" } },
    benchmark: null,
    manifest: null,
    extras: [],
  });
  assert.equal(summary.physicsLevel, "toy");
  assert.equal(summary.warning, "toy only");
});

test("missing or unreadable file selections return actionable errors", async () => {
  const empty = await parseJSONFiles(null);
  assert.equal(empty.successCount, 0);
  assert.match(empty.errors[0], /No files/i);
  const unreadable = await parseJSONFiles([null, { name: "bad.json", size: 1 }]);
  assert.equal(unreadable.successCount, 0);
  assert.equal(unreadable.errors.length, 2);
});

test("duplicate recognized artifacts do not silently replace the first file", async () => {
  const result = await parseJSONFiles([
    fakeFile("benchmark_report_first.json", JSON.stringify({ median_absolute_error: { x: 1 } })),
    fakeFile("benchmark_report_second.json", JSON.stringify({ median_absolute_error: { x: 2 } })),
  ]);
  assert.equal(result.successCount, 1);
  assert.equal(result.imported.benchmark.content.median_absolute_error.x, 1);
  assert.match(result.errors[0], /first one was kept/i);
});

test("non-finite numeric values and invalid file sizes are rejected", async () => {
  const result = await parseJSONFiles([
    fakeFile("overflow.json", '{"value":1e400}'),
    { ...fakeFile("unknown-size.json", "{}"), size: Number.NaN },
  ]);
  assert.equal(result.successCount, 0);
  assert.match(result.errors[0], /non-finite/i);
  assert.match(result.errors[1], /size.*invalid/i);
});
