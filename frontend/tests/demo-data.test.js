import test from "node:test";
import assert from "node:assert/strict";
import { buildIllustrativeParameterPoints, buildIllustrativeSpectrum, DEMO_CAVEAT } from "../src/data/demo-data.js";

test("illustrative parameter points are deterministic for a seed", () => {
  const a = buildIllustrativeParameterPoints(16, 17);
  const b = buildIllustrativeParameterPoints(16, 17);
  assert.deepEqual(a, b);
  assert.equal(a.length, 16);
  assert.ok(a.every((row) => Number.isFinite(row.omega_m) && Number.isFinite(row.sigma8)));
});

test("illustrative parameter points stay inside declared display ranges", () => {
  const points = buildIllustrativeParameterPoints(1000, 81);
  assert.ok(points.every((point) => point.omega_m >= 0.18 && point.omega_m <= 0.42));
  assert.ok(points.every((point) => point.sigma8 >= 0.65 && point.sigma8 <= 0.95));
  assert.ok(points.every((point) => point.sum_mnu_ev >= 0 && point.sum_mnu_ev <= 0.45));
});

test("toy spectrum has positive finite power values across its display range", () => {
  const points = buildIllustrativeSpectrum({ amplitude: 1.2, tilt: -0.1, suppression: 0.4 });
  assert.equal(points.length, 120);
  assert.ok(points.every(({ k, p }) => Number.isFinite(k) && k > 0 && Number.isFinite(p) && p > 0));
  assert.ok(points[0].k < points.at(-1).k);
});

test("synthetic example warnings are explicit", () => {
  assert.match(DEMO_CAVEAT, /not observational data/i);
  assert.match(DEMO_CAVEAT, /posterior samples/i);
});
