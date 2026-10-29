import test from "node:test";
import assert from "node:assert/strict";
import { scaleFactorFromRedshift, redshiftFromScaleFactor, formatEpoch } from "../src/core/cosmology.js";

test("scale factor follows a = 1 / (1 + z)", () => {
  assert.equal(scaleFactorFromRedshift(0), 1);
  assert.equal(scaleFactorFromRedshift(1), 0.5);
  assert.ok(Math.abs(scaleFactorFromRedshift(9) - 0.1) < 1e-12);
});

test("redshift and scale factor round-trip within floating precision", () => {
  for (const z of [0, 0.1, 0.8, 2, 7.5, 10]) {
    assert.ok(Math.abs(redshiftFromScaleFactor(scaleFactorFromRedshift(z)) - z) < 1e-10);
  }
});

test("concept helper rejects invalid model inputs", () => {
  for (const value of [-1, Number.NaN, Number.POSITIVE_INFINITY]) assert.throws(() => scaleFactorFromRedshift(value), RangeError);
  for (const value of [0, -1, 1.1, Number.NaN]) assert.throws(() => redshiftFromScaleFactor(value), RangeError);
});

test("epoch labels are descriptive and bounded", () => {
  assert.equal(formatEpoch(0), "Present day");
  assert.equal(formatEpoch(2), "Earlier universe");
  assert.equal(formatEpoch(8), "High-redshift era");
});
