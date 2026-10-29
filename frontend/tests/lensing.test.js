import test from "node:test";
import assert from "node:assert/strict";
import { pointMassLensImages, pointMassLensResidual } from "../src/core/lensing.js";

test("point-mass lens helper returns ordered positive and negative image offsets", () => {
  const images = pointMassLensImages(0.5, 0.2);
  assert.ok(images.positive > 0);
  assert.ok(images.negative < 0);
  assert.ok(images.positive > Math.abs(images.negative));
});

test("lens image positions satisfy the dimensionless point-mass equation", () => {
  const beta = 0.42;
  const thetaE = 0.17;
  const images = pointMassLensImages(beta, thetaE);
  assert.ok(Math.abs(pointMassLensResidual(images.positive, beta, thetaE)) < 1e-10);
  assert.ok(Math.abs(pointMassLensResidual(images.negative, beta, thetaE)) < 1e-10);
});

test("zero lens strength leaves only the unlensed source solution", () => {
  assert.deepEqual(pointMassLensImages(0.82, 0), { positive: 0.82, negative: null });
});

test("lens helper rejects invalid offsets and scales", () => {
  assert.throws(() => pointMassLensImages(-0.2, 0.1), RangeError);
  assert.throws(() => pointMassLensImages(0.2, -0.1), RangeError);
  assert.throws(() => pointMassLensResidual(0, 0.2, 0.1), RangeError);
});
