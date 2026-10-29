import test from "node:test";
import assert from "node:assert/strict";
import { recommendModelPlan } from "../src/core/model-advisor.js";

test("low data-to-parameter ratio favors a classical baseline", () => {
  const plan = recommendModelPlan({ nSimulations: 60, nParameters: 5 });
  assert.equal(plan.simulationsPerParameter, 12);
  assert.match(plan.recommendation, /classical summary-statistic baseline/);
  assert.match(plan.caveat, /not a trained AI/);
});

test("field inputs enable a 3-D model comparison only in a higher data regime", () => {
  const plan = recommendModelPlan({
    nSimulations: 1200,
    nParameters: 6,
    hasFieldData: true,
    budget: "workstation",
  });
  assert.match(plan.recommendation, /3-D CNN/);
  assert.equal(plan.nextSteps.length, 3);
});

test("compute guidance responds to the selected budget", () => {
  const laptop = recommendModelPlan({ nSimulations: 500, nParameters: 5, budget: "laptop" });
  const hpc = recommendModelPlan({ nSimulations: 500, nParameters: 5, budget: "hpc" });
  assert.match(laptop.computeAdvice, /small batches/);
  assert.match(hpc.computeAdvice, /independent simulations/);
});

test("invalid advisor inputs fail with actionable errors", () => {
  assert.throws(
    () => recommendModelPlan({ nSimulations: 0, nParameters: 5 }),
    /positive whole number of independent simulations/,
  );
  assert.throws(
    () => recommendModelPlan({ nSimulations: 100, nParameters: 0 }),
    /positive whole number of inferred parameters/,
  );
  assert.throws(
    () => recommendModelPlan({ nSimulations: 100, nParameters: 5, budget: "unknown" }),
    /Choose laptop, workstation, or HPC/,
  );
});
