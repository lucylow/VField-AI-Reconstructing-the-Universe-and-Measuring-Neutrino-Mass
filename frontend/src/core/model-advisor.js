const BUDGETS = new Set(["laptop", "workstation", "hpc"]);

/**
 * Recommend a model comparison plan from declared data volume and compute budget.
 * This is a transparent heuristic, not a trained model or an accuracy predictor.
 */
export function recommendModelPlan({
  nSimulations,
  nParameters,
  hasFieldData = false,
  budget = "laptop",
}) {
  if (!Number.isInteger(nSimulations) || nSimulations < 1) {
    throw new RangeError("Enter a positive whole number of independent simulations.");
  }
  if (!Number.isInteger(nParameters) || nParameters < 1) {
    throw new RangeError("Enter a positive whole number of inferred parameters.");
  }
  if (!BUDGETS.has(budget)) throw new RangeError("Choose laptop, workstation, or HPC compute.");

  const simulationsPerParameter = nSimulations / nParameters;
  let recommendation;
  let rationale;
  if (simulationsPerParameter < 20) {
    recommendation = "Start with a classical summary-statistic baseline";
    rationale = "The simulation budget is very small relative to the number of inferred parameters. Keep neural estimators exploratory until independent calibration simulations are available.";
  } else if (simulationsPerParameter < 100) {
    recommendation = "Compare a classical baseline with a compact summary model";
    rationale = "A small MDN or summary MLP may be a useful challenger, but avoid a large field encoder as the default. Keep the locked test set untouched.";
  } else if (hasFieldData) {
    recommendation = "Benchmark a compact 3-D CNN + NPE against the spectrum baseline";
    rationale = "The declared data-to-parameter ratio and field inputs justify testing spatial information, provided simulations are independent and grids/units are consistent.";
  } else {
    recommendation = "Compare a summary MLP/MDN + posterior estimator with the classical baseline";
    rationale = "Without compatible field inputs, focus on documented power-spectrum or correlation summaries and compare posterior calibration, not only point error.";
  }

  const computeAdvice = budget === "laptop"
    ? "Laptop budget: begin with summary models and small batches; use a field encoder only as a compact, measured ablation."
    : budget === "workstation"
      ? "Workstation budget: reserve capacity for multiple seeds and held-out calibration, not just larger models."
      : "HPC budget: spend additional compute on independent simulations, matched baselines, and robustness suites before scaling model capacity.";

  return {
    simulationsPerParameter,
    recommendation,
    rationale,
    computeAdvice,
    nextSteps: [
      "Split by independent simulation/group IDs before fitting preprocessing.",
      "Compare candidates on the same prior, observation model, and held-out simulations.",
      "Require SBC and interval-coverage checks before treating a posterior as usable.",
    ],
    caveat: "Rule-based research guidance only—not a trained AI, performance prediction, or scientific approval.",
  };
}
