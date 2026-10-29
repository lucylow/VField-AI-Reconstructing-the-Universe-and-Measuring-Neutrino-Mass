import { $, setText } from "../core/dom.js";
import { recommendModelPlan } from "../core/model-advisor.js";

export function initModelAdvisor() {
  const form = $("#advisor-form");
  const result = $("#advisor-result");
  if (!form || !result) return;

  const render = () => {
    const nSimulations = Number($("#advisor-simulations", form)?.value);
    const nParameters = Number($("#advisor-parameters", form)?.value);
    const budget = $("#advisor-budget", form)?.value;
    const hasFieldData = $("#advisor-fields", form)?.checked ?? false;
    try {
      const plan = recommendModelPlan({ nSimulations, nParameters, hasFieldData, budget });
      result.classList.remove("advisor-error");
      setText("#advisor-ratio", plan.simulationsPerParameter.toFixed(1), result);
      setText("#advisor-recommendation", plan.recommendation, result);
      setText("#advisor-rationale", plan.rationale, result);
      setText("#advisor-compute", plan.computeAdvice, result);
      setText("#advisor-caveat", plan.caveat, result);
      const list = $("#advisor-next-steps", result);
      list?.replaceChildren();
      plan.nextSteps.forEach((step) => {
        const item = document.createElement("li");
        item.textContent = step;
        list?.append(item);
      });
    } catch (error) {
      result.classList.add("advisor-error");
      setText("#advisor-ratio", "—", result);
      setText("#advisor-recommendation", "Check the study inputs", result);
      setText("#advisor-rationale", error?.message ?? "Enter valid study details.", result);
      setText("#advisor-compute", "", result);
      setText("#advisor-caveat", "", result);
      $("#advisor-next-steps", result)?.replaceChildren();
    }
  };

  form.addEventListener("submit", (event) => event.preventDefault());
  form.addEventListener("input", render);
  form.addEventListener("change", render);
  render();
}
