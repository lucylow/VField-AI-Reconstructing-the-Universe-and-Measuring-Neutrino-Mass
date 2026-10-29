"""Preflight and evaluation reports that separate model quality from evidence quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from nufield.ai.contracts import AIExperimentContract
from nufield.ai.metrics import regression_metrics


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    severity: str
    explanation: str
    observed: Any = None
    expected: Any = None

    def to_dict(self):
        return asdict(self)


def scientific_claim_gates(
    *,
    contract: AIExperimentContract,
    group_leakage_detected: bool,
    calibration_error: float | None,
    max_calibration_error: float = 0.08,
    systematic_bias_fraction: float | None = None,
    max_systematic_bias_fraction: float = 0.25,
) -> dict[str, Any]:
    """Evaluate basic evidence gates; a pass is necessary, never sufficient, for a claim."""
    gates = []
    contract_errors = contract.validate(for_scientific_claim=True)
    gates.append(
        Gate(
            "experiment_contract",
            not contract_errors,
            "critical",
            "Declared data, prior, split and simulator assumptions must pass.",
            contract_errors,
            [],
        )
    )
    gates.append(
        Gate(
            "group_leakage",
            not group_leakage_detected,
            "critical",
            "No parent simulation may cross train/validation/test boundaries.",
            bool(group_leakage_detected),
            False,
        )
    )
    if calibration_error is None or not np.isfinite(calibration_error):
        gates.append(
            Gate(
                "posterior_calibration",
                False,
                "critical",
                "Calibration must be evaluated on independent simulated truths.",
                calibration_error,
                f"<= {max_calibration_error}",
            )
        )
    else:
        gates.append(
            Gate(
                "posterior_calibration",
                calibration_error <= max_calibration_error,
                "critical",
                "Calibration error is compared with a predeclared threshold.",
                float(calibration_error),
                f"<= {max_calibration_error}",
            )
        )
    if systematic_bias_fraction is None or not np.isfinite(systematic_bias_fraction):
        gates.append(
            Gate(
                "systematic_robustness",
                False,
                "high",
                "Systematic perturbation study is required.",
                systematic_bias_fraction,
                f"<= {max_systematic_bias_fraction}",
            )
        )
    else:
        gates.append(
            Gate(
                "systematic_robustness",
                systematic_bias_fraction <= max_systematic_bias_fraction,
                "high",
                "Worst declared systematic bias is compared with the threshold.",
                float(systematic_bias_fraction),
                f"<= {max_systematic_bias_fraction}",
            )
        )
    return {
        "gates": [g.to_dict() for g in gates],
        "all_gates_pass": bool(all(g.passed for g in gates)),
        "warning": "Passing these software gates does not replace expert review or peer review.",
    }


def heldout_regression_report(
    target: Any,
    prediction: Any,
    *,
    variance: Any | None = None,
    partition_name: str = "locked_test",
    data_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Create a held-out prediction report with explicit provenance fields."""
    if partition_name.lower() not in {"locked_test", "heldout_test", "independent_test"}:
        raise ValueError("this function only accepts explicitly held-out test partitions")
    metrics = regression_metrics(target, prediction, variance=variance)
    return {
        "partition": partition_name,
        "data_fingerprint": data_fingerprint,
        "metrics": metrics,
        "interpretation": "Metrics describe this split and do not prove simulator validity.",
    }
