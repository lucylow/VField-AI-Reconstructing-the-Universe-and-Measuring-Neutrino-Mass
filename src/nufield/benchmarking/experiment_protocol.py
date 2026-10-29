"""Machine-readable checklist for fair field-level vs summary-statistic comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BenchmarkProtocol:
    protocol_id: str = "nufield-mnu-baseline-v1"
    target_parameter: str = "Mnu"
    prior_identical: bool = False
    simulator_family_matched: bool = False
    same_test_simulations: bool = False
    same_mask_and_noise: bool = False
    nuisance_parameters_included: bool = False
    preprocessing_fit_train_only: bool = False
    test_set_locked_before_tuning: bool = False
    all_random_seeds_recorded: bool = False
    posterior_calibration_run: bool = False
    compute_cost_recorded: bool = False
    real_data_used_only_after_validation: bool = True

    def blockers(self) -> list[str]:
        required = [
            "prior_identical",
            "simulator_family_matched",
            "same_test_simulations",
            "same_mask_and_noise",
            "nuisance_parameters_included",
            "preprocessing_fit_train_only",
            "test_set_locked_before_tuning",
            "all_random_seeds_recorded",
            "posterior_calibration_run",
            "compute_cost_recorded",
        ]
        return [name for name in required if not getattr(self, name)]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "blocking_items": self.blockers(),
            "comparison_ready": not self.blockers(),
            "warning": "Checklist completion is procedural, not evidence of model correctness or publication approval.",
        }
