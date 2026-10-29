"""Transparent model training budget calculator and run-plan helper."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingBudgetEstimate:
    n_simulations: int
    train_fraction: float
    epochs: int
    batch_size: int
    steps_per_epoch: int
    total_optimizer_steps: int
    estimated_gpu_hours: float
    estimated_peak_memory_gb: float
    recommended_run_mode: str
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_training_budget(
    *,
    n_simulations: int,
    train_fraction: float = 0.8,
    epochs: int = 100,
    batch_size: int = 8,
    seconds_per_step: float = 0.15,
    peak_memory_gb: float = 8.0,
    safety_factor: float = 1.5,
    budget: str = "workstation",
) -> TrainingBudgetEstimate:
    """Estimate time and memory from measured pilot throughput, not theoretical model scores.

    `seconds_per_step` should be measured by running 20-100 representative batches on the target
    machine. Default is only a placeholder for planning and is not guaranteed hardware performance.
    """
    if n_simulations < 10:
        raise ValueError("n_simulations must be >=10")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0,1)")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if seconds_per_step <= 0 or peak_memory_gb <= 0 or safety_factor < 1:
        raise ValueError("invalid timing/memory assumptions")
    if budget not in {"laptop", "workstation", "hpc"}:
        raise ValueError("invalid budget")
    n_train = max(1, int(n_simulations * train_fraction))
    steps = math.ceil(n_train / batch_size)
    seconds = steps * epochs * seconds_per_step * safety_factor
    # Memory isn't inferred from parameter count here; this is the user-supplied pilot observed peak.
    mode = "pilot_required"
    if budget == "laptop" and peak_memory_gb > 8:
        mode = "reduce_grid_or_batch"
    elif budget == "workstation" and peak_memory_gb > 32:
        mode = "single_gpu_hpc_or_reduce_grid"
    elif budget == "hpc" and peak_memory_gb > 80:
        mode = "multi_gpu_or_patch_training_review"
    else:
        mode = "single-device-plan"
    assumptions = (
        "Seconds/step and peak memory should be measured on a representative pilot.",
        "Estimate excludes simulation generation, preprocessing, checkpoint I/O, and validation/SBC runs.",
        "Reserve additional compute for multiple seeds and independent simulation-based calibration.",
    )
    return TrainingBudgetEstimate(
        n_simulations,
        train_fraction,
        epochs,
        batch_size,
        steps,
        steps * epochs,
        seconds / 3600,
        peak_memory_gb * safety_factor,
        mode,
        assumptions,
    )
