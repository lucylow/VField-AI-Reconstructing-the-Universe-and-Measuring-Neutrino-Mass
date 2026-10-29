"""Budget-bounded, validation-only hyperparameter grid for conditional MDN experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from nufield.models.conditional_mdn import MDNConfig
from nufield.training.mdn_workflow import train_mdn_from_npz


@dataclass(frozen=True)
class MDNTrial:
    trial_id: str
    n_components: int
    hidden_dim: int
    hidden_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    fit_seed: int

    def to_dict(self):
        return asdict(self)


def build_mdn_trials(
    *,
    max_trials: int = 8,
    budget: str = "workstation",
    fit_seeds: tuple[int, ...] = (17,),
    components: tuple[int, ...] = (2, 4, 8),
    hidden_dims: tuple[int, ...] = (64, 128),
    hidden_layers: tuple[int, ...] = (1, 2),
    dropouts: tuple[float, ...] = (0.0, 0.05),
    learning_rates: tuple[float, ...] = (3e-4, 1e-3),
    weight_decay: float = 1e-5,
    batch_sizes: tuple[int, ...] = (32, 64),
) -> list[MDNTrial]:
    """Construct a deterministic grid; never consult a locked test score while ranking candidates."""
    if budget not in {"laptop", "workstation", "hpc"}:
        raise ValueError("invalid budget")
    if max_trials < 1 or weight_decay < 0:
        raise ValueError("invalid trial limit/weight decay")
    for name, values in (
        ("components", components),
        ("hidden_dims", hidden_dims),
        ("hidden_layers", hidden_layers),
        ("fit_seeds", fit_seeds),
        ("batch_sizes", batch_sizes),
    ):
        if not values or any(v < 1 for v in values):
            raise ValueError(f"{name} values must be positive and nonempty")
    if not dropouts or any(not 0 <= d < 1 for d in dropouts):
        raise ValueError("dropouts must be in [0,1)")
    if not learning_rates or any(lr <= 0 for lr in learning_rates):
        raise ValueError("learning rates must be positive")
    config = list(
        product(
            components, hidden_dims, hidden_layers, dropouts, learning_rates, batch_sizes, fit_seeds
        )
    )
    config.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    cap = {"laptop": 4, "workstation": 16, "hpc": 96}[budget]
    limit = min(max_trials, cap)
    if len(config) > limit:
        idx = np.round(np.linspace(0, len(config) - 1, limit)).astype(int)
        config = [config[i] for i in sorted(set(idx.tolist()))]
    trials = []
    for k, (ncomp, hid, depth, drop, lr, bs, seed) in enumerate(config, 1):
        tid = f"mdn-k{ncomp}-h{hid}-l{depth}-d{drop:g}-lr{lr:g}-bs{bs}-s{seed}"
        trials.append(
            MDNTrial(
                tid,
                int(ncomp),
                int(hid),
                int(depth),
                float(drop),
                float(lr),
                float(weight_decay),
                int(bs),
                int(seed),
            )
        )
    return trials


def run_mdn_grid(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    parameter_names: list[str],
    prior_low: list[float],
    prior_high: list[float],
    target_name: str = "sum_mnu_ev",
    observation_key: str = "features",
    theta_key: str = "theta",
    group_id_key: str = "group_id",
    simulation_id_key: str = "simulation_id",
    split_seed: int = 42,
    max_trials: int = 6,
    budget: str = "workstation",
    epochs: int = 100,
    patience: int = 15,
    n_validation_observations: int = 32,
    posterior_samples: int = 512,
    max_coverage_gap: float = 0.25,
    minimum_validation_groups: int = 20,
    allow_toy: bool = False,
) -> dict[str, Any]:
    """Run controlled MDN candidates with one frozen group split and validation-only ranking.

    This is a model-selection driver, not final assessment. Each model's early stopping uses an inner
    training-group split; the outer validation groups rank candidates; the test partition is not
    opened by any trial. Run one explicit final test evaluation only after freezing the selected model.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trials = build_mdn_trials(max_trials=max_trials, budget=budget)
    records = []
    for trial in trials:
        cfg = MDNConfig(
            n_components=trial.n_components,
            hidden_dim=trial.hidden_dim,
            n_hidden_layers=trial.hidden_layers,
            dropout=trial.dropout,
            learning_rate=trial.learning_rate,
            weight_decay=trial.weight_decay,
            batch_size=trial.batch_size,
            max_epochs=epochs,
            patience=patience,
            seed=trial.fit_seed,
            device="cpu",
        )
        path = out / trial.trial_id
        report = train_mdn_from_npz(
            dataset_path,
            path,
            observation_key=observation_key,
            theta_key=theta_key,
            group_id_key=group_id_key,
            simulation_id_key=simulation_id_key,
            parameter_names=parameter_names,
            prior_low=prior_low,
            prior_high=prior_high,
            target_name=target_name,
            seed=split_seed,
            config=cfg,
            n_validation_observations=n_validation_observations,
            posterior_samples=posterior_samples,
            evaluate_test_once=False,
            allow_toy=allow_toy,
        )
        m = report["validation_posterior_metrics"]
        widths = np.asarray(prior_high, dtype=float) - np.asarray(prior_low, dtype=float)
        crps = np.asarray(m["mean_crps_by_parameter"], dtype=float) / widths
        cov68 = np.asarray(m["coverage"]["0.68"]["coverage_by_parameter"], dtype=float)
        gap = float(np.max(np.abs(cov68 - 0.68)))
        n_groups = report["split_group_counts"]["validation"]
        gate = bool(
            np.isfinite(crps).all()
            and gap <= max_coverage_gap
            and n_groups >= minimum_validation_groups
        )
        records.append(
            {
                "trial_id": trial.trial_id,
                "trial": trial.to_dict(),
                "output_dir": str(path.resolve()),
                "validation_mean_normalized_crps": float(crps.mean()),
                "validation_normalized_crps_by_parameter": crps.tolist(),
                "validation_68pct_coverage_by_parameter": cov68.tolist(),
                "max_68pct_coverage_gap": gap,
                "n_outer_validation_groups": n_groups,
                "calibration_screen_passed": gate,
                "selection_partition": "outer validation only",
                "test_used": False,
            }
        )
    eligible = [r for r in records if r["calibration_screen_passed"]]
    rank_source = eligible if eligible else []
    ranked = sorted(rank_source, key=lambda r: r["validation_mean_normalized_crps"])
    summary = {
        "schema_version": "1.0",
        "dataset_path": str(Path(dataset_path).resolve()),
        "split_seed": split_seed,
        "budget": budget,
        "trial_count": len(records),
        "trials": records,
        "ranked_validation_only": ranked,
        "selected_candidate": ranked[0] if ranked else None,
        "selection_policy": {
            "metric": "mean CRPS normalized by prior range",
            "max_68pct_coverage_gap": max_coverage_gap,
            "minimum_validation_parent_groups": minimum_validation_groups,
            "requires_coverage_screen": True,
        },
        "test_policy": "No trial evaluates the locked test partition. Freeze candidate/config before calling train_mdn_from_npz with evaluate_test_once=true on the frozen split protocol.",
        "scientific_approval": False,
        "warning": "The coverage screen is only a coarse model-selection filter. Run SBC and independent calibration suites before any physical interpretation.",
    }
    (out / "mdn_grid_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
