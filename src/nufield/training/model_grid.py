"""Generate predeclared model/dataset ablation matrices before running expensive experiments.

This generator emits configuration ideas only. It does not train models, tune on held-out tests, or
assume the largest model is best. A benchmark's comparison key records data contract and split ID.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    dataset_id: str
    model_id: str
    observation_view: str
    seed: int
    split_id: str
    estimator: str
    embedding_dim: int
    hidden_features: int
    dropout: float
    training_budget: str
    required_ablation_group: str
    expected_artifacts: tuple[str, ...] = (
        "config.json",
        "training_log.csv",
        "metrics.json",
        "posterior_samples.npz",
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_matrix(
    *,
    dataset_id: str,
    seeds: list[int],
    split_id: str,
    model_ids: tuple[str, ...] = ("pk_classical", "mlp_npe", "cnn3d_npe"),
    observation_views: tuple[str, ...] = ("pk", "field"),
    estimators: tuple[str, ...] = ("maf", "nsf"),
    budget: str = "workstation",
) -> list[ExperimentConfig]:
    if not dataset_id or not seeds or not split_id:
        raise ValueError("dataset_id, seeds and split_id required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    valid_models = {"pk_classical", "mlp_npe", "cnn3d_npe", "fno3d_npe", "hybrid_npe"}
    if set(model_ids) - valid_models:
        raise ValueError(f"unknown model IDs: {sorted(set(model_ids) - valid_models)}")
    if budget not in {"laptop", "workstation", "hpc"}:
        raise ValueError("invalid budget")
    configs = []
    for model_id, view, seed in product(model_ids, observation_views, seeds):
        if model_id == "pk_classical":
            est_list = ("classical",)
        elif model_id == "hybrid_npe":
            est_list = estimators
        else:
            est_list = estimators
        for estimator in est_list:
            # Keep model-view combinations physically coherent rather than generating nonsensical jobs.
            if model_id in {"mlp_npe", "pk_classical"} and view != "pk":
                continue
            if model_id in {"cnn3d_npe", "fno3d_npe"} and view != "field":
                continue
            if model_id == "hybrid_npe" and view != "hybrid":
                continue
            eid = f"{dataset_id}__{model_id}__{view}__{estimator}__seed{seed}"
            emb = 64 if model_id == "mlp_npe" else 128
            hidden = 64 if model_id == "mlp_npe" else 96
            drop = 0.05 if model_id in {"cnn3d_npe", "fno3d_npe", "hybrid_npe"} else 0.0
            configs.append(
                ExperimentConfig(
                    eid,
                    dataset_id,
                    model_id,
                    view,
                    int(seed),
                    split_id,
                    estimator,
                    emb,
                    hidden,
                    drop,
                    budget,
                    f"{model_id}:{view}",
                )
            )
    if not configs:
        raise ValueError("the supplied model/view choices produced no coherent experiments")
    return configs
