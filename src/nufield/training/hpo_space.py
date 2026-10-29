"""Predeclared, compute-bounded neural inference hyperparameter spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    architecture: str
    density_estimator: str
    learning_rate: float
    batch_size: int
    embedding_dim: int
    dropout: float
    seed: int
    epochs: int
    split_hash: str
    selection_partition: str = "validation"

    def to_dict(self):
        return asdict(self)


def build_hpo_space(
    *,
    architecture: str,
    split_hash: str,
    seeds: tuple[int, ...] = (17, 29, 43),
    learning_rates: tuple[float, ...] = (1e-3, 3e-4),
    batch_sizes: tuple[int, ...] = (32, 64),
    embedding_dims: tuple[int, ...] = (64, 128),
    dropouts: tuple[float, ...] = (0.0, 0.05),
    estimators: tuple[str, ...] = ("maf", "nsf"),
    epochs: int = 150,
    max_trials: int = 48,
    budget: str = "workstation",
) -> list[TrialSpec]:
    """Build a deterministic search grid; do not tune based on locked test metrics."""
    if architecture not in {"spectrum_mlp", "cnn3d", "fno3d", "hybrid"}:
        raise ValueError("unsupported architecture")
    if not split_hash.strip():
        raise ValueError("split_hash is required")
    if not seeds or any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique non-negative integers")
    if (
        any(x <= 0 for x in learning_rates)
        or any(x < 1 for x in batch_sizes)
        or any(x < 1 for x in embedding_dims)
    ):
        raise ValueError("invalid HPO range")
    if any(not 0 <= x < 1 for x in dropouts):
        raise ValueError("dropout values must be in [0,1)")
    if epochs < 1 or max_trials < 1:
        raise ValueError("epochs and max_trials must be positive")
    if budget not in {"laptop", "workstation", "hpc"}:
        raise ValueError("invalid budget")
    configs = list(
        product(learning_rates, batch_sizes, embedding_dims, dropouts, estimators, seeds)
    )
    # Stable selection rather than random order so repeated plans are byte-for-byte reproducible.
    configs.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))
    budget_cap = {"laptop": 12, "workstation": 48, "hpc": 240}[budget]
    limit = min(max_trials, budget_cap)
    if len(configs) > limit:
        # Evenly spaced deterministic subset covers parameter-space endpoints without favoring one setting.
        idx = (
            __import__("numpy")
            .round(__import__("numpy").linspace(0, len(configs) - 1, limit))
            .astype(int)
        )
        configs = [configs[i] for i in sorted(set(idx.tolist()))]
    out = []
    for i, (lr, bs, dim, drop, est, seed) in enumerate(configs, 1):
        tid = f"{architecture}-{est}-lr{lr:g}-bs{bs}-d{dim}-drop{drop:g}-s{seed}"
        out.append(
            TrialSpec(
                tid,
                architecture,
                est,
                float(lr),
                int(bs),
                int(dim),
                float(drop),
                int(seed),
                int(epochs),
                split_hash,
            )
        )
    return out


def hpo_manifest(
    trials: list[TrialSpec],
    *,
    dataset_id: str,
    validation_groups_hash: str,
    test_groups_hash: str,
    metric_name: str = "mean_crps",
) -> dict[str, Any]:
    if not trials:
        raise ValueError("at least one trial required")
    if not dataset_id or not validation_groups_hash or not test_groups_hash:
        raise ValueError("dataset and group hashes are required")
    if validation_groups_hash == test_groups_hash:
        raise ValueError("validation and test group hashes must differ")
    split_hashes = {t.split_hash for t in trials}
    if len(split_hashes) != 1:
        raise ValueError("all trials must use the same frozen split")
    ids = [t.trial_id for t in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("trial IDs must be unique")
    return {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "validation_groups_hash": validation_groups_hash,
        "locked_test_groups_hash": test_groups_hash,
        "selection_metric": metric_name,
        "selection_partition": "validation",
        "trial_count": len(trials),
        "trials": [t.to_dict() for t in trials],
        "rules": [
            "Do not use locked test metrics to choose hyperparameters.",
            "Keep priors, preprocessing, data release and group split fixed.",
            "Run independent seeds for shortlisted configurations.",
            "Use SBC/coverage diagnostics in addition to the selection score.",
        ],
        "scientific_approval": False,
    }
