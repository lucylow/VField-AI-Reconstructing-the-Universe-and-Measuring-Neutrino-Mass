"""Independent-model ensembles and predictive aggregation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from nufield.ai.training import TrainConfig, TrainResult, fit_model
from nufield.ai.uncertainty import EnsembleUncertainty, decompose_ensemble
from nufield.exceptions import OptionalDependencyError

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class EnsembleFitResult:
    models: list[Any]
    training_runs: list[TrainResult]
    seeds: list[int]
    config: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "members": len(self.models),
            "seeds": self.seeds,
            "best_epochs": [r.best_epoch for r in self.training_runs],
            "best_validation_losses": [r.best_val_loss for r in self.training_runs],
            "config": self.config,
            "warning": "Use held-out group evaluation; ensemble spread is not calibration by itself.",
        }


def fit_ensemble(
    model_factory: Callable[[int], Any],
    train_loader_factory: Callable[[int], Any],
    validation_loader_factory: Callable[[int], Any],
    *,
    n_members: int = 5,
    config: TrainConfig | None = None,
    loss_fn: Callable | None = None,
    base_seed: int = 1000,
    group_ids_factory: Callable[[int], tuple[Any, Any, Any | None]] | None = None,
) -> EnsembleFitResult:
    """Train independently initialized ensemble members using distinct deterministic seeds.

    Loader factories are called per member so a shuffled loader may receive an independently seeded
    generator. Parent simulation groups must remain identical across members and disjoint across splits.
    """
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to train ensembles")
    if n_members < 2:
        raise ValueError("n_members must be >=2 for epistemic spread estimation")
    base = config or TrainConfig()
    models, runs, seeds = [], [], []
    for i in range(n_members):
        seed = base_seed + i * 9973
        current = TrainConfig(**{**asdict(base), "seed": seed})
        model = model_factory(seed)
        groups = group_ids_factory(i) if group_ids_factory is not None else (None, None, None)
        train_loader = train_loader_factory(seed)
        val_loader = validation_loader_factory(seed)
        run = fit_model(
            model,
            train_loader,
            val_loader,
            config=current,
            loss_fn=loss_fn,
            train_group_ids=groups[0],
            validation_group_ids=groups[1],
            test_group_ids=groups[2],
        )
        models.append(run.model)
        runs.append(run)
        seeds.append(seed)
    return EnsembleFitResult(models, runs, seeds, asdict(base))


def predict_members(
    models: list[Any], inputs: Any, *, batch_size: int = 128, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict means and optional aleatoric variances for each ensemble member.

    Inputs may be a NumPy array or tensor and must fit in host memory. For very large 3-D cubes,
    run chunked prediction in a dedicated data loader rather than materializing the entire dataset.
    """
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to predict with ensembles")
    if len(models) < 2 or batch_size < 1:
        raise ValueError("at least two models and positive batch_size required")
    x = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs, dtype=torch.float32)
    if x.ndim < 2 or len(x) == 0:
        raise ValueError("inputs must have a nonempty sample dimension")
    mean_members, variance_members = [], []
    all_have_variance = True
    for model in models:
        model = model.to(device)
        model.eval()
        chunks, var_chunks = [], []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                out = model(x[start : start + batch_size].to(device))
                if isinstance(out, dict) and "mean" in out:
                    mean = out["mean"]
                    logvar = out.get("log_variance")
                    if logvar is None:
                        all_have_variance = False
                    else:
                        var_chunks.append(logvar.exp().detach().cpu().numpy())
                elif torch.is_tensor(out):
                    mean = out
                    all_have_variance = False
                else:
                    raise ValueError("model must return tensor or dict containing mean")
                chunks.append(mean.detach().cpu().numpy())
        mean_members.append(np.concatenate(chunks, axis=0))
        if var_chunks:
            variance_members.append(np.concatenate(var_chunks, axis=0))
    means = np.stack(mean_members, axis=0)
    variances = np.stack(variance_members, axis=0) if all_have_variance else None
    return means, variances


def ensemble_summary(
    models: list[Any], inputs: Any, *, batch_size: int = 128, device: str = "cpu"
) -> EnsembleUncertainty:
    """Return ensemble mean and total/epistemic/aleatoric variance decomposition."""
    means, variances = predict_members(models, inputs, batch_size=batch_size, device=device)
    return decompose_ensemble(means, variances)
