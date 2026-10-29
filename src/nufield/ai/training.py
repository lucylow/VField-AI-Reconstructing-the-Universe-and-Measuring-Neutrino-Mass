"""Explicit, early-stopping supervised training loop for optional PyTorch models.

The trainer expects the caller to create group-disjoint train, validation and test partitions.
It accepts only train and validation loaders: the test loader is intentionally not used here so
that model selection cannot silently become test-set tuning.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nufield.ai.losses import gaussian_nll
from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set Python, NumPy and torch seeds; record deterministic settings in experiment metadata."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda device and reject unavailable explicit accelerators."""
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to train neural models")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"} and not requested.startswith("cuda:"):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is not available")
    return requested


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    device: str = "auto"
    seed: int = 123
    deterministic: bool = True
    amp: bool = False
    scheduler_patience: int = 7
    scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-6
    checkpoint_path: str | None = None
    history_path: str | None = None
    early_stopping_metric: str = "val_loss"

    def validate(self) -> None:
        if min(self.max_epochs, self.patience, self.scheduler_patience) < 1:
            raise ValueError("epoch and patience values must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("invalid optimizer settings")
        if not 0 < self.scheduler_factor < 1 or self.min_learning_rate <= 0:
            raise ValueError("invalid scheduler settings")
        if self.min_delta < 0:
            raise ValueError("min_delta must be nonnegative")
        if self.early_stopping_metric != "val_loss":
            raise ValueError(
                "only val_loss is supported for early stopping; do not use test metrics"
            )


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    seconds: float
    gradient_norm_mean: float


@dataclass
class TrainResult:
    model: Any
    history: list[EpochRecord]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool
    device: str
    config: dict[str, Any]
    warning: str = (
        "Evaluate the selected checkpoint once on a locked, group-disjoint test partition."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": [asdict(x) for x in self.history],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stopped_early": self.stopped_early,
            "device": self.device,
            "config": self.config,
            "warning": self.warning,
        }


def _move(value: Any, device: str) -> Any:
    if torch is not None and torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    return value


def _extract_batch(batch: Any, device: str):
    batch = _move(batch, device)
    if isinstance(batch, dict):
        if "target" not in batch:
            raise ValueError("dict batches must contain a 'target' entry")
        target = batch["target"]
        # Batch metadata must not accidentally become a model feature. Keep only declared inputs.
        metadata_keys = {"target", "group_id", "group_ids", "source_id", "metadata", "provenance"}
        inputs = {k: v for k, v in batch.items() if k not in metadata_keys}
        if len(inputs) == 1 and "inputs" in inputs:
            inputs = inputs["inputs"]
        return inputs, target
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("each batch must be (inputs, target) or a dict with target key")


def _call_model(model, inputs):
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _loss(output, target, loss_fn: Callable | None):
    if loss_fn is not None:
        value = loss_fn(output, target)
    elif isinstance(output, dict) and "mean" in output and "log_variance" in output:
        value = gaussian_nll(output, target)
    elif torch is not None and torch.is_tensor(output):
        value = torch.nn.functional.mse_loss(output, target)
    else:
        raise ValueError("provide loss_fn unless model returns mean/log_variance or a tensor")
    if not torch.is_tensor(value) or value.ndim != 0:
        raise ValueError("loss_fn must return a scalar torch tensor")
    if not torch.isfinite(value):
        raise FloatingPointError("loss became NaN or infinite")
    return value


def _atomic_torch_save(state: dict[str, Any], path: str | Path) -> None:
    if torch is None:
        raise OptionalDependencyError("PyTorch is required to save checkpoints")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        torch.save(state, temp_name)
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def fit_model(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    *,
    config: TrainConfig | None = None,
    loss_fn: Callable | None = None,
    train_group_ids: Any | None = None,
    validation_group_ids: Any | None = None,
    test_group_ids: Any | None = None,
) -> TrainResult:
    """Fit a PyTorch model with early stopping and best-checkpoint restoration.

    When group IDs are provided, validation and test groups are checked for overlap with training.
    Supplying IDs is recommended; for cosmological simulation data, parent simulation identifiers
    should be used rather than row IDs or map patch IDs.
    """
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to use fit_model")
    cfg = config or TrainConfig()
    cfg.validate()
    if (train_group_ids is None) != (validation_group_ids is None):
        raise ValueError("train_group_ids and validation_group_ids must be provided together")
    if train_group_ids is not None:
        tr = set(np.asarray(train_group_ids).astype(str).tolist())
        va = set(np.asarray(validation_group_ids).astype(str).tolist())
        if not tr or not va or tr.intersection(va):
            raise ValueError(
                "training and validation parent groups must be nonempty and disjoint; overlap is forbidden"
            )
        if test_group_ids is not None and tr.intersection(
            set(np.asarray(test_group_ids).astype(str).tolist())
        ):
            raise ValueError("training and test parent groups overlap")
        if test_group_ids is not None and va.intersection(
            set(np.asarray(test_group_ids).astype(str).tolist())
        ):
            raise ValueError("validation and test parent groups overlap")
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    device = resolve_device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
        min_lr=cfg.min_learning_rate,
    )
    amp_enabled = bool(cfg.amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    history: list[EpochRecord] = []
    stopped_early = False

    for epoch in range(1, cfg.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        train_losses: list[float] = []
        grad_norms: list[float] = []
        for batch in train_loader:
            inputs, target = _extract_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = _call_model(model, inputs)
                train_loss = _loss(output, target, loss_fn)
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("gradient norm became NaN or infinite")
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(train_loss.detach().cpu()))
            grad_norms.append(float(grad_norm.detach().cpu()))
        if not train_losses:
            raise ValueError("training loader yielded no batches")

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in validation_loader:
                inputs, target = _extract_batch(batch, device)
                output = _call_model(model, inputs)
                val_losses.append(float(_loss(output, target, loss_fn).detach().cpu()))
        if not val_losses:
            raise ValueError("validation loader yielded no batches")
        train_mean, val_mean = float(np.mean(train_losses)), float(np.mean(val_losses))
        scheduler.step(val_mean)
        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append(
            EpochRecord(
                epoch,
                train_mean,
                val_mean,
                lr_now,
                time.perf_counter() - started,
                float(np.mean(grad_norms)),
            )
        )
        if val_mean < best_loss - cfg.min_delta:
            best_loss, best_epoch, bad_epochs = val_mean, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if cfg.checkpoint_path:
                _atomic_torch_save(
                    {
                        "state_dict": best_state,
                        "epoch": epoch,
                        "val_loss": best_loss,
                        "config": asdict(cfg),
                    },
                    cfg.checkpoint_path,
                )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                stopped_early = True
                break

    model.load_state_dict(best_state)
    result = TrainResult(model, history, best_epoch, best_loss, stopped_early, device, asdict(cfg))
    if cfg.history_path:
        path = Path(cfg.history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return result


def load_checkpoint(model: Any, path: str | Path, *, device: str = "cpu", strict: bool = True):
    """Load a checkpoint's model weights, preferring safe ``weights_only`` deserialization."""
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to load model checkpoints")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)
    return model.to(device)
