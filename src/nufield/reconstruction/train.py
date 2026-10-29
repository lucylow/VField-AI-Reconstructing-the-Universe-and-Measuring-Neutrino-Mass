"""Optional seeded trainer with strict checks and atomic, resumable checkpoints."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from nufield.exceptions import OptionalDependencyError
from nufield.reconstruction.data import validate_paired_fields


def train_unet3d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    optimizer_name: str = "adamw",
    scheduler_name: str = "none",
    lambda_field: float = 1.0,
    lambda_pk: float = 0.0,
    lambda_gradient: float = 0.0,
    lambda_mean: float = 0.0,
    box_size_mpc_h: float | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict:
    """Fit/resume U-Net; epochs is total target epoch count (not additional epochs)."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise OptionalDependencyError("Install nufield-starter[ml] to train the optional U-Net") from exc
    from nufield.reconstruction.network import UNet3D

    xa, ya = validate_paired_fields(x, y)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or seed < 0:
        raise ValueError("epochs/batch_size must be positive, learning_rate > 0, seed >= 0")
    if optimizer_name not in {"adamw", "adam"} or scheduler_name not in {"none", "cosine"}:
        raise ValueError("optimizer_name must be adamw/adam and scheduler_name none/cosine")
    weights = (lambda_field, lambda_pk, lambda_gradient, lambda_mean)
    if not all(np.isfinite(w) and w >= 0 for w in weights) or not any(w > 0 for w in weights):
        raise ValueError("loss weights must be finite, non-negative, and not all zero")
    if lambda_pk > 0 and (box_size_mpc_h is None or not np.isfinite(box_size_mpc_h) or box_size_mpc_h <= 0):
        raise ValueError("lambda_pk > 0 requires documented positive box_size_mpc_h")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Total target epochs may be increased on resume; all other optimizer/data-shape settings
    # remain locked so an incompatible checkpoint cannot be silently reused.
    run_cfg = {"batch_size": int(batch_size), "learning_rate": float(learning_rate), "seed": int(seed), "optimizer": optimizer_name, "scheduler": scheduler_name, "input_shape": list(xa.shape[1:]), "loss_weights": [float(w) for w in weights], "box_size_mpc_h": box_size_mpc_h}
    generator = torch.Generator().manual_seed(seed)
    model = UNet3D().to(device_obj)
    opt_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=learning_rate, weight_decay=1e-4 if optimizer_name == "adamw" else 0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if scheduler_name == "cosine" else None
    history: list[dict[str, float | int]] = []
    start_epoch, data_meta = 0, provenance or {}
    if resume_from is not None:
        # Keep RNG byte states on CPU, then load model/optimizer state onto the selected device.
        saved = torch.load(resume_from, map_location="cpu", weights_only=False)
        if saved.get("training_config") != run_cfg:
            raise ValueError("resume checkpoint training config differs; keep original settings")
        if saved.get("provenance", {}) != data_meta:
            raise ValueError("resume checkpoint data/split provenance differs")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if scheduler is not None:
            if saved.get("scheduler_state") is None:
                raise ValueError("resume checkpoint has no scheduler state")
            scheduler.load_state_dict(saved["scheduler_state"])
            scheduler.T_max = max(int(epochs), int(saved.get("target_epochs", epochs)))
        start_epoch = int(saved["epoch"])
        if start_epoch > epochs:
            raise ValueError("checkpoint epoch exceeds requested total epochs")
        history = list(saved.get("history", []))
        generator.set_state(saved["loader_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
    loader = DataLoader(TensorDataset(torch.from_numpy(xa[:, None]), torch.from_numpy(ya[:, None])), batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        field_mses = []
        for bx, by in loader:
            bx, by = bx.to(device_obj), by.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            field_mse = torch.nn.functional.mse_loss(pred, by)
            if lambda_pk > 0 or lambda_gradient > 0 or lambda_mean > 0:
                from nufield.models.physics_losses import composite_reconstruction_loss
                components = composite_reconstruction_loss(
                    pred, by, box_size_mpc_h=box_size_mpc_h or 1.0,
                    lambda_field=lambda_field, lambda_pk=lambda_pk,
                    lambda_grad=lambda_gradient, lambda_mean=lambda_mean,
                )
                loss = components["total"]
            else:
                loss = lambda_field * field_mse
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            field_mses.append(float(field_mse.detach().cpu()))
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(field_mses)), "train_loss": float(np.mean(losses)), "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if checkpoint_path is not None:
            target = Path(checkpoint_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {"checkpoint_schema": 1, "epoch": epoch + 1, "target_epochs": int(epochs), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict() if scheduler else None, "training_config": run_cfg, "provenance": data_meta, "history": history, "loader_rng_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            fd, tmp = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
            os.close(fd)
            try:
                torch.save(payload, tmp)
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    return {"model": model, "history": history, "start_epoch": start_epoch, "checkpoint": str(checkpoint_path) if checkpoint_path else None, "physics_level": "not specified by trainer; inspect data provenance"}
