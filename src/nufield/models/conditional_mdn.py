"""Conditional mixture-density network baseline for low-dimensional cosmological parameters.

This is an alternative neural posterior baseline to `sbi`'s NPE. It predicts a mixture density in
transformed parameter space from an observation embedding. Use for transparent controlled studies,
not as an automatic replacement for SBI's richer flow estimators. Calibration and support checks
are mandatory before scientific interpretation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from nufield.exceptions import OptionalDependencyError


@dataclass(frozen=True)
class MDNConfig:
    n_components: int = 8
    hidden_dim: int = 128
    n_hidden_layers: int = 2
    dropout: float = 0.05
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 250
    patience: int = 25
    min_delta: float = 1e-4
    seed: int = 17
    device: str = "cpu"

    def validate(self):
        if (
            min(
                self.n_components,
                self.hidden_dim,
                self.n_hidden_layers,
                self.batch_size,
                self.max_epochs,
                self.patience,
            )
            < 1
        ):
            raise ValueError("MDN dimensions and epoch counts must be positive")
        if (
            not 0 <= self.dropout < 1
            or self.learning_rate <= 0
            or self.weight_decay < 0
            or self.min_delta < 0
        ):
            raise ValueError("invalid optimizer/dropout configuration")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu/cuda/mps")

    def to_dict(self):
        return asdict(self)


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise OptionalDependencyError(
            "Install nufield-starter[ml] to use the mixture-density network"
        ) from exc
    return torch, nn


def build_conditional_mdn(input_dim: int, target_dim: int, config: MDNConfig | None = None):
    torch, nn = _torch()
    cfg = config or MDNConfig()
    cfg.validate()
    if min(input_dim, target_dim) < 1:
        raise ValueError("input_dim and target_dim must be positive")

    class ConditionalMixtureDensityNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            width = input_dim
            for _ in range(cfg.n_hidden_layers):
                layers.extend(
                    [nn.Linear(width, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()]
                )
                if cfg.dropout:
                    layers.append(nn.Dropout(cfg.dropout))
                width = cfg.hidden_dim
            self.encoder = nn.Sequential(*layers)
            self.logits = nn.Linear(width, cfg.n_components)
            self.means = nn.Linear(width, cfg.n_components * target_dim)
            self.log_scales = nn.Linear(width, cfg.n_components * target_dim)
            self.target_dim = target_dim
            self.n_components = cfg.n_components

        def forward(self, x):
            if x.ndim != 2 or x.shape[1] != input_dim:
                raise ValueError(f"expected [B,{input_dim}] observations")
            if not torch.isfinite(x).all():
                raise ValueError("MDN input contains NaN/Inf")
            h = self.encoder(x)
            logits = self.logits(h)
            means = self.means(h).reshape(-1, cfg.n_components, target_dim)
            log_scales = (
                self.log_scales(h).reshape(-1, cfg.n_components, target_dim).clamp(-6.0, 3.0)
            )
            return logits, means, log_scales

        def log_prob(self, x, y):
            if y.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[1] != target_dim:
                raise ValueError("target must be [B,P]")
            logits, means, logs = self(x)
            scale = logs.exp()
            z = (y[:, None, :] - means) / scale
            component = -0.5 * (z.square() + 2 * logs + float(np.log(2 * np.pi))).sum(dim=-1)
            return torch.logsumexp(torch.log_softmax(logits, dim=-1) + component, dim=-1)

        @torch.no_grad()
        def sample(self, x, n_samples: int, *, seed: int = 0):
            if n_samples < 1:
                raise ValueError("n_samples must be positive")
            logits, means, logs = self(x)
            probs = torch.softmax(logits, dim=-1)
            generator = torch.Generator(device=probs.device)
            generator.manual_seed(int(seed))
            mix = torch.multinomial(probs, n_samples, replacement=True, generator=generator)
            b = x.shape[0]
            row = torch.arange(b, device=x.device)[:, None]
            chosen_mu = means[row, mix, :]
            chosen_log = logs[row, mix, :]
            noise = torch.randn(
                chosen_mu.shape, dtype=chosen_mu.dtype, device=chosen_mu.device, generator=generator
            )
            return chosen_mu + chosen_log.exp() * noise

    model = ConditionalMixtureDensityNetwork()
    return model


def fit_conditional_mdn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    config: MDNConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit using explicit external validation arrays. There is no hidden test split here."""
    torch, _ = _torch()
    cfg = config or MDNConfig()
    cfg.validate()
    xt = np.asarray(x_train, dtype=np.float32)
    yt = np.asarray(y_train, dtype=np.float32)
    xv = np.asarray(x_validation, dtype=np.float32)
    yv = np.asarray(y_validation, dtype=np.float32)
    if xt.ndim != 2 or yt.ndim != 2 or xv.ndim != 2 or yv.ndim != 2:
        raise ValueError("MDN workflow expects matrices X[N,F], theta[N,P]")
    if len(xt) < 4 or len(xv) < 2 or len(yt) != len(xt) or len(yv) != len(xv):
        raise ValueError("insufficient/misaligned train or validation arrays")
    if xt.shape[1] != xv.shape[1] or yt.shape[1] != yv.shape[1]:
        raise ValueError("train/validation feature/target dimensions differ")
    if not all(np.isfinite(z).all() for z in (xt, yt, xv, yv)):
        raise ValueError("MDN arrays must be finite")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if cfg.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS requested but unavailable")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    model = build_conditional_mdn(xt.shape[1], yt.shape[1], cfg).to(device)
    tx = torch.as_tensor(xt, device=device)
    ty = torch.as_tensor(yt, device=device)
    vx = torch.as_tensor(xv, device=device)
    vy = torch.as_tensor(yv, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    best = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = rng.permutation(len(tx))
        train_losses = []
        for start in range(0, len(perm), cfg.batch_size):
            ix = torch.as_tensor(perm[start : start + cfg.batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = -model.log_prob(tx[ix], ty[ix]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MDN training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float((-model.log_prob(vx, vy).mean()).cpu())
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_nll": train_loss, "validation_nll": val_loss})
        if val_loss < best - cfg.min_delta:
            best = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("MDN never produced a finite improving validation loss")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    report = {
        "config": cfg.to_dict(),
        "n_train": len(xt),
        "n_validation": len(xv),
        "input_dim": int(xt.shape[1]),
        "target_dim": int(yt.shape[1]),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_nll_transformed_space": best,
        "history": history,
        "validation_partition_used_for_early_stopping": True,
        "test_partition_used": False,
        "density_model": "diagonal Gaussian mixture in transformed standardized target coordinates",
        "scientific_approval": False,
        "warning": "MDN training likelihood and validation NLL do not establish posterior calibration or physics validity.",
    }
    return model, report


def draw_physical_posterior(
    model,
    x: np.ndarray,
    *,
    n_samples: int = 1000,
    seed: int = 0,
    device: str = "cpu",
    feature_state: dict,
    target_state: dict,
    bounds_state: dict,
) -> np.ndarray:
    """Sample in standardized-logit target coordinates and map back to declared finite bounds."""
    from nufield.data_registry.preprocessing import Standardizer
    from nufield.models.bounded_transform import BoundedParameterTransform

    torch, _ = _torch()
    xsc = Standardizer.from_state_dict(feature_state)
    ysc = Standardizer.from_state_dict(target_state)
    bound = BoundedParameterTransform.from_state_dict(bounds_state)
    features = np.asarray(x, dtype=np.float32)
    if features.ndim == 1:
        features = features[None, :]
    xs = xsc.transform(features)
    t = torch.as_tensor(xs, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        scaled = model.sample(t, n_samples, seed=seed).cpu().numpy()
    original_shape = scaled.shape
    z = ysc.inverse_transform(scaled.reshape(-1, model.target_dim))
    physical = bound.inverse(z)
    return physical.reshape(original_shape)


def load_mdn_checkpoint(path: str, *, device: str = "cpu"):
    """Restore a locally trained MDN checkpoint from an explicitly trusted artifact."""
    from pathlib import Path

    import torch

    try:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    except TypeError:  # compatibility with PyTorch releases without weights_only
        checkpoint = torch.load(Path(path), map_location=device)
    required = {
        "state_dict",
        "model_config",
        "input_dim",
        "target_dim",
        "parameter_names",
        "feature_scaler",
        "target_scaler",
        "bounded_transform",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing fields {sorted(missing)}")
    cfg = MDNConfig(**checkpoint["model_config"])
    model = build_conditional_mdn(int(checkpoint["input_dim"]), int(checkpoint["target_dim"]), cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, checkpoint
