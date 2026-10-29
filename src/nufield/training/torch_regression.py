"""Leakage-aware training loop for a heteroscedastic neural regression baseline."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from nufield.data_registry.preprocessing import Standardizer
from nufield.exceptions import OptionalDependencyError
from nufield.models.gaussian_regressor import GaussianRegressor, gaussian_nll
from nufield.models.parameter_scaler import ParameterScaler


@dataclass(frozen=True)
class RegressionTrainConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.05
    epochs: int = 300
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    min_delta: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 42
    device: str = "auto"

    def validate(self):
        if not self.hidden_dims or any(x < 1 for x in self.hidden_dims):
            raise ValueError("hidden_dims must be positive")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs/batch_size/patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("invalid optimizer settings")
        if not 0 <= self.dropout < 1 or self.min_delta < 0:
            raise ValueError("invalid dropout/min_delta")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto/cpu/cuda")


@dataclass
class RegressionFit:
    model: object
    input_scaler: Standardizer
    target_scaler: ParameterScaler
    report: dict
    device: str

    def predict(self, observations: np.ndarray) -> dict[str, np.ndarray]:
        try:
            import torch
        except ImportError as exc:
            raise OptionalDependencyError("Install [ml]") from exc
        x = self.input_scaler.transform(np.asarray(observations))
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            mean, log_var = self.model(tensor)
            mu = mean.cpu().numpy()
            variance = torch.exp(log_var).cpu().numpy()
        return {
            "mean": self.target_scaler.inverse_mean(mu).astype(np.float32),
            "variance": self.target_scaler.inverse_variance(variance).astype(np.float32),
        }


def fit_gaussian_regressor(
    x_train: np.ndarray,
    theta_train: np.ndarray,
    x_validation: np.ndarray,
    theta_validation: np.ndarray,
    *,
    parameter_names: list[str],
    config: RegressionTrainConfig | None = None,
    output_dir: str | Path | None = None,
) -> RegressionFit:
    """Train a point/diagonal-uncertainty baseline using only train statistics and validation loss.

    Do not interpret the diagonal Gaussian scale as calibrated posterior uncertainty without a
    coverage experiment. The final test set must not be passed to this function.
    """
    try:
        import torch
    except ImportError as exc:
        raise OptionalDependencyError(
            "Install nufield-starter[ml] to train PyTorch models"
        ) from exc
    cfg = config or RegressionTrainConfig()
    cfg.validate()
    xtr = np.asarray(x_train, dtype=np.float32)
    ytr = np.asarray(theta_train, dtype=np.float32)
    xva = np.asarray(x_validation, dtype=np.float32)
    yva = np.asarray(theta_validation, dtype=np.float32)
    if xtr.ndim != 2 or xva.ndim != 2 or xtr.shape[1] != xva.shape[1]:
        raise ValueError("X must be 2-D with matching feature dimensions")
    if (
        ytr.ndim != 2
        or yva.ndim != 2
        or ytr.shape[1] != yva.shape[1]
        or ytr.shape[1] != len(parameter_names)
    ):
        raise ValueError("theta/parameter dimensions mismatch")
    if len(xtr) < 10 or len(xva) < 3:
        raise ValueError("need at least 10 train and 3 validation simulations")
    if len(xtr) != len(ytr) or len(xva) != len(yva):
        raise ValueError("row counts mismatch")
    if not all(np.isfinite(z).all() for z in (xtr, ytr, xva, yva)):
        raise ValueError("all arrays must be finite")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device in {"auto", "cuda"}:
        device = "cuda"
    else:
        device = "cpu" if cfg.device == "auto" else cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    xs = Standardizer().fit(xtr)
    ys = ParameterScaler().fit(ytr)
    xtr_s = xs.transform(xtr)
    xva_s = xs.transform(xva)
    ytr_s = ys.transform(ytr)
    yva_s = ys.transform(yva)
    tx = torch.as_tensor(xtr_s, dtype=torch.float32)
    ty = torch.as_tensor(ytr_s, dtype=torch.float32)
    vx = torch.as_tensor(xva_s, dtype=torch.float32, device=device)
    vy = torch.as_tensor(yva_s, dtype=torch.float32, device=device)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tx, ty),
        batch_size=min(cfg.batch_size, len(tx)),
        shuffle=True,
        generator=generator,
    )
    model = GaussianRegressor(xtr.shape[1], ytr.shape[1], cfg.hidden_dims, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    best = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, log_var = model(xb)
            loss = gaussian_nll(mean, log_var, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            mv, lv = model(vx)
            val_loss = float(gaussian_nll(mv, lv, vy).cpu())
        row = {
            "epoch": epoch + 1,
            "train_nll": float(np.mean(train_losses)),
            "validation_nll": val_loss,
        }
        history.append(row)
        if val_loss < best - cfg.min_delta:
            best = val_loss
            best_epoch = epoch + 1
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    report = {
        "model": "heteroscedastic Gaussian MLP baseline",
        "n_train": len(xtr),
        "n_validation": len(xva),
        "input_dim": int(xtr.shape[1]),
        "target_names": parameter_names,
        "best_validation_nll_standardized_theta": float(best),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "wall_seconds": time.perf_counter() - start,
        "device": device,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "history": history,
        "warning": "Diagonal Gaussian regression uncertainty is not automatically a calibrated Bayesian posterior; compare with NPE and run empirical coverage/SBC on independent simulations.",
    }
    fit = RegressionFit(model, xs, ys, report, device)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "regression_report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            out / "preprocessing.npz",
            x_mean=np.asarray(xs.mean_),
            x_scale=np.asarray(xs.scale_),
            theta_mean=np.asarray(ys.mean_),
            theta_scale=np.asarray(ys.scale_),
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(xtr.shape[1]),
                "target_dim": int(ytr.shape[1]),
                "hidden_dims": cfg.hidden_dims,
                "dropout": cfg.dropout,
            },
            out / "gaussian_regressor.pt",
        )
    return fit
