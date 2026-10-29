"""Optional adapter that trains SBI neural posterior estimation on packed observations.

We recommend NPE + a custom embedding net for high-dimensional fields as a first neural posterior
baseline. It must be compared to simpler likelihood/power-spectrum inference and calibrated using
independent simulation-based calibration runs. This code does not claim calibration by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.exceptions import OptionalDependencyError


@dataclass(frozen=True)
class NPEConfig:
    density_estimator: str = "nsf"
    hidden_features: int = 96
    num_transforms: int = 5
    max_num_epochs: int = 200
    training_batch_size: int = 64
    validation_fraction: float = 0.1
    stop_after_epochs: int = 20
    seed: int = 123
    device: str = "auto"
    show_train_summary: bool = False

    def validate(self) -> None:
        if self.density_estimator not in {"nsf", "maf", "maf_rqs", "mdn", "made"}:
            raise ValueError("density_estimator must be nsf/maf/maf_rqs/mdn/made")
        if (
            min(
                self.hidden_features,
                self.num_transforms,
                self.max_num_epochs,
                self.training_batch_size,
                self.stop_after_epochs,
            )
            < 1
        ):
            raise ValueError("training dimensions and epoch counts must be positive")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0,0.5)")


@dataclass
class NPEResult:
    posterior: object
    inference: object
    training_summary: dict
    warning: str = "Posterior is a learned approximation; run SBC/coverage and misspecification tests before scientific use."


def fit_npe(
    observations: np.ndarray,
    theta: np.ndarray,
    *,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    parameter_names: list[str],
    embedding_net,
    config: NPEConfig | None = None,
) -> NPEResult:
    """Train NPE with explicit prior bounds and a user-supplied embedding model.

    Inputs must be simulations from the stated prior or handled with a documented proposal/prior
    correction. Train/validation/test splitting must happen before calling this function. For true
    held-out evaluation pass only training simulations here and keep test simulations untouched.
    """
    try:
        import torch
        from sbi.inference import NPE
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform
    except ImportError as exc:
        raise OptionalDependencyError(
            "Install nufield-starter[sbi] to train neural posterior estimation"
        ) from exc
    cfg = config or NPEConfig()
    cfg.validate()
    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(theta, dtype=np.float32)
    lo = np.asarray(prior_low, dtype=np.float32)
    hi = np.asarray(prior_high, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("observations must be [N,D] and theta [N,P]")
    if y.shape[1] != len(parameter_names) or lo.shape != (y.shape[1],) or hi.shape != (y.shape[1],):
        raise ValueError("parameter/prior dimensions mismatch")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("training tensors must be finite")
    if not (lo < hi).all():
        raise ValueError("every prior lower bound must be below upper bound")
    if (y < lo).any() or (y > hi).any():
        raise ValueError("training theta contains values outside declared prior bounds")
    if len(y) < 50:
        raise ValueError(
            "at least 50 simulations are required for this training helper; this is a floor, not a scientific sufficiency claim"
        )
    seed = int(cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and cfg.device in {"auto", "cuda"}:
        device = "cuda"
    else:
        device = "cpu" if cfg.device == "auto" else cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    t = torch.as_tensor(y, dtype=torch.float32, device="cpu")
    obs = torch.as_tensor(x, dtype=torch.float32, device="cpu")
    prior = BoxUniform(low=torch.as_tensor(lo), high=torch.as_tensor(hi))
    density_builder = posterior_nn(
        model=cfg.density_estimator,
        embedding_net=embedding_net,
        hidden_features=cfg.hidden_features,
        num_transforms=cfg.num_transforms,
    )
    inference = NPE(
        prior=prior,
        density_estimator=density_builder,
        device=device,
        show_progress_bars=cfg.show_train_summary,
    )
    inference = inference.append_simulations(t, obs)
    estimator = inference.train(
        max_num_epochs=cfg.max_num_epochs,
        training_batch_size=cfg.training_batch_size,
        validation_fraction=cfg.validation_fraction,
        stop_after_epochs=cfg.stop_after_epochs,
        show_train_summary=cfg.show_train_summary,
    )
    posterior = inference.build_posterior(estimator)
    return NPEResult(
        posterior=posterior,
        inference=inference,
        training_summary={
            "method": "NPE",
            "density_estimator": cfg.density_estimator,
            "n_train": int(len(y)),
            "observation_width": int(x.shape[1]),
            "n_parameters": int(y.shape[1]),
            "parameter_names": parameter_names,
            "prior_low": lo.tolist(),
            "prior_high": hi.tolist(),
            "seed": seed,
            "device": device,
            "max_num_epochs": cfg.max_num_epochs,
            "validation_fraction": cfg.validation_fraction,
            "warning": "not calibrated until independent validation",
        },
    )


def sample_posterior(
    result: NPEResult,
    observed_packed: np.ndarray,
    *,
    n_samples: int = 5000,
    seed: int = 1,
    show_progress_bars: bool = False,
) -> np.ndarray:
    """Draw posterior samples for one packed observation and return a finite [S,P] array."""
    try:
        import torch
    except ImportError as exc:
        raise OptionalDependencyError("Install nufield-starter[ml]") from exc
    x = np.asarray(observed_packed, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[0] != 1:
            raise ValueError("observed_packed must contain exactly one observation")
        x = x[0]
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("observed_packed must be one finite vector")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    torch.manual_seed(seed)
    samples = result.posterior.sample(
        (int(n_samples),), x=torch.as_tensor(x), show_progress_bars=show_progress_bars
    )
    arr = samples.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise RuntimeError("posterior sampler returned invalid samples")
    return arr
