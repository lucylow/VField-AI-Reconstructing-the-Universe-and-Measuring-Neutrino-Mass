"""Typed settings for reproducible field reconstruction benchmarks."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from nufield.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    dataset: str
    output_dir: str = "outputs/reconstruction"
    target_key: str = "initial_field"
    galaxy_key: str = "galaxy_overdensity"
    lensing_key: str = "weak_lensing_map"
    cmb_key: str = "cmb_features"
    group_key: str = "group_id"
    simulation_id_key: str = "simulation_id"
    survey_mask_key: str = "survey_mask"
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    ridge_alpha: float = 1e-6
    neural_epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 1e-3
    architecture: str = "unet3d"
    optimizer: str = "adamw"
    scheduler: str = "none"
    device: str = "cpu"
    lambda_field: float = 1.0
    lambda_pk: float = 0.0
    lambda_gradient: float = 0.0
    lambda_mean: float = 0.0
    box_size_mpc_h: float | None = None
    cosmological_parameters: tuple[tuple[str, float], ...] = ()
    run_neural: bool = True
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset.strip():
            raise ConfigurationError("dataset path must not be empty")
        numeric = (self.train_fraction, self.validation_fraction, self.test_fraction,
                   self.ridge_alpha, self.learning_rate, self.lambda_field, self.lambda_pk,
                   self.lambda_gradient, self.lambda_mean)
        if not all(math.isfinite(v) for v in numeric) or (
            self.box_size_mpc_h is not None and not math.isfinite(self.box_size_mpc_h)
        ):
            raise ConfigurationError("numeric settings must be finite")
        if self.seed < 0:
            raise ConfigurationError("seed must be non-negative")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(not 0 < v < 1 for v in fractions) or abs(sum(fractions) - 1) > 1e-8:
            raise ConfigurationError("split fractions must each be in (0,1) and sum to 1")
        if self.ridge_alpha < 0:
            raise ConfigurationError("ridge_alpha must be non-negative")
        if self.neural_epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ConfigurationError("neural_epochs/batch_size must be positive and learning_rate > 0")
        if self.optimizer not in {"adamw", "adam"}:
            raise ConfigurationError("optimizer must be 'adamw' or 'adam'")
        if self.scheduler not in {"none", "cosine"}:
            raise ConfigurationError("scheduler must be 'none' or 'cosine'")
        if self.architecture != "unet3d":
            raise ConfigurationError("architecture currently supports only 'unet3d'")
        weights = (self.lambda_field, self.lambda_pk, self.lambda_gradient, self.lambda_mean)
        if any(w < 0 for w in weights) or not any(w > 0 for w in weights):
            raise ConfigurationError("loss weights must be non-negative with at least one positive weight")
        if self.lambda_pk > 0 and (self.box_size_mpc_h is None or self.box_size_mpc_h <= 0):
            raise ConfigurationError("lambda_pk > 0 requires a documented positive box_size_mpc_h")
        try:
            raw_pairs = self.cosmological_parameters.items() if isinstance(self.cosmological_parameters, dict) else self.cosmological_parameters
            pairs = tuple((str(name), float(value)) for name, value in raw_pairs)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("cosmological_parameters must be name/value pairs") from exc
        object.__setattr__(self, "cosmological_parameters", pairs)
        names = [item[0] for item in pairs]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ConfigurationError("cosmological parameter names must be non-empty and unique")
        if not all(math.isfinite(value) for _, value in pairs):
            raise ConfigurationError("cosmological parameter values must be finite")

    def to_dict(self) -> dict:
        return asdict(self)


def load_reconstruction_config(path: str | Path) -> ReconstructionConfig:
    """Load YAML config and reject unknown keys with an actionable message."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = raw.get("reconstruction", raw)
    allowed = set(ReconstructionConfig.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown reconstruction setting(s): {sorted(unknown)}")
    raw = dict(raw)
    cosmology = raw.get("cosmological_parameters", ())
    try:
        if isinstance(cosmology, dict):
            raw["cosmological_parameters"] = tuple(sorted((str(k), float(v)) for k, v in cosmology.items()))
        elif cosmology:
            raw["cosmological_parameters"] = tuple((str(k), float(v)) for k, v in cosmology)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("cosmological_parameters must be a mapping of finite numbers") from exc
    try:
        return ReconstructionConfig(**raw)
    except TypeError as exc:
        raise ConfigurationError(f"Invalid reconstruction config: {exc}") from exc
