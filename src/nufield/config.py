"""Validated configuration for toy experiments and small-scale benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from nufield.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    n_sims: int = 160
    grid_size: int = 12
    box_size_mpc_h: float = 500.0
    mean_counts_per_cell: float = 4.0
    seed: int = 7
    n_k_bins: int = 10

    def __post_init__(self):
        if self.n_sims < 20:
            raise ConfigurationError(
                "n_sims must be >= 20 to make even a smoke-test split meaningful"
            )
        if self.grid_size < 8:
            raise ConfigurationError("grid_size must be >= 8")
        if self.box_size_mpc_h <= 0:
            raise ConfigurationError("box_size_mpc_h must be > 0")
        if self.mean_counts_per_cell <= 0:
            raise ConfigurationError("mean_counts_per_cell must be > 0")
        if self.n_k_bins < 4:
            raise ConfigurationError("n_k_bins must be >= 4")


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    bandwidth: float = 1.0
    top_k: int = 128
    posterior_samples: int = 2000

    def __post_init__(self):
        if self.bandwidth <= 0:
            raise ConfigurationError("bandwidth must be positive")
        if self.top_k < 1:
            raise ConfigurationError("top_k must be >= 1")
        if self.posterior_samples < 10:
            raise ConfigurationError("posterior_samples must be >= 10")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_fraction: float = 0.65
    validation_fraction: float = 0.15
    test_fraction: float = 0.20

    def __post_init__(self):
        vals = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(v <= 0 or v >= 1 for v in vals):
            raise ConfigurationError("split fractions must each be in (0, 1)")
        if abs(sum(vals) - 1.0) > 1e-8:
            raise ConfigurationError("split fractions must sum to 1")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    simulation: SimulationConfig = SimulationConfig()
    inference: InferenceConfig = InferenceConfig()
    split: SplitConfig = SplitConfig()
    physics_level: str = "toy"

    def __post_init__(self):
        if self.physics_level != "toy":
            raise ConfigurationError(
                "Starter config only supports physics_level='toy'. A validated physics backend "
                "must be implemented and independently checked before changing this label."
            )

    def to_dict(self):
        return {
            "physics_level": self.physics_level,
            "simulation": vars_dict(self.simulation),
            "inference": vars_dict(self.inference),
            "split": vars_dict(self.split),
        }


def vars_dict(obj):
    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate YAML config. Unknown top-level keys are rejected."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    raw = raw.get("experiment", raw)
    allowed = {"simulation", "inference", "split", "physics_level", "project", "features"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown configuration section(s): {sorted(unknown)}")
    sim = dict(raw.get("simulation", {}))
    feat = dict(raw.get("features", {}))
    if "n_k_bins" in feat:
        sim["n_k_bins"] = feat["n_k_bins"]
    return ExperimentConfig(
        simulation=SimulationConfig(**sim),
        inference=InferenceConfig(**raw.get("inference", {})),
        split=SplitConfig(**raw.get("split", {})),
        physics_level=raw.get("physics_level", "toy"),
    )


def config_sha256(config: ExperimentConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
