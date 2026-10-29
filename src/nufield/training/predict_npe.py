"""Apply a saved νField SBI posterior to a single explicitly mapped observation bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nufield.data_registry.preprocessing import Standardizer
from nufield.exceptions import OptionalDependencyError
from nufield.models.hybrid_encoder import pack_observations


def predict_one(
    posterior_path: str | Path,
    training_config_path: str | Path,
    preprocessing_path: str | Path,
    observation_path: str | Path,
    output_path: str | Path,
    *,
    n_samples: int = 10000,
    seed: int = 123,
    device: str = "cpu",
) -> dict:
    """Predict with the exact recorded architecture and preprocessing contract.

    A model cannot be applied to a new data modality or field shape merely because the array can be
    flattened. Inputs must follow the training configuration and physical observation model.
    """
    try:
        import torch
        from sbi.inference import DirectPosterior
    except ImportError as exc:
        raise OptionalDependencyError(
            "Install nufield-starter[sbi] to use the saved posterior"
        ) from exc
    cfg = json.loads(Path(training_config_path).read_text())
    if cfg.get("physics_level") == "toy":
        raise ValueError(
            "Refusing to produce an apparent physics prediction from a toy-trained posterior"
        )
    if n_samples < 2:
        raise ValueError("n_samples must be >=2")
    architecture = cfg.get("architecture", "hybrid")
    meta = cfg["observation_pack"]
    with np.load(observation_path, allow_pickle=False) as obs:
        field_key = cfg["field_key"]
        spectrum_key = cfg["spectrum_key"]
        if field_key not in obs or spectrum_key not in obs:
            raise KeyError("observation NPZ lacks configured field/spectrum key")
        fields = np.asarray(obs[field_key], dtype=np.float32)
        spectrum = np.asarray(obs[spectrum_key], dtype=np.float32)
        lens_key = cfg.get("lensing_key")
        if lens_key and lens_key not in obs:
            raise KeyError(
                f"model was trained with required lensing key {lens_key!r}; input lacks that modality"
            )
        lensing = np.asarray(obs[lens_key], dtype=np.float32) if lens_key else None
    if fields.ndim == 4:
        fields = fields[:, None]
    if fields.ndim != 5:
        raise ValueError("field input must be [N,D,H,W] or [N,C,D,H,W]")
    if spectrum.ndim != 2:
        raise ValueError("spectrum must be [N,K]")
    if architecture == "hybrid":
        if lensing is None:
            lens_shape = tuple(meta["lensing_shape"][-2:])
            packed = pack_observations(fields, spectrum, lensing=None, lensing_shape=lens_shape)
        else:
            packed = pack_observations(fields, spectrum, lensing=lensing)
        now = packed.metadata()
        for key in ("packed_width", "field_shape", "spectrum_dim", "lensing_shape"):
            if now[key] != meta[key]:
                raise ValueError(
                    f"observation {key}={now[key]!r} differs from training configuration {meta[key]!r}"
                )
        raw_input = packed.values
    elif architecture == "spectrum_mlp":
        if int(spectrum.shape[1]) != int(meta["spectrum_dim"]):
            raise ValueError("spectrum feature width differs from training configuration")
        raw_input = spectrum
    elif architecture in {"cnn3d", "fno3d"}:
        expected = tuple(meta["field_shape"])
        if tuple(fields.shape[1:]) != expected:
            raise ValueError(
                f"field shape {tuple(fields.shape[1:])} differs from configured {expected}"
            )
        raw_input = fields
    else:
        raise ValueError(
            f"Unknown saved architecture {architecture!r}; retrain/export with a supported architecture"
        )
    with np.load(preprocessing_path, allow_pickle=False) as z:
        mean = np.asarray(z["mean"], dtype=np.float64)
        scale = np.asarray(z["scale"], dtype=np.float64)
        kind = str(z["kind"].item()) if "kind" in z else "featurewise"
    if kind == "channel":
        if raw_input.ndim != 5:
            raise ValueError("channel preprocessing requires raw field tensors")
        x = ((raw_input - mean) / scale).astype(np.float32).reshape(len(raw_input), -1)
    elif kind == "featurewise":
        if raw_input.ndim != 2:
            raise ValueError("featurewise preprocessing requires 2-D observations")
        scaler = Standardizer(mean_=mean, scale_=scale)
        x = scaler.transform(raw_input)
    else:
        raise ValueError(f"Unsupported preprocessing kind {kind!r}")
    if x.shape[0] != 1:
        raise ValueError("observation NPZ must contain exactly one observation row")
    if x.shape[1] != int(meta["packed_width"]):
        raise ValueError("observation vector width differs from training configuration")
    # `sbi` files use Python serialization; only load artifacts produced by trusted workflows.
    posterior = DirectPosterior.load(str(posterior_path), map_location=device)
    torch.manual_seed(seed)
    samples = posterior.sample(
        (n_samples,), x=torch.as_tensor(x[0], dtype=torch.float32), show_progress_bars=False
    )
    array = samples.detach().cpu().numpy().astype(np.float32)
    names = cfg["parameter_names"]
    summary = {
        "n_samples": n_samples,
        "parameter_names": names,
        "parameters": {},
        "architecture": architecture,
        "physics_level": cfg.get("physics_level"),
        "scientific_status": "UNVALIDATED_UNLESS_INDEPENDENT_VALIDATION_PASSES",
        "warning": "Posterior samples are conditional on the saved simulator, prior, preprocessing and model. This command does not establish physical validity or calibration.",
    }
    for j, name in enumerate(names):
        summary["parameters"][name] = {
            "q025": float(
                np.quantile(array[:, j], 0.025),
            ),
            "q16": float(np.quantile(array[:, j], 0.16)),
            "median": float(np.quantile(array[:, j], 0.5)),
            "q84": float(np.quantile(array[:, j], 0.84)),
            "q975": float(np.quantile(array[:, j], 0.975)),
            "mean": float(array[:, j].mean()),
            "std": float(array[:, j].std()),
        }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(out.with_suffix(".npz"), samples=array, parameter_names=np.asarray(names))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-predict-npe",
        description="Sample a trained νField SBI posterior for one mapped observation",
    )
    p.add_argument("--posterior", required=True)
    p.add_argument("--training-config", required=True)
    p.add_argument("--preprocessing", required=True)
    p.add_argument("--observation", required=True)
    p.add_argument("--output", default="outputs/prediction/posterior")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args(argv)
    result = predict_one(
        args.posterior,
        args.training_config,
        args.preprocessing,
        args.observation,
        args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).with_suffix(".json")),
                "n_samples": result["n_samples"],
                "status": result["scientific_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
