"""Generate bounded posterior samples from a locally trained conditional MDN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nufield.models.conditional_mdn import draw_physical_posterior, load_mdn_checkpoint


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-predict-mdn", description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--observations", required=True, help="NPZ whose feature matrix is explicitly mapped"
    )
    p.add_argument("--observation-key", default="features")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--output", default="outputs/mdn_posterior.npz")
    a = p.parse_args(argv)
    model, ckpt = load_mdn_checkpoint(a.checkpoint, device=a.device)
    with np.load(a.observations, allow_pickle=False) as z:
        if a.observation_key not in z:
            raise KeyError(f"{a.observation_key!r} missing; keys={z.files}")
        features = np.asarray(z[a.observation_key], dtype=np.float32)
        ids = (
            np.asarray(z["simulation_id"]).astype(str)
            if "simulation_id" in z
            else np.asarray([f"unlabelled-row-{i}" for i in range(len(features))])
        )
    samples = draw_physical_posterior(
        model,
        features,
        n_samples=a.n_samples,
        seed=a.seed,
        device=a.device,
        feature_state=ckpt["feature_scaler"],
        target_state=ckpt["target_scaler"],
        bounds_state=ckpt["bounded_transform"],
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        samples=samples,
        simulation_id=ids,
        parameter_names=np.asarray(ckpt["parameter_names"], dtype=str),
        prior_low=np.asarray(ckpt["bounded_transform"]["low"]),
        prior_high=np.asarray(ckpt["bounded_transform"]["high"]),
    )
    print(
        json.dumps(
            {
                "output": str(out.resolve()),
                "posterior_shape": list(samples.shape),
                "parameter_names": ckpt["parameter_names"],
                "dataset_sha256_used_for_training": ckpt.get("dataset_sha256"),
                "scientific_approval": False,
                "warning": "Posterior samples are only as valid as the model, simulation design, support, and calibration tests.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
