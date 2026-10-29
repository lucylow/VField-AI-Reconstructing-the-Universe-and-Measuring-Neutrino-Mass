"""Command-line entry point for grouped-split NPE training."""

from __future__ import annotations

import argparse
import json

from nufield.models.sbi_npe import NPEConfig
from nufield.training.npe_workflow import train_npe_from_npz


def _csv(value: str, cast=float):
    try:
        return [cast(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cannot parse comma-separated values: {value}") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-train-npe",
        description="Train an NPE model on explicitly mapped simulation arrays",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/npe")
    p.add_argument("--field-key", required=True)
    p.add_argument("--spectrum-key", required=True)
    p.add_argument("--theta-key", default="theta")
    p.add_argument("--simulation-id-key", default="simulation_id")
    p.add_argument("--group-id-key")
    p.add_argument("--lensing-key")
    p.add_argument("--lensing-shape", default="16,16")
    p.add_argument(
        "--parameter-names", required=True, help="comma-separated in exact theta column order"
    )
    p.add_argument("--prior-low", required=True, help="comma-separated physical prior lower bounds")
    p.add_argument(
        "--prior-high", required=True, help="comma-separated physical prior upper bounds"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-fraction", type=float, default=0.75)
    p.add_argument(
        "--density-estimator", choices=["nsf", "maf", "maf_rqs", "mdn", "made"], default="nsf"
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-features", type=int, default=96)
    p.add_argument("--num-transforms", type=int, default=5)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--stop-after-epochs", type=int, default=20)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--architecture",
        choices=["hybrid", "spectrum_mlp", "cnn3d", "fno3d"],
        default="hybrid",
        help="embedding architecture to compare in a predeclared model matrix",
    )
    p.add_argument(
        "--periodic-box",
        action="store_true",
        help="required for fno3d; declare a uniformly gridded periodic simulation box",
    )
    p.add_argument(
        "--fno-modes", default="6,6,6", help="three retained Fourier mode counts, only for fno3d"
    )
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--posterior-samples", type=int, default=1000)
    p.add_argument(
        "--allow-toy",
        action="store_true",
        help="only for code smoke tests; does not authorize scientific claims",
    )
    p.add_argument(
        "--evaluate-test-once",
        action="store_true",
        help="evaluate the locked test groups only after the model/config is frozen; never use these metrics for model selection",
    )
    p.add_argument(
        "--allow-synthetic-ids",
        action="store_true",
        help="only when the data source genuinely has no per-row IDs",
    )
    args = p.parse_args(argv)
    params = [x.strip() for x in args.parameter_names.split(",") if x.strip()]
    lo = _csv(args.prior_low)
    hi = _csv(args.prior_high)
    shape = tuple(_csv(args.lensing_shape, int))
    if len(shape) != 2:
        p.error("--lensing-shape must have exactly two integers")
    modes = tuple(_csv(args.fno_modes, int))
    if len(modes) != 3:
        p.error("--fno-modes must be three comma-separated positive integers")
    if min(modes) < 1:
        p.error("--fno-modes values must be positive")
    if args.architecture == "fno3d" and not args.periodic_box:
        p.error("--architecture fno3d requires --periodic-box")
    if not (len(params) == len(lo) == len(hi)):
        p.error("parameter names and prior-low/high lengths must match")
    cfg = NPEConfig(
        density_estimator=args.density_estimator,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        max_num_epochs=args.epochs,
        training_batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        stop_after_epochs=args.stop_after_epochs,
        seed=args.seed,
        device=args.device,
    )
    result = train_npe_from_npz(
        args.dataset,
        args.output_dir,
        field_key=args.field_key,
        spectrum_key=args.spectrum_key,
        theta_key=args.theta_key,
        parameter_names=params,
        prior_low=lo,
        prior_high=hi,
        simulation_id_key=args.simulation_id_key,
        group_id_key=args.group_id_key,
        lensing_key=args.lensing_key,
        lensing_shape=shape,
        allow_synthetic_ids=args.allow_synthetic_ids,
        allow_toy=args.allow_toy,
        seed=args.seed,
        train_fraction=args.train_fraction,
        config=cfg,
        n_eval=args.n_eval,
        posterior_samples=args.posterior_samples,
        architecture=args.architecture,
        periodic_box=args.periodic_box,
        fno_modes=modes,
        evaluate_test_once=args.evaluate_test_once,
    )
    print(
        json.dumps(
            {
                "report": f"{args.output_dir}/npe_report.json",
                "status": result["scientific_status"],
                "n_validation_evaluated": result["n_validation_evaluated"],
                "n_test_evaluated": result["n_test_evaluated"],
                "final_test_evaluated": result["final_test_evaluated"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
