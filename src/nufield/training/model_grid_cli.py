"""CLI that emits a preregistered data/model/estimator/seed matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.training.model_grid import build_model_matrix


def _items(value: str, cast=str):
    try:
        return [cast(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cannot parse value {value!r}") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-model-grid", description="Generate predeclared model comparison jobs"
    )
    p.add_argument("--dataset-id", required=True, help="immutable DatasetCard dataset_id")
    p.add_argument("--split-id", required=True, help="frozen split manifest ID/hash")
    p.add_argument("--seeds", default="11,22,33", help="comma-separated model training seeds")
    p.add_argument("--models", default="pk_classical,mlp_npe,cnn3d_npe")
    p.add_argument(
        "--views", default="pk,field", help="pk,field,hybrid; incompatible pairs are skipped"
    )
    p.add_argument("--estimators", default="maf,nsf")
    p.add_argument("--budget", choices=["laptop", "workstation", "hpc"], default="workstation")
    p.add_argument("--output", default="configs/generated_model_matrix.json")
    args = p.parse_args(argv)
    configs = build_model_matrix(
        dataset_id=args.dataset_id,
        seeds=_items(args.seeds, int),
        split_id=args.split_id,
        model_ids=tuple(_items(args.models)),
        observation_views=tuple(_items(args.views)),
        estimators=tuple(_items(args.estimators)),
        budget=args.budget,
    )
    output = {
        "schema_version": "1.0",
        "dataset_id": args.dataset_id,
        "split_id": args.split_id,
        "n_experiments": len(configs),
        "predeclared_only": True,
        "experiments": [c.to_dict() for c in configs],
        "warning": "Generated matrix is a plan. Verify each model's physical assumptions and keep one final test set locked.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(path), "n_experiments": len(configs), "split_id": args.split_id},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
