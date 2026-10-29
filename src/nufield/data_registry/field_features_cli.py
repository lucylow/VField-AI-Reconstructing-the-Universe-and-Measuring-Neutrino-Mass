"""Extract transparent periodic-box summary features from explicitly mapped NPZ fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nufield.data_registry.field_features import feature_schema, summarize_field_batch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-field-features",
        description="Create a P(k)+field-statistics baseline feature matrix",
    )
    p.add_argument("dataset")
    p.add_argument("--field-key", required=True)
    p.add_argument("--box-size", type=float, required=True)
    p.add_argument("--power-bins", type=int, default=16)
    p.add_argument("--no-gradient", action="store_true")
    p.add_argument("--output", default="outputs/field_features.npz")
    p.add_argument("--schema-output", default="outputs/feature_schema.json")
    args = p.parse_args(argv)
    with np.load(args.dataset, allow_pickle=False) as z:
        if args.field_key not in z:
            raise KeyError(f"field key {args.field_key!r} missing; keys={z.files}")
        fields = np.asarray(z[args.field_key])
    features, names = summarize_field_batch(
        fields,
        box_size=args.box_size,
        power_bins=args.power_bins,
        include_gradient=not args.no_gradient,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features, feature_names=np.asarray(names, dtype=str))
    schema = feature_schema(
        names,
        source_key=args.field_key,
        box_size=args.box_size,
        preprocessing={
            "method": "periodic field_summary",
            "power_bins": args.power_bins,
            "gradient_features": not args.no_gradient,
            "k_units": "inverse box_size units",
            "power_normalization": "see nufield.features.periodic_power",
        },
    )
    sch = Path(args.schema_output)
    sch.parent.mkdir(parents=True, exist_ok=True)
    sch.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "n_samples": len(features),
                "n_features": features.shape[1],
                "features_output": str(output),
                "schema_output": str(sch),
                "warning": "This baseline feature extractor is for periodic simulation fields, not survey catalogs.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
