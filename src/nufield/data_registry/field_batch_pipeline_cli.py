"""CLI for streaming a mapped HDF5 field dataset to model-ready arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.data_registry.field_batch_pipeline import process_hdf5_field_batch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-field-pipeline", description=__doc__)
    p.add_argument("input_file")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fields-key", required=True)
    p.add_argument("--theta-key", required=True)
    p.add_argument("--parameter-names", required=True)
    p.add_argument("--simulation-id-key", required=True)
    p.add_argument("--group-id-key", required=True)
    p.add_argument("--box-size-mpc-h", type=float, required=True)
    p.add_argument("--redshift", type=float, required=True)
    p.add_argument(
        "--field-kind",
        choices=["density", "overdensity", "matter_density", "galaxy_density_mock"],
        required=True,
    )
    p.add_argument("--power-bins", type=int, default=16)
    p.add_argument("--summary-channel", type=int, default=0)
    p.add_argument("--max-memory-gb", type=float, default=8.0)
    p.add_argument("--no-copy-fields", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    names = [x.strip() for x in a.parameter_names.split(",") if x.strip()]
    result = process_hdf5_field_batch(
        a.input_file,
        a.output_dir,
        fields_key=a.fields_key,
        theta_key=a.theta_key,
        parameter_names=names,
        simulation_id_key=a.simulation_id_key,
        group_id_key=a.group_id_key,
        box_size_mpc_h=a.box_size_mpc_h,
        redshift=a.redshift,
        field_kind=a.field_kind,
        power_bins=a.power_bins,
        summary_channel=a.summary_channel,
        max_memory_gb=a.max_memory_gb,
        copy_fields=not a.no_copy_fields,
        overwrite=a.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(a.output_dir).resolve()),
                "n_rows": result["n_rows"],
                "feature_width": result["feature_schema"]["width"],
                "copied_fields": result["copied_fields"],
                "scientific_approval": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
