"""CLI for mapping a user-verified HDF5 particle product to periodic fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.data_registry.particle_pipeline import process_hdf5_particle_batch


def _names(value: str) -> list[str]:
    names = [x.strip() for x in value.split(",") if x.strip()]
    if not names or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("provide unique comma-separated parameter names")
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-particle-pipeline", description=__doc__)
    p.add_argument("input_file")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--positions-key", required=True, help="exact HDF5 path to [N,Nparticle,3] or [Nparticle,3]"
    )
    p.add_argument("--theta-key", required=True, help="exact HDF5 path to parameter vector/matrix")
    p.add_argument(
        "--parameter-names", required=True, type=_names, help="names in exact theta column order"
    )
    p.add_argument("--simulation-id-key", required=True)
    p.add_argument(
        "--group-id-key", required=True, help="parent simulation/realization family identifier"
    )
    p.add_argument("--box-size-mpc-h", required=True, type=float)
    p.add_argument("--grid-size", type=int, default=32)
    p.add_argument("--redshift", type=float, required=True)
    p.add_argument("--power-bins", type=int, default=16)
    p.add_argument("--max-memory-gb", type=float, default=8.0)
    p.add_argument("--position-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    manifest = process_hdf5_particle_batch(
        a.input_file,
        a.output_dir,
        positions_key=a.positions_key,
        theta_key=a.theta_key,
        parameter_names=a.parameter_names,
        simulation_id_key=a.simulation_id_key,
        group_id_key=a.group_id_key,
        box_size_mpc_h=a.box_size_mpc_h,
        grid_size=a.grid_size,
        redshift=a.redshift,
        power_bins=a.power_bins,
        max_memory_gb=a.max_memory_gb,
        position_dtype=a.position_dtype,
        overwrite=a.overwrite,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(a.output_dir).resolve()),
                "n_realizations": manifest["n_realizations"],
                "field_shape": manifest["field_shape"],
                "parameter_names": manifest["parameter_names"],
                "source_sha256": manifest["source_sha256"],
                "scientific_approval": False,
                "warning": "Prototype periodic-box processing only. Validate against a trusted mass-assignment/power-spectrum implementation.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
