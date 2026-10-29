"""CLI for metadata-first HDF5 schema profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.data_registry.hdf5_profile import profile_hdf5


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-profile-hdf5", description=__doc__)
    p.add_argument("input_file")
    p.add_argument("--max-objects", type=int, default=5000)
    p.add_argument("--no-attributes", action="store_true")
    p.add_argument("--output", default="outputs/hdf5_profile.json")
    a = p.parse_args(argv)
    result = profile_hdf5(
        a.input_file, max_objects=a.max_objects, include_attributes=not a.no_attributes
    )
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "objects": result["object_count_returned"],
                "truncated": result["truncated"],
                "file_bytes": result["file_bytes"],
                "scientific_semantics_inferred": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
