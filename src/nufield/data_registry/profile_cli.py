"""CLI for explicitly profiling a local NPZ's schema without guessing its scientific meaning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.data_registry.npz_profile import profile_npz


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-profile", description="Profile local NPZ array keys/shapes/dtypes/checksum"
    )
    p.add_argument("dataset")
    p.add_argument("--inspect-values", action="store_true")
    p.add_argument("--max-summary-values", type=int, default=2_000_000)
    p.add_argument("--output", default="outputs/npz_profile.json")
    a = p.parse_args(argv)
    result = profile_npz(
        a.dataset, inspect_values=a.inspect_values, max_summary_values=a.max_summary_values
    )
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "archive_sha256": result["archive_sha256"],
                "arrays": [
                    {"key": r["key"], "shape": r["shape"], "dtype": r["dtype"]}
                    for r in result["arrays"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
