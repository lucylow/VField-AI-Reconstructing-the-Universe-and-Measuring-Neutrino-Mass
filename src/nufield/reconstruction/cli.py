"""Command-line interface for reproducible reconstruction benchmarks."""
from __future__ import annotations

import argparse
import json

from nufield.reconstruction.benchmark import run_reconstruction_benchmark
from nufield.reconstruction.config import load_reconstruction_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to reconstruction YAML config")
    args = parser.parse_args(argv)
    report = run_reconstruction_benchmark(load_reconstruction_config(args.config))
    print(json.dumps({"report": report["artifacts"]["report_json"], "models": sorted(report["results"]), "group_disjoint": report["group_disjoint"], "scientific_approval": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
