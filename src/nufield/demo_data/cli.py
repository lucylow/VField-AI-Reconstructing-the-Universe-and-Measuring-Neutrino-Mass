"""Command-line entry point for building synthetic demo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from .bundle import write_demo_bundle
from .profiles import DEMO_SCENARIOS, SCIENCE_WARNING, scenario_catalog


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI options with bounded defaults and explicit warnings."""
    parser = argparse.ArgumentParser(
        prog="nufield-demo-data",
        description="Create synthetic NUFIELD fixtures for frontend demonstrations and tests.",
        epilog="WARNING: generated files are fabricated demo data, not cosmological results.",
    )
    parser.add_argument(
        "--profile",
        default="baseline",
        help="Scenario ID or 'all'. Use --list-profiles to see options.",
    )
    parser.add_argument(
        "--seed", type=int, default=90721, help="Non-negative seed (normalized to 32 bits)."
    )
    parser.add_argument(
        "--sample-count", type=int, default=500, help="Parameter sample rows (20–100000)."
    )
    parser.add_argument(
        "--observation-count", type=int, default=300, help="Survey catalog rows (20–100000)."
    )
    parser.add_argument("--chain-count", type=int, default=4, help="Pseudo-chain count (1–8).")
    parser.add_argument(
        "--samples-per-chain", type=int, default=250, help="Rows per pseudo-chain (50–10000)."
    )
    parser.add_argument(
        "--spectrum-points", type=int, default=180, help="Power spectrum chart points (30–2000)."
    )
    parser.add_argument(
        "--history-count", type=int, default=48, help="Fake experiment history rows (8–5000)."
    )
    parser.add_argument(
        "--field-slices", type=int, default=80, help="Field summary slices (20–2000)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/mock_data/baseline"),
        help="Output directory for one profile or root for --profile all.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip flat CSV exports.")
    parser.add_argument("--no-zip", action="store_true", help="Skip per-profile ZIP archive.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Refuse to replace an existing output directory.",
    )
    parser.add_argument("--list-profiles", action="store_true", help="Print scenario IDs and exit.")
    parser.add_argument(
        "--json-summary", action="store_true", help="Print machine-readable summary JSON."
    )
    return parser


def _options_from_args(args: argparse.Namespace, scenario: str | None = None) -> dict[str, Any]:
    return {
        "scenario": scenario or args.profile,
        "seed": args.seed,
        "sample_count": args.sample_count,
        "observation_count": args.observation_count,
        "chain_count": args.chain_count,
        "samples_per_chain": args.samples_per_chain,
        "spectrum_points": args.spectrum_points,
        "history_count": args.history_count,
        "field_slices": args.field_slices,
    }


def _write_library_index(root: Path, results: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": "1.0.0",
        "library_kind": "nufield-synthetic-demo-scenario-library",
        "scenario_count": len(results),
        "scenarios": results,
        "scientific_warning": SCIENCE_WARNING,
        "interpretation": "This index only catalogs UI fixtures. It is not a scientific benchmark.",
    }
    (root / "scenario_library_index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# NUFIELD Synthetic Demo Scenario Library",
        "",
        "> Synthetic mock data for UI demos, product walkthroughs, and test fixtures. Not observations or scientific results.",
        "",
        f"Generated scenario count: **{len(results)}**",
        "",
        "| Scenario | Validation fixture state | Parameter rows | Catalog rows | Folder |",
        "|---|---|---:|---:|---|",
    ]
    for result in results:
        summary = result["summary"]
        relative = Path(result["output_dir"]).name
        lines.append(
            f"| {summary['scenario']} | `{summary['validation_status']}` | "
            f"{summary['parameter_rows']:,} | {summary['survey_rows']:,} | `{relative}/` |"
        )
    lines.extend(
        [
            "",
            "## What the scenarios exercise",
            "",
            "- Baseline rendering and reproducible seed behavior.",
            "- Lower and higher synthetic neutrino-mass input labels for comparison views.",
            "- Increased observation noise and calibration-offset states.",
            "- Held-out domain shift, coverage underflow, and deliberate split overlap.",
            "- Low-count and high-volume table states.",
            "- Importer edge-case descriptions for error-state walkthroughs.",
            "",
            "Every scenario folder has per-artifact JSON files, a single `bundle.json`, optional CSVs, checksums, and a ZIP archive.",
            "A synthetic `PASS` only checks expected fixture properties. It does not validate physics, model calibration, or data provenance.",
            "",
        ]
    )
    (root / "README_SCENARIO_LIBRARY.md").write_text("\n".join(lines), encoding="utf-8")


def _zip_library(root: Path, archive_path: Path) -> None:
    """Create a single download containing every scenario folder."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8
    ) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() == archive_path.resolve():
                continue
            archive.write(path, arcname=path.relative_to(root.parent))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_profiles:
        if args.json_summary:
            print(json.dumps(scenario_catalog(), indent=2, ensure_ascii=False))
        else:
            for key, scenario in DEMO_SCENARIOS.items():
                print(f"{key:22} {scenario.label:26} [{scenario.group}] — {scenario.description}")
        return 0

    if args.profile != "all" and args.profile not in DEMO_SCENARIOS:
        parser.error(
            f"Unknown profile '{args.profile}'. Use --list-profiles to view valid profile IDs."
        )
    if args.seed < 0:
        print("Note: negative seeds are normalized to unsigned 32-bit values.", file=sys.stderr)

    if args.profile != "all":
        result = write_demo_bundle(
            args.output,
            _options_from_args(args),
            include_csv=not args.no_csv,
            include_zip=not args.no_zip,
            overwrite=not args.no_overwrite,
        )
        if args.json_summary:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Wrote {result['files_written']} files to: {result['output_dir']}")
            print(f"Bundle: {result['bundle_path']}")
            if result["zip_path"]:
                print(f"ZIP:    {result['zip_path']}")
            print(
                f"Rows: {result['summary']['parameter_rows']:,} parameters, {result['summary']['survey_rows']:,} catalog objects"
            )
            print(f"Scenario validation fixture: {result['summary']['validation_status']}")
            print("WARNING: synthetic UI fixtures only; do not interpret as cosmology results.")
        return 0

    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    # Smaller defaults for the all-profile library keep download size reasonable.
    library_sample_count = min(args.sample_count, 250)
    library_observation_count = min(args.observation_count, 150)
    library_chain_count = min(args.chain_count, 2)
    library_samples_per_chain = min(args.samples_per_chain, 100)
    library_spectrum_points = min(args.spectrum_points, 90)
    library_history_count = min(args.history_count, 16)
    for offset, scenario in enumerate(DEMO_SCENARIOS):
        options = {
            "scenario": scenario,
            "seed": (args.seed + offset * 101) & 0xFFFFFFFF,
            "sample_count": library_sample_count,
            "observation_count": library_observation_count,
            "chain_count": library_chain_count,
            "samples_per_chain": library_samples_per_chain,
            "spectrum_points": library_spectrum_points,
            "history_count": library_history_count,
            "field_slices": min(args.field_slices, 60),
        }
        result = write_demo_bundle(
            root / scenario,
            options,
            include_csv=not args.no_csv,
            include_zip=not args.no_zip,
            overwrite=not args.no_overwrite,
        )
        results.append(result)
        print(
            f"[{len(results):02d}/{len(DEMO_SCENARIOS)}] {scenario}: {result['summary']['parameter_rows']} parameter rows, {result['files_written']} files"
        )

    # Convert internal absolute paths to paths relative to the library folder.
    for result in results:
        result["output_dir"] = str(Path(result["output_dir"]).relative_to(root))
        result["bundle_path"] = str(Path(result["bundle_path"]).relative_to(root))
        if result["zip_path"]:
            result["zip_path"] = str(Path(result["zip_path"]).relative_to(root))
    _write_library_index(root, results)
    master_archive = root.parent / f"{root.name}-scenario-library.zip"
    _zip_library(root, master_archive)
    if args.json_summary:
        print(
            json.dumps(
                {
                    "library_root": str(root),
                    "master_zip": str(master_archive),
                    "scenarios": results,
                },
                indent=2,
            )
        )
    else:
        print(f"Scenario library: {root}")
        print(f"Master download:  {master_archive}")
        print("WARNING: every generated file is synthetic and is not a scientific result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
