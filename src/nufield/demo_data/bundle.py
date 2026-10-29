"""Build, validate, and export reproducible synthetic NUFIELD demo bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .generators import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    build_dataset_metadata,
    build_manifest,
    build_provenance,
    generate_ablation_report,
    generate_benchmark_report,
    generate_cmb_summary,
    generate_data_quality_report,
    generate_expansion_history,
    generate_experiment_history,
    generate_field_summary,
    generate_import_error_examples,
    generate_lensing_catalog,
    generate_parameter_rows,
    generate_posterior_chains,
    generate_power_spectrum,
    generate_survey_catalog,
    generate_validation_report,
)
from .profiles import DEMO_SCENARIOS, PARAMETER_KEYS, SCIENCE_WARNING, DemoScenario, get_scenario
from .randoms import descriptive_stats, safe_count, stable_digest


@dataclass(frozen=True, slots=True)
class DemoOptions:
    """Validated sizes that bound run time and output size."""

    scenario: str = "baseline"
    seed: int = 90721
    sample_count: int = 500
    observation_count: int = 300
    chain_count: int = 4
    samples_per_chain: int = 250
    spectrum_points: int = 180
    history_count: int = 48
    field_slices: int = 80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_options(options: dict[str, Any] | DemoOptions | None = None) -> DemoOptions:
    """Normalize CLI/API options and enforce hard maximums for generation."""
    if isinstance(options, DemoOptions):
        raw = asdict(options)
    else:
        raw = dict(options or {})
    scenario = str(raw.get("scenario", "baseline"))
    if scenario not in DEMO_SCENARIOS:
        scenario = "baseline"
    try:
        seed = int(raw.get("seed", 90721)) & 0xFFFFFFFF
    except (TypeError, ValueError, OverflowError):
        seed = 90721
    return DemoOptions(
        scenario=scenario,
        seed=seed,
        sample_count=safe_count(
            raw.get("sample_count", raw.get("sampleCount", 500)),
            default=500,
            minimum=20,
            maximum=100_000,
        ),
        observation_count=safe_count(
            raw.get("observation_count", raw.get("observationCount", 300)),
            default=300,
            minimum=20,
            maximum=100_000,
        ),
        chain_count=safe_count(
            raw.get("chain_count", raw.get("chainCount", 4)), default=4, minimum=1, maximum=8
        ),
        samples_per_chain=safe_count(
            raw.get("samples_per_chain", raw.get("samplesPerChain", 250)),
            default=250,
            minimum=50,
            maximum=10_000,
        ),
        spectrum_points=safe_count(
            raw.get("spectrum_points", raw.get("spectrumPoints", 180)),
            default=180,
            minimum=30,
            maximum=2_000,
        ),
        history_count=safe_count(
            raw.get("history_count", raw.get("historyCount", 48)),
            default=48,
            minimum=8,
            maximum=5_000,
        ),
        field_slices=safe_count(
            raw.get("field_slices", raw.get("fieldSlices", 80)),
            default=80,
            minimum=20,
            maximum=2_000,
        ),
    )


def _artifact_seed(seed: int, artifact_name: str) -> int:
    """Derive independent, repeatable streams for every artifact family."""
    digest = stable_digest(f"{seed}:{artifact_name}")
    return int(digest[:8], 16)


def _rng(seed: int, artifact_name: str) -> random.Random:
    return random.Random(_artifact_seed(seed, artifact_name))


def build_demo_bundle(options: dict[str, Any] | DemoOptions | None = None) -> dict[str, Any]:
    """Generate every mock artifact for a scenario with stable seeded rows.

    `created_at` uses wall-clock UTC and therefore differs between calls. All
    seeded data rows, profile definitions, and numeric summaries are repeatable
    for the same normalized options.
    """
    normalized = normalize_options(options)
    scenario: DemoScenario = get_scenario(normalized.scenario)
    option_data = normalized.to_dict()
    run_id = f"mock-{scenario.scenario_id}-{normalized.seed:08x}"

    parameter_rows = generate_parameter_rows(
        _rng(normalized.seed, "parameter_rows"),
        scenario,
        sample_count=normalized.sample_count,
        seed=normalized.seed,
        run_id=run_id,
    )
    posterior_chains = generate_posterior_chains(
        _rng(normalized.seed, "posterior_chains"),
        scenario,
        chain_count=normalized.chain_count,
        samples_per_chain=normalized.samples_per_chain,
        run_id=run_id,
    )
    power_spectrum = generate_power_spectrum(
        _rng(normalized.seed, "power_spectrum"),
        scenario,
        count=normalized.spectrum_points,
    )
    expansion_history = generate_expansion_history(
        _rng(normalized.seed, "expansion_history"),
        scenario,
        count=max(50, min(160, normalized.spectrum_points)),
    )
    survey_catalog = generate_survey_catalog(
        _rng(normalized.seed, "survey_catalog"),
        scenario,
        count=normalized.observation_count,
        run_id=run_id,
    )
    cmb_count = min(500, max(40, round(normalized.spectrum_points * 1.4)))
    cmb_summary = generate_cmb_summary(
        _rng(normalized.seed, "cmb_summary"), scenario, count=cmb_count
    )
    lens_count = min(250, max(12, round(normalized.observation_count * 0.14)))
    lensing_catalog = generate_lensing_catalog(
        _rng(normalized.seed, "lensing_catalog"),
        scenario,
        count=lens_count,
        run_id=run_id,
    )
    benchmark = generate_benchmark_report(
        _rng(normalized.seed, "benchmark_report"),
        scenario,
        sample_rows=parameter_rows,
        run_id=run_id,
    )
    validation = generate_validation_report(scenario, benchmark, run_id=run_id)
    history = generate_experiment_history(
        _rng(normalized.seed, "experiment_history"),
        scenario,
        count=normalized.history_count,
        sample_count=normalized.sample_count,
        observation_count=normalized.observation_count,
        seed=normalized.seed,
        run_id=run_id,
    )
    ablation = generate_ablation_report(
        _rng(normalized.seed, "ablation_report"), scenario, sample_count=normalized.sample_count
    )
    data_quality = generate_data_quality_report(scenario, survey_catalog, seed=normalized.seed)
    field_summary = generate_field_summary(
        _rng(normalized.seed, "field_summary"), scenario, count=normalized.field_slices
    )

    grid_size = (
        128
        if scenario.scenario_id == "high_resolution"
        else 16
        if scenario.scenario_id == "sparse_data"
        else 64
    )
    dataset = build_dataset_metadata(
        scenario,
        seed=normalized.seed,
        sample_count=normalized.sample_count,
        observation_count=normalized.observation_count,
        grid_size=grid_size,
        run_id=run_id,
    )
    artifact_names = [
        "dataset.json",
        "dataset.manifest.json",
        "benchmark_report.json",
        "parameter_samples.json",
        "posterior_chains.json",
        "power_spectrum.json",
        "expansion_history.json",
        "survey_catalog.json",
        "cmb_summary.json",
        "lensing_catalog.json",
        "validation_report.json",
        "experiment_history.json",
        "ablation_report.json",
        "data_quality_report.json",
        "field_summary.json",
    ]
    if scenario.scenario_id == "malformed_artifacts":
        artifact_names.append("import_error_examples.json")
    manifest = build_manifest(
        dataset,
        seed=normalized.seed,
        scenario=scenario,
        run_id=run_id,
        options=option_data,
        artifact_names=artifact_names,
    )

    # Convenience summaries are computed from generated rows; they are not inference.
    parameter_summary = {
        key: descriptive_stats([row[key] for row in parameter_rows]) for key in PARAMETER_KEYS
    }
    run_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle_kind": "nufield-synthetic-demo-bundle",
        "generator_version": GENERATOR_VERSION,
        "generated_at": run_time,
        "run_id": run_id,
        "options": option_data,
        "scenario": scenario.to_dict(),
        "provenance": build_provenance(
            scenario, seed=normalized.seed, options=option_data, run_id=run_id
        ),
        "dataset": dataset,
        "manifest": manifest,
        "benchmark_report": benchmark,
        "parameter_samples": {
            "schema_version": SCHEMA_VERSION,
            "is_synthetic": True,
            "run_id": run_id,
            "rows": parameter_rows,
            "parameter_names": list(PARAMETER_KEYS),
            "summary_statistics": parameter_summary,
            "scientific_warning": SCIENCE_WARNING,
        },
        "posterior_chains": posterior_chains,
        "power_spectrum": power_spectrum,
        "expansion_history": expansion_history,
        "survey_catalog": survey_catalog,
        "cmb_summary": cmb_summary,
        "lensing_catalog": lensing_catalog,
        "validation_report": validation,
        "experiment_history": history,
        "ablation_report": ablation,
        "data_quality_report": data_quality,
        "field_summary": field_summary,
    }
    if scenario.scenario_id == "malformed_artifacts":
        bundle["import_error_examples"] = generate_import_error_examples()
    bundle["summary"] = summarize_bundle(bundle)
    return bundle


def summarize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Summarize the size and generated states for CLI output and manifests."""
    parameter_rows = bundle.get("parameter_samples", {}).get("rows", [])
    survey_rows = bundle.get("survey_catalog", {}).get("rows", [])
    spectrum_points = bundle.get("power_spectrum", {}).get("points", [])
    chains = bundle.get("posterior_chains", {}).get("chains", [])
    history_rows = bundle.get("experiment_history", {}).get("runs", [])
    return {
        "scenario": bundle.get("scenario", {}).get("label", "Unknown scenario"),
        "seed": bundle.get("options", {}).get("seed"),
        "parameter_rows": len(parameter_rows),
        "survey_rows": len(survey_rows),
        "posterior_trace_rows": sum(len(chain.get("samples", [])) for chain in chains),
        "power_spectrum_points": len(spectrum_points),
        "history_rows": len(history_rows),
        "artifact_count": len(list_artifacts(bundle)),
        "validation_status": bundle.get("validation_report", {}).get("overall_status", "UNKNOWN"),
        "all_synthetic": bundle.get("dataset", {}).get("is_synthetic") is True,
        "scientific_warning": SCIENCE_WARNING,
    }


def list_artifacts(bundle: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return canonical named artifact objects in stable export order."""
    pairs: list[tuple[str, Any]] = [
        ("dataset.json", bundle["dataset"]),
        ("dataset.manifest.json", bundle["manifest"]),
        ("benchmark_report.json", bundle["benchmark_report"]),
        ("parameter_samples.json", bundle["parameter_samples"]),
        ("posterior_chains.json", bundle["posterior_chains"]),
        ("power_spectrum.json", bundle["power_spectrum"]),
        ("expansion_history.json", bundle["expansion_history"]),
        ("survey_catalog.json", bundle["survey_catalog"]),
        ("cmb_summary.json", bundle["cmb_summary"]),
        ("lensing_catalog.json", bundle["lensing_catalog"]),
        ("validation_report.json", bundle["validation_report"]),
        ("experiment_history.json", bundle["experiment_history"]),
        ("ablation_report.json", bundle["ablation_report"]),
        ("data_quality_report.json", bundle["data_quality_report"]),
        ("field_summary.json", bundle["field_summary"]),
    ]
    if "import_error_examples" in bundle:
        pairs.append(("import_error_examples.json", bundle["import_error_examples"]))
    return pairs


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flat CSV table with stable column ordering and UTF-8 encoding."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_demo_bundle(
    output_dir: str | Path,
    options: dict[str, Any] | DemoOptions | None = None,
    *,
    include_csv: bool = True,
    include_zip: bool = True,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write JSON artifacts, optional CSVs, bundle ZIP, and integrity manifest.

    Output is written to a temporary sibling directory and atomically swapped
    into place when possible. This prevents a failed generation from leaving
    half-written JSON files in the target directory.
    """
    target = Path(output_dir).expanduser().resolve()
    normalized = normalize_options(options)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {target}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        bundle = build_demo_bundle(normalized)
        for filename, artifact in list_artifacts(bundle):
            (temporary / filename).write_text(_json_text(artifact), encoding="utf-8")
        (temporary / "bundle.json").write_text(_json_text(bundle), encoding="utf-8")

        if include_csv:
            _write_csv(temporary / "parameter_samples.csv", bundle["parameter_samples"]["rows"])
            _write_csv(temporary / "survey_catalog.csv", bundle["survey_catalog"]["rows"])
            _write_csv(temporary / "power_spectrum.csv", bundle["power_spectrum"]["points"])
            _write_csv(temporary / "expansion_history.csv", bundle["expansion_history"]["points"])
            _write_csv(temporary / "cmb_summary.csv", bundle["cmb_summary"]["points"])
            _write_csv(temporary / "lensing_catalog.csv", bundle["lensing_catalog"]["rows"])
            for chain in bundle["posterior_chains"]["chains"]:
                _write_csv(temporary / f"posterior_{chain['chain_id']}.csv", chain["samples"])

        readme = _build_readme(bundle, include_csv=include_csv)
        (temporary / "README_DEMO_DATA.md").write_text(readme, encoding="utf-8")
        file_hashes = {
            path.name: _sha256(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        integrity = {
            "manifest_version": "1.0",
            "is_synthetic": True,
            "algorithm": "SHA-256",
            "files": file_hashes,
            "warning": "Hashes establish file integrity only; they do not establish scientific validity.",
        }
        (temporary / "checksums.json").write_text(_json_text(integrity), encoding="utf-8")

        if include_zip:
            with zipfile.ZipFile(
                temporary / "nufield-demo-artifacts.zip",
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=8,
            ) as archive:
                for path in sorted(temporary.iterdir()):
                    if path.is_file() and path.name != "nufield-demo-artifacts.zip":
                        archive.write(path, arcname=path.name)

        # Replace the old folder only after all generated files were verified.
        if target.exists():
            backup = target.with_name(f"{target.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            target.rename(backup)
            try:
                temporary.rename(target)
            except Exception:
                if not target.exists() and backup.exists():
                    backup.rename(target)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            temporary.rename(target)

        return {
            "output_dir": str(target),
            "bundle_path": str(target / "bundle.json"),
            "zip_path": str(target / "nufield-demo-artifacts.zip") if include_zip else None,
            "artifact_count": len(list_artifacts(bundle)),
            "files_written": len([path for path in target.iterdir() if path.is_file()]),
            "summary": bundle["summary"],
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _build_readme(bundle: dict[str, Any], *, include_csv: bool) -> str:
    """Create an explanatory README alongside fixtures for demo hand-offs."""
    summary = bundle["summary"]
    lines = [
        "# νField Synthetic Demo Data",
        "",
        "> **Synthetic data warning.** These files are fabricated for frontend demos, UI testing, and software validation. They are not observational data, calibrated cosmology predictions, or scientific inference results.",
        "",
        "## Generated profile",
        "",
        f"- Scenario: `{bundle['scenario']['scenario_id']}` — {bundle['scenario']['label']}",
        f"- Seed: `{bundle['options']['seed']}`",
        f"- Generator: `{bundle['generator_version']}`",
        f"- Run ID: `{bundle['run_id']}`",
        f"- Parameter rows: {summary['parameter_rows']:,}",
        f"- Catalog rows: {summary['survey_rows']:,}",
        f"- Pseudo-chain rows: {summary['posterior_trace_rows']:,}",
        f"- Experiment history rows: {summary['history_rows']:,}",
        f"- Validation state: `{summary['validation_status']}` (fixture checks only)",
        "",
        "## Files",
        "",
        "- `bundle.json`: all generated artifacts in one object.",
        "- `dataset.json` / `dataset.manifest.json`: importer-compatible metadata and provenance.",
        "- `benchmark_report.json`: synthetic metric values, coverage placeholders, and split IDs.",
        "- `parameter_samples.json`: generated sample rows with train/validation/test labels.",
        "- `posterior_chains.json`: pseudo-traces for line charts; not real MCMC samples.",
        "- `power_spectrum.json`, `cmb_summary.json`, `expansion_history.json`: chart-ready illustrative series.",
        "- `survey_catalog.json`, `lensing_catalog.json`: generated object tables.",
        "- `validation_report.json`, `data_quality_report.json`: controlled UI test states.",
        "- `experiment_history.json`, `ablation_report.json`, `field_summary.json`: timeline, comparison, and field-slice fixtures.",
        "- `checksums.json`: SHA-256 checksums for generated files (integrity, not validity).",
    ]
    if include_csv:
        lines.extend(
            [
                "- CSV files: flat exports of parameter rows, catalogs, spectra, and each pseudo-chain."
            ]
        )
    lines.extend(
        [
            "",
            "## Scientific boundaries",
            "",
            "1. No real survey observations are included.",
            "2. The generator does not call CLASS, CAMB, an N-body solver, or the project inference pipeline.",
            "3. Names like `sigma8`, `omega_m`, `sum_mnu_ev`, and `coverage_diagnostic` are schema/display labels only.",
            "4. A synthetic `PASS` checks fixture consistency only and says nothing about physical validity.",
            "5. The generator is deterministic for a fixed seed and options at the row level; output timestamps differ between runs.",
            "",
            "## Rebuild",
            "",
            "```bash",
            "python scripts/generate_mock_data.py --profile baseline --seed 90721 --output examples/mock_data/baseline",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
