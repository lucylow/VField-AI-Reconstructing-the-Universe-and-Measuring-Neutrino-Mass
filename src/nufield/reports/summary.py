"""Create concise markdown summaries from machine-readable benchmark output."""

from __future__ import annotations

import json
from pathlib import Path


def benchmark_to_markdown(report_path: str | Path, output_path: str | Path | None = None) -> str:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    lines = [
        "# νField benchmark summary",
        "",
        "> **TOY PHYSICS ONLY:** no physical cosmological constraints are implied.",
        "",
        f"- Test simulations: {report.get('n_test', 'unknown')}",
        f"- Training simulations: {report.get('n_train', 'unknown')}",
        f"- Split disjoint: {report.get('split_disjoint', 'unknown')}",
        f"- Method: {report.get('method', {}).get('algorithm', 'unknown')}",
        "",
        "## Point estimates",
        "",
        "| Parameter | Mean bias | Median absolute error |",
        "|---|---:|---:|",
    ]
    bias = report.get("mean_bias", {})
    mae = report.get("median_absolute_error", {})
    for name in report.get("parameter_names", []):
        lines.append(f"| `{name}` | {bias.get(name, 'n/a')} | {mae.get(name, 'n/a')} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        report.get("scientific_interpretation", "No interpretation recorded."),
        "",
        "## Notes",
        "",
        "This summary is generated from the JSON report. Inspect the configuration, data manifest,",
        "and all calibration diagnostics before drawing any inference conclusion.",
        "",
    ]
    result = "\n".join(lines)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(result, encoding="utf-8")
    return result
