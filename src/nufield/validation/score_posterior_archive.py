"""Score a saved posterior-sample archive and record the partition used for evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nufield.validation.posterior_diagnostics import posterior_diagnostic_report


def score_archive(
    path: str | Path,
    *,
    parameter_names: list[str],
    samples_key: str = "samples",
    truth_key: str = "truth",
    group_key: str = "group_id",
    partition: str = "validation",
) -> dict:
    archive = Path(path)
    if not archive.exists():
        raise FileNotFoundError(archive)
    if partition not in {"train", "validation", "test", "sbc"}:
        raise ValueError("partition must be train/validation/test/sbc")
    with np.load(archive, allow_pickle=False) as z:
        for key in (samples_key, truth_key):
            if key not in z:
                raise KeyError(f"missing {key!r}; keys={z.files}")
        samples = np.asarray(z[samples_key])
        truth = np.asarray(z[truth_key])
        groups = np.asarray(z[group_key]).astype(str) if group_key in z else None
    report = posterior_diagnostic_report(samples, truth, parameter_names)
    group_hash = None
    if groups is not None:
        if len(groups) != len(samples):
            raise ValueError("group_id array does not align with posterior samples")
        group_hash = hashlib.sha256("\n".join(sorted(groups.tolist())).encode()).hexdigest()
    report.update(
        {
            "archive_path": str(archive.resolve()),
            "partition_declared_by_caller": partition,
            "group_hash": group_hash,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "warning": "The caller-supplied partition label is not independently verified. Never use test metrics for model selection.",
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="nufield-score-posterior",
        description="Score posterior samples from a held-out evaluation archive",
    )
    p.add_argument("archive")
    p.add_argument("--parameter-names", required=True)
    p.add_argument("--samples-key", default="samples")
    p.add_argument("--truth-key", default="truth")
    p.add_argument("--group-key", default="group_id")
    p.add_argument("--partition", choices=["train", "validation", "test", "sbc"], required=True)
    p.add_argument("--output", default="outputs/posterior_score.json")
    a = p.parse_args(argv)
    r = score_archive(
        a.archive,
        parameter_names=[x.strip() for x in a.parameter_names.split(",") if x.strip()],
        samples_key=a.samples_key,
        truth_key=a.truth_key,
        group_key=a.group_key,
        partition=a.partition,
    )
    if a.partition == "test":
        r["warning"] += (
            " TEST PARTITION: reserve for final frozen evaluation, not model/HPO selection."
        )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "partition": a.partition,
                "mean_crps_by_parameter": r["mean_crps_by_parameter"],
                "group_hash": r["group_hash"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
