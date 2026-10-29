"""CLI for paired comparison of saved posterior sample archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.benchmarking.posterior_archive_compare import compare_posterior_archives


def _csv(s):
    try:
        return [float(v.strip()) for v in s.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nufield-compare-posteriors", description=__doc__)
    p.add_argument("archive_a")
    p.add_argument("archive_b")
    p.add_argument("--parameter-names", required=True)
    p.add_argument("--prior-low", required=True, type=_csv)
    p.add_argument("--prior-high", required=True, type=_csv)
    p.add_argument("--bootstrap", type=int, default=3000)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--output", default="outputs/posterior_comparison.json")
    a = p.parse_args(argv)
    names = [v.strip() for v in a.parameter_names.split(",") if v.strip()]
    report = compare_posterior_archives(
        a.archive_a,
        a.archive_b,
        parameter_names=names,
        prior_low=a.prior_low,
        prior_high=a.prior_high,
        n_bootstrap=a.bootstrap,
        confidence=a.confidence,
        seed=a.seed,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out.resolve()),
                "n_test_rows": report["n_test_rows"],
                "n_parent_groups": report["n_parent_groups"],
                "crps_difference": report["normalized_crps_group_bootstrap_b_minus_a"][
                    "difference_mean_b_minus_a"
                ],
                "scientific_approval": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
