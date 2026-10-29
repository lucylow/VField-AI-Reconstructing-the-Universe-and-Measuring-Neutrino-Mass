"""Export model architecture cards as JSON for a preregistration or code review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nufield.models.model_cards import model_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nufield-model-cards",
        description="Export architecture assumptions and acceptance tests",
    )
    parser.add_argument("--output", help="optional output JSON path")
    args = parser.parse_args(argv)
    payload = {
        "schema_version": "1.0",
        "models": model_registry(),
        "warning": "Cards describe assumptions, not validated scientific performance or a model ranking.",
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
