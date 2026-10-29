"""A tiny append-only experiment registry using JSON Lines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nufield.provenance import sha256_file


class ExperimentRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        experiment_id: str,
        command: str,
        config: dict[str, Any],
        status: str,
        artifact_path: str | Path | None = None,
        notes: str = "",
    ) -> dict:
        if not experiment_id or any(ch.isspace() for ch in experiment_id):
            raise ValueError("experiment_id must be non-empty and contain no whitespace")
        if status not in {"planned", "running", "complete", "failed", "superseded"}:
            raise ValueError("unsupported status")
        item = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "command": command,
            "status": status,
            "config": config,
            "notes": notes,
        }
        if artifact_path is not None and Path(artifact_path).exists():
            item["artifact"] = {"path": str(artifact_path), "sha256": sha256_file(artifact_path)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, sort_keys=True, default=str) + "\n")
        return item

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_no}") from exc
        return rows
