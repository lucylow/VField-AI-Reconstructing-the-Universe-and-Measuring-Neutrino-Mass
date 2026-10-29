"""Transparent compute-budget bookkeeping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    task: str
    wall_seconds: float
    cpu_count: int
    gpu_count: int
    peak_memory_mb: float
    data_read_mb: float
    data_written_mb: float

    def __post_init__(self):
        if (
            not self.task.strip()
            or min(self.wall_seconds, self.peak_memory_mb, self.data_read_mb, self.data_written_mb)
            < 0
        ):
            raise ValueError("invalid resource measurement")
        if self.cpu_count < 0 or self.gpu_count < 0:
            raise ValueError("counts cannot be negative")


def aggregate_measurements(rows: list[ResourceMeasurement]) -> dict:
    if not rows:
        raise ValueError("rows cannot be empty")
    return {
        "n_tasks": len(rows),
        "wall_seconds_sum": sum(r.wall_seconds for r in rows),
        "max_peak_memory_mb": max(r.peak_memory_mb for r in rows),
        "cpu_core_hours_sum": sum(r.wall_seconds * r.cpu_count for r in rows) / 3600,
        "gpu_hours_sum": sum(r.wall_seconds * r.gpu_count for r in rows) / 3600,
        "data_read_mb_sum": sum(r.data_read_mb for r in rows),
        "data_written_mb_sum": sum(r.data_written_mb for r in rows),
        "measurements": [asdict(r) for r in rows],
    }


def write_budget(path: str | Path, summary: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
