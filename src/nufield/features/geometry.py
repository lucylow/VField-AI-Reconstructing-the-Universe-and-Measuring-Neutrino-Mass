"""Grid/box metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PeriodicGrid:
    n: int
    box_size: float
    unit: str = "Mpc/h"

    def __post_init__(self):
        if self.n < 2 or self.box_size <= 0:
            raise ValueError("n must be >=2 and box_size >0")

    @property
    def cell_size(self) -> float:
        return self.box_size / self.n

    @property
    def fundamental_k(self) -> float:
        return 2 * np.pi / self.box_size

    @property
    def nyquist_k(self) -> float:
        return np.pi / self.cell_size

    @property
    def voxel_volume(self) -> float:
        return self.cell_size**3

    def coordinates(self, centers: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offset = 0.5 if centers else 0.0
        axis = (np.arange(self.n) + offset) * self.cell_size
        return np.meshgrid(axis, axis, axis, indexing="ij")


def validate_cubic_grid(field: np.ndarray) -> int:
    arr = np.asarray(field)
    if arr.ndim != 3 or len(set(arr.shape)) != 1:
        raise ValueError("expected cubic 3-D grid")
    return int(arr.shape[0])
