"""Periodic-grid excursion-set morphology diagnostics for cosmological density fields.

This module extracts simple topology and geometry from thresholded density grids. It is intended as a
supplement to power-spectrum inference: topology is sensitive to phase information, but the measures
implemented here are voxel-grid diagnostics and depend on resolution, smoothing, threshold convention,
and periodic boundaries. They are not continuum Minkowski-functional estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ExcursionTopology:
    threshold: float
    occupied_voxels: int
    volume_fraction: float
    exposed_face_count: int
    surface_area_mpc2: float
    euler_characteristic: int
    box_size_mpc: float
    grid_size: int
    periodic: bool
    threshold_kind: str

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "occupied_voxels": self.occupied_voxels,
            "volume_fraction": self.volume_fraction,
            "exposed_face_count": self.exposed_face_count,
            "surface_area_mpc2": self.surface_area_mpc2,
            "euler_characteristic": self.euler_characteristic,
            "box_size_mpc": self.box_size_mpc,
            "grid_size": self.grid_size,
            "periodic": self.periodic,
            "threshold_kind": self.threshold_kind,
            "warning": "Voxel topology changes with grid resolution, smoothing and thresholding; compare matched pipelines.",
        }


def _validate_field(field: np.ndarray, box_size_mpc: float) -> np.ndarray:
    x = np.asarray(field, float)
    if x.ndim != 3 or len(set(x.shape)) != 1 or x.shape[0] < 2:
        raise ValueError("field must be a cubic 3-D array with side >=2")
    if np.any(~np.isfinite(x)) or not np.isfinite(box_size_mpc) or box_size_mpc <= 0:
        raise ValueError("field must be finite and box_size_mpc positive")
    return x


def _cell_complex_euler(occupied: np.ndarray, *, periodic: bool) -> int:
    """Compute Euler characteristic of the union of occupied cubical voxels.

    Unique vertices, edges and square faces are deduplicated in integer lattice coordinates. With
    periodic boundaries coordinates are reduced modulo N, which gives χ=0 for a completely filled
    3-torus as required. Complexity is O(number of occupied voxels) with Python sets; use compiled code
    for large production meshes.
    """
    n = occupied.shape[0]
    cubes = np.argwhere(occupied)
    if cubes.size == 0:
        return 0
    vertices: set[tuple[int, int, int]] = set()
    edges: set[tuple[int, int, int, int]] = set()
    faces: set[tuple[int, int, int, int]] = set()

    def canonical(coord: tuple[int, int, int]) -> tuple[int, int, int]:
        if periodic:
            return tuple(v % n for v in coord)
        return coord

    for i, j, k in cubes:
        i, j, k = int(i), int(j), int(k)
        # Eight vertices of this cube.
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    vertices.add(canonical((i + dx, j + dy, k + dz)))
        # Edges: orientation 0=x,1=y,2=z; coordinate is the lower endpoint.
        for dx in (0, 1):
            for dy in (0, 1):
                edges.add((0, *canonical((i, j + dy, k + dx))))
        for dx in (0, 1):
            for dz in (0, 1):
                edges.add((1, *canonical((i + dx, j, k + dz))))
        # Edges parallel to z vary only in k; the tuple stores their lower endpoint.
        for dx in (0, 1):
            for dy in (0, 1):
                # overwrite/add correct z-edge at x=i+dx, y=j+dy, z=k
                edges.add((2, *canonical((i + dx, j + dy, k))))
        # Faces: orientation normal axis x/y/z; coordinate is the lower face corner.
        for dx in (0, 1):
            faces.add((0, *canonical((i + dx, j, k))))
        for dy in (0, 1):
            faces.add((1, *canonical((i, j + dy, k))))
        for dz in (0, 1):
            faces.add((2, *canonical((i, j, k + dz))))
    return int(len(vertices) - len(edges) + len(faces) - cubes.shape[0])


def excursion_topology(
    field: np.ndarray,
    threshold: float,
    box_size_mpc: float,
    *,
    threshold_kind: str = "field_value",
    periodic: bool = True,
    occupancy: str = "above",
) -> ExcursionTopology:
    """Compute a thresholded excursion set, exposed voxel-face area, and Euler characteristic.

    ``threshold_kind='field_value'`` compares field values directly. ``'quantile'`` interprets the
    threshold as a quantile in [0,1]. Occupied cells can be ``above`` or ``below`` the threshold.
    Surface area is based on exposed faces; periodic mode wraps neighbor comparisons across box edges.
    """
    x = _validate_field(field, box_size_mpc)
    if threshold_kind not in {"field_value", "quantile"}:
        raise ValueError("threshold_kind must be 'field_value' or 'quantile'")
    if threshold_kind == "quantile" and not 0 <= threshold <= 1:
        raise ValueError("quantile threshold must lie in [0,1]")
    if occupancy not in {"above", "below"}:
        raise ValueError("occupancy must be 'above' or 'below'")
    cut = float(np.quantile(x, threshold)) if threshold_kind == "quantile" else float(threshold)
    occ = x >= cut if occupancy == "above" else x <= cut
    n = x.shape[0]
    exposed = 0
    for axis in range(3):
        if periodic:
            neighbour = np.roll(occ, 1, axis=axis)
            exposed += int(np.count_nonzero(occ != neighbour))
        else:
            # Interior occupied/unoccupied interfaces counted once; outer box boundary faces counted once.
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(1, None)
            b[axis] = slice(None, -1)
            exposed += int(np.count_nonzero(occ[tuple(a)] != occ[tuple(b)]))
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis] = 0
            high[axis] = -1
            exposed += int(np.count_nonzero(occ[tuple(low)])) + int(
                np.count_nonzero(occ[tuple(high)])
            )
    dx = box_size_mpc / n
    # For periodic box each occupied-unoccupied interface is seen on both sides by occ != roll(occ,1).
    if periodic:
        exposed //= 1  # each interface counted once by the single shifted difference per axis
    area = exposed * dx**2
    euler = _cell_complex_euler(occ, periodic=periodic)
    return ExcursionTopology(
        cut,
        int(occ.sum()),
        float(occ.mean()),
        exposed,
        float(area),
        euler,
        float(box_size_mpc),
        n,
        bool(periodic),
        threshold_kind + ":" + occupancy,
    )


def topology_curve(
    field: np.ndarray,
    thresholds: np.ndarray,
    box_size_mpc: float,
    *,
    threshold_kind: str = "field_value",
    periodic: bool = True,
    occupancy: str = "above",
) -> tuple[ExcursionTopology, ...]:
    """Compute a threshold curve in exactly the supplied order (typically sorted by user)."""
    t = np.asarray(thresholds, float)
    if t.ndim != 1 or t.size < 2 or np.any(~np.isfinite(t)):
        raise ValueError("thresholds must be a finite vector with at least two values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("thresholds must be strictly increasing")
    return tuple(
        excursion_topology(
            field,
            float(v),
            box_size_mpc,
            threshold_kind=threshold_kind,
            periodic=periodic,
            occupancy=occupancy,
        )
        for v in t
    )


def topology_resolution_comparison(
    field_low: np.ndarray,
    field_high: np.ndarray,
    box_size_mpc: float,
    *,
    quantile_thresholds: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8),
) -> dict[str, object]:
    """Compare morphology at matched quantile thresholds for a low/high-resolution field pair."""
    a, b = _validate_field(field_low, box_size_mpc), _validate_field(field_high, box_size_mpc)
    q = tuple(float(x) for x in quantile_thresholds)
    if len(q) < 2 or any(x < 0 or x > 1 for x in q) or any(y <= x for x, y in zip(q, q[1:])):
        raise ValueError("quantile thresholds must be sorted, unique and lie in [0,1]")
    ra = [excursion_topology(a, t, box_size_mpc, threshold_kind="quantile") for t in q]
    rb = [
        excursion_topology(b, box_size_mpc=box_size_mpc, threshold=t, threshold_kind="quantile")
        for t in q
    ]
    return {
        "quantiles": list(q),
        "low_grid_size": int(a.shape[0]),
        "high_grid_size": int(b.shape[0]),
        "low_surface_area_mpc2": [r.surface_area_mpc2 for r in ra],
        "high_surface_area_mpc2": [r.surface_area_mpc2 for r in rb],
        "low_euler_characteristic": [r.euler_characteristic for r in ra],
        "high_euler_characteristic": [r.euler_characteristic for r in rb],
        "warning": "Quantile thresholds match volume fraction, not physical smoothing scale; comparison requires matched smoothing and field normalization.",
    }


__all__ = [
    "ExcursionTopology",
    "excursion_topology",
    "topology_curve",
    "topology_resolution_comparison",
]
