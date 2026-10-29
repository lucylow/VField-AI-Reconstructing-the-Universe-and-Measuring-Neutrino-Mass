"""Format-agnostic local array loading with explicit key maps and no guessed schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayBundle:
    observations: np.ndarray
    theta: np.ndarray
    simulation_ids: np.ndarray
    group_ids: np.ndarray
    parameter_names: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self):
        if len(self.observations) != len(self.theta):
            raise ValueError("observations/theta rows mismatch")
        if len(self.simulation_ids) != len(self.theta):
            raise ValueError("simulation_ids/theta rows mismatch")
        if len(self.group_ids) != len(self.theta):
            raise ValueError("group_ids/theta rows mismatch")
        if self.theta.ndim != 2 or self.observations.ndim < 2:
            raise ValueError("invalid observations or theta dimensions")
        if self.theta.shape[1] != len(self.parameter_names):
            raise ValueError("parameter_names length mismatch")


def load_npz_bundle(
    path: str | Path,
    *,
    observation_key: str,
    theta_key: str,
    parameter_names: list[str],
    simulation_id_key: str = "simulation_id",
    group_id_key: str | None = None,
    allow_synthetic_ids: bool = False,
) -> ArrayBundle:
    """Load a user-mapped NPZ file. Never assumes the key names used by an external suite."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        required = {observation_key, theta_key}
        if not required.issubset(keys):
            raise KeyError(
                f"missing keys {sorted(required - keys)}; available keys: {sorted(keys)}"
            )
        x = np.asarray(z[observation_key])
        theta = np.asarray(z[theta_key])
        if simulation_id_key in keys:
            ids = np.asarray(z[simulation_id_key]).astype(str)
        elif allow_synthetic_ids:
            ids = np.asarray([f"row-{i:08d}" for i in range(len(theta))])
        else:
            raise KeyError(
                f"no {simulation_id_key!r} key; set allow_synthetic_ids only for a documented smoke test"
            )
        if group_id_key and group_id_key in keys:
            groups = np.asarray(z[group_id_key]).astype(str)
        elif group_id_key and not allow_synthetic_ids:
            raise KeyError(f"group key {group_id_key!r} is absent")
        else:
            groups = ids.copy()
    if x.ndim < 2 or theta.ndim != 2:
        raise ValueError("observations must be [N,...], theta must be [N,P]")
    if len(x) != len(theta) or len(ids) != len(theta) or len(groups) != len(theta):
        raise ValueError("all arrays must share the same sample dimension")
    if len(parameter_names) != theta.shape[1]:
        raise ValueError("parameter_names length differs from theta columns")
    return ArrayBundle(
        x,
        theta,
        ids,
        groups,
        tuple(parameter_names),
        {
            "path": str(path.resolve()),
            "format": "npz",
            "array_keys": {
                "observations": observation_key,
                "theta": theta_key,
                "simulation_ids": simulation_id_key,
                "group_ids": group_id_key,
            },
            "allow_synthetic_ids": allow_synthetic_ids,
            "warning": "Array loading checks schema only; it does not validate physical meaning or simulator fidelity.",
        },
    )


def load_hdf5_bundle(
    path: str | Path,
    *,
    observation_key: str,
    theta_key: str,
    parameter_names: list[str],
    simulation_id_key: str | None = None,
    group_id_key: str | None = None,
    allow_synthetic_ids: bool = False,
) -> ArrayBundle:
    """Load mapped HDF5 arrays. Dataset paths must be copied from the source's actual data model."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install nufield-starter[data] to read HDF5 files") from exc
    path = Path(path)
    with h5py.File(path, "r") as h5:
        for key in (observation_key, theta_key):
            if key not in h5:
                raise KeyError(f"HDF5 key {key!r} not found in {path}")
        x = np.asarray(h5[observation_key])
        theta = np.asarray(h5[theta_key])
        if simulation_id_key and simulation_id_key in h5:
            ids = np.asarray(h5[simulation_id_key]).astype(str)
        elif allow_synthetic_ids:
            ids = np.asarray([f"row-{i:08d}" for i in range(len(theta))])
        else:
            raise KeyError(
                "simulation ID dataset absent; refusing to fabricate IDs for research data"
            )
        if group_id_key and group_id_key in h5:
            groups = np.asarray(h5[group_id_key]).astype(str)
        elif group_id_key:
            raise KeyError(f"group ID dataset {group_id_key!r} absent")
        else:
            groups = ids.copy()
    if x.ndim < 2 or theta.ndim != 2 or len(x) != len(theta):
        raise ValueError("invalid HDF5 observation/theta shape")
    if len(ids) != len(theta) or len(groups) != len(theta):
        raise ValueError("HDF5 IDs do not align with arrays")
    if theta.shape[1] != len(parameter_names):
        raise ValueError("parameter_names length mismatch")
    return ArrayBundle(
        x,
        theta,
        ids,
        groups,
        tuple(parameter_names),
        {
            "path": str(path.resolve()),
            "format": "hdf5",
            "array_keys": {
                "observations": observation_key,
                "theta": theta_key,
                "simulation_ids": simulation_id_key,
                "group_ids": group_id_key,
            },
            "warning": "HDF5 schema was explicitly mapped; physics, units, IDs, and selection design require independent review.",
        },
    )


def inspect_hdf5_tree(path: str | Path, *, max_items: int = 200) -> list[dict[str, Any]]:
    """List HDF5 dataset paths and shapes to aid explicit schema mapping; no payloads are read."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install nufield-starter[data] to inspect HDF5 files") from exc
    found: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:

        def visit(name, obj):
            if len(found) >= max_items:
                return
            if isinstance(obj, h5py.Dataset):
                found.append(
                    {
                        "path": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "compression": obj.compression,
                    }
                )

        h5.visititems(visit)
    return found
