"""Streaming HDF5 particle-catalog -> periodic mesh -> feature-dataset pipeline.

This is a generic adapter for explicitly mapped periodic simulation particle positions. It is NOT a
reader for arbitrary Quijote/Gadget snapshots: use it only when the selected product really contains
an HDF5 position array with the declared coordinate convention. Observed DESI RA/DEC/redshift
catalogs are not periodic simulation-box coordinates and must not be processed with this routine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nufield.data_registry.cic import cloud_in_cell_overdensity
from nufield.data_registry.field_features import field_summary
from nufield.data_registry.manifest import sha256_file


def process_hdf5_particle_batch(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    positions_key: str,
    theta_key: str,
    parameter_names: list[str],
    simulation_id_key: str,
    group_id_key: str,
    box_size_mpc_h: float,
    grid_size: int,
    redshift: float,
    power_bins: int = 16,
    max_memory_gb: float = 8.0,
    position_dtype: str = "float32",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Process one HDF5 file with one realization per leading dimension, saving `.npy` memmaps.

    Required expected schema: positions `[N,Nparticles,3]` or `[Nparticles,3]` for a single run;
    theta `[N,P]` or `[P]` for one run; IDs and parent group IDs `[N]` or scalar strings. Dataset
    paths are passed explicitly and never guessed. The routine verifies row alignment and writes
    each simulation field/features sequentially to memory-mapped arrays to avoid a full output cube
    batch in RAM. Individual position arrays are still loaded one realization at a time.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install nufield-starter[data] to process HDF5") from exc
    src = Path(input_path)
    out = Path(output_dir)
    if not src.exists():
        raise FileNotFoundError(src)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{out} is not empty; use overwrite=True only when replacing a derived dataset intentionally"
        )
    if box_size_mpc_h <= 0 or grid_size < 4 or redshift < 0 or power_bins < 2 or max_memory_gb <= 0:
        raise ValueError("invalid box/resolution/redshift/bin/memory value")
    if not parameter_names or len(set(parameter_names)) != len(parameter_names):
        raise ValueError("parameter names must be unique and nonempty")
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as h5:
        for key in (positions_key, theta_key, simulation_id_key, group_id_key):
            if key not in h5:
                raise KeyError(
                    f"HDF5 key {key!r} absent; inspect_hdf5_tree and map the official schema explicitly"
                )
        pds = h5[positions_key]
        tds = h5[theta_key]
        ids_ds = h5[simulation_id_key]
        grp_ds = h5[group_id_key]
        if pds.ndim == 2:
            if pds.shape[1] != 3:
                raise ValueError("positions_key must be [Nparticles,3] or [N,Nparticles,3]")
            n = 1
            n_particles = int(pds.shape[0])
        elif pds.ndim == 3 and pds.shape[2] == 3:
            n = int(pds.shape[0])
            n_particles = int(pds.shape[1])
        else:
            raise ValueError("positions array shape must be [Nparticles,3] or [N,Nparticles,3]")
        if tds.ndim == 1 and n == 1:
            theta_shape = (1, int(tds.shape[0]))
        elif tds.ndim == 2 and tds.shape[0] == n:
            theta_shape = tuple(tds.shape)
        else:
            raise ValueError(
                "theta shape must be [N,P], or [P] when input contains one realization"
            )
        if theta_shape[1] != len(parameter_names):
            raise ValueError("theta columns differ from parameter_names")
        if len(ids_ds.shape) > 1 or len(grp_ds.shape) > 1:
            raise ValueError("simulation/group ID datasets must be 1-D or scalar for N=1")
        if n > 1 and (ids_ds.shape[0] != n or grp_ds.shape[0] != n):
            raise ValueError("ID/group arrays must align with realization axis")
        if n == 1 and ids_ds.shape not in {(), (1,)}:
            raise ValueError("single-realization ID must be scalar or length 1")
        # Estimate working set before allocating output arrays. Particle coordinates are read one row at a time.
        positions_bytes = n_particles * 3 * np.dtype(position_dtype).itemsize
        mesh_bytes = grid_size**3 * 8
        estimated_work_gb = (positions_bytes + mesh_bytes * 3) / (1024**3)
        if estimated_work_gb > max_memory_gb:
            raise MemoryError(
                f"one realization estimated at {estimated_work_gb:.2f} GB > max_memory_gb={max_memory_gb:.2f}; lower grid/position dtype or use HPC"
            )
        # NPY memory maps allow large arrays without allocating the full derived dataset in process memory.
        fields = np.lib.format.open_memmap(
            out / "fields.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, grid_size, grid_size, grid_size),
        )
        features = None
        feature_names = None
        theta_out = np.lib.format.open_memmap(
            out / "theta.npy", mode="w+", dtype=np.float64, shape=theta_shape
        )
        ids = []
        groups = []
        for i in range(n):
            pos = np.asarray(pds[...] if pds.ndim == 2 else pds[i], dtype=position_dtype)
            if not np.isfinite(pos).all():
                raise ValueError(f"non-finite particle positions in realization {i}")
            # Require input positions already wrapped. No silently wrapping possibly mis-scaled coordinates.
            field = cloud_in_cell_overdensity(pos, box_size=box_size_mpc_h, grid_size=grid_size)
            fields[i] = field.astype(np.float32)
            summary, names = field_summary(field, box_size=box_size_mpc_h, power_bins=power_bins)
            if features is None:
                features = np.lib.format.open_memmap(
                    out / "features.npy", mode="w+", dtype=np.float32, shape=(n, len(summary))
                )
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("feature schema changed across realizations")
            features[i] = summary.astype(np.float32)
            raw_theta = np.asarray(tds[...] if tds.ndim == 1 else tds[i], dtype=float).reshape(-1)
            if not np.isfinite(raw_theta).all():
                raise ValueError(f"non-finite parameters in realization {i}")
            theta_out[i] = raw_theta
            raw_id = ids_ds[()] if ids_ds.shape == () else ids_ds[i]
            raw_group = grp_ds[()] if grp_ds.shape == () else grp_ds[i]
            ids.append(_decode(raw_id))
            groups.append(_decode(raw_group))
        fields.flush()
        theta_out.flush()
        features.flush()
    if len(set(ids)) != len(ids):
        # IDs may be repeated only if this HDF5 contains repeated views; that should be represented
        # by explicit parent groups and reviewed. This simple pipeline assumes one row per run.
        raise ValueError(
            "simulation IDs are not unique; this particle processor expects one row per realization"
        )
    np.save(out / "simulation_id.npy", np.asarray(ids, dtype=str))
    np.save(out / "group_id.npy", np.asarray(groups, dtype=str))
    np.save(out / "feature_names.npy", np.asarray(feature_names or [], dtype=str))
    manifest = {
        "schema_version": "1.0",
        "dataset_id": out.name,
        "source_file": str(src.resolve()),
        "source_sha256": sha256_file(src),
        "source_keys": {
            "positions": positions_key,
            "theta": theta_key,
            "simulation_id": simulation_id_key,
            "group_id": group_id_key,
        },
        "n_realizations": n,
        "n_particles_per_realization": n_particles,
        "parameter_names": parameter_names,
        "observable_kind": "periodic_cic_overdensity_field",
        "field_shape": [grid_size] * 3,
        "box_size_mpc_h": box_size_mpc_h,
        "redshift": redshift,
        "mass_assignment": "Cloud-in-Cell, periodic",
        "feature_method": "field_summary + volume-normalized periodic power estimator",
        "power_bins": power_bins,
        "position_dtype": position_dtype,
        "estimated_working_memory_gb": estimated_work_gb,
        "output_arrays": {
            "fields": "fields.npy",
            "features": "features.npy",
            "theta": "theta.npy",
            "simulation_id": "simulation_id.npy",
            "group_id": "group_id.npy",
            "feature_names": "feature_names.npy",
        },
        "physics_level": "prototype_requires_reference_validation",
        "scientific_approval": False,
        "limitations": [
            "Input must already be a periodic simulation-box particle catalog in [0,L).",
            "This routine does not read Gadget binary snapshots or infer units from a file.",
            "CIC assignment and P(k) normalization should be validated against a trusted cosmology pipeline.",
            "No galaxy bias, redshift-space distortion, survey window, covariance, or likelihood is included.",
        ],
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _decode(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)
