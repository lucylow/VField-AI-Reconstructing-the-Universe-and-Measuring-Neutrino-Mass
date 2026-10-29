"""Streaming HDF5 mesh-field dataset builder for AI-ready feature/field arrays.

This handles a deliberately narrow, explicit contract: periodic density or overdensity cubes with
shape [N,D,H,W] or [N,C,D,H,W], paired `theta`, unique `simulation_id`, and `group_id`. It does not
read release-specific simulation formats automatically. Survey masks and observed DESI catalogs must
not be treated as periodic cubes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nufield.data_registry.field_features import field_summary
from nufield.data_registry.manifest import sha256_file


def process_hdf5_field_batch(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    fields_key: str,
    theta_key: str,
    parameter_names: list[str],
    simulation_id_key: str,
    group_id_key: str,
    box_size_mpc_h: float,
    redshift: float,
    field_kind: str,
    power_bins: int = 16,
    summary_channel: int = 0,
    max_memory_gb: float = 8.0,
    copy_fields: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install nufield-starter[data] for HDF5 field processing") from exc
    src = Path(input_path)
    out = Path(output_dir)
    if not src.is_file():
        raise FileNotFoundError(src)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; use overwrite=True intentionally")
    if field_kind not in {"density", "overdensity", "matter_density", "galaxy_density_mock"}:
        raise ValueError("field_kind must be explicitly documented")
    if box_size_mpc_h <= 0 or redshift < 0 or power_bins < 2 or max_memory_gb <= 0:
        raise ValueError("invalid box/redshift/bin/memory parameter")
    if not parameter_names or len(parameter_names) != len(set(parameter_names)):
        raise ValueError("parameter names must be unique and nonempty")
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as h5:
        required = [fields_key, theta_key, simulation_id_key, group_id_key]
        missing = [k for k in required if k not in h5]
        if missing:
            raise KeyError(f"missing exact HDF5 keys {missing}; inspect source schema first")
        fds = h5[fields_key]
        tds = h5[theta_key]
        idsds = h5[simulation_id_key]
        gds = h5[group_id_key]
        if fds.ndim == 3:
            n = 1
            channels = 1
            shape = tuple(map(int, fds.shape))
        elif fds.ndim == 4:
            n = int(fds.shape[0])
            channels = 1
            shape = tuple(map(int, fds.shape[1:]))
        elif fds.ndim == 5:
            n = int(fds.shape[0])
            channels = int(fds.shape[1])
            shape = tuple(map(int, fds.shape[2:]))
        else:
            raise ValueError("fields shape must be [D,H,W], [N,D,H,W], or [N,C,D,H,W]")
        if len(set(shape)) != 1 or min(shape) < 4:
            raise ValueError(
                "only cubic fields with resolution >=4 are supported in this prototype"
            )
        if not 0 <= summary_channel < channels:
            raise ValueError("summary_channel out of bounds")
        if tds.ndim == 1 and n == 1:
            theta_shape = (1, int(tds.shape[0]))
        elif tds.ndim == 2 and tds.shape[0] == n:
            theta_shape = tuple(tds.shape)
        else:
            raise ValueError("theta array must be [N,P] or [P] for a single cube")
        if theta_shape[1] != len(parameter_names):
            raise ValueError("theta columns differ from parameter_names")
        if n > 1 and (idsds.shape != (n,) or gds.shape != (n,)):
            raise ValueError("simulation_id and group_id arrays must each be [N]")
        if n == 1 and idsds.shape not in {(), (1,)}:
            raise ValueError("single simulation ID must be scalar or length one")
        row_bytes = int(np.prod(shape)) * channels * np.dtype(fds.dtype).itemsize
        working_estimate = (row_bytes * 3 + int(np.prod(shape)) * 8 * 3) / (1024**3)
        if working_estimate > max_memory_gb:
            raise MemoryError(
                f"one field estimates {working_estimate:.2f} GiB working memory > configured {max_memory_gb:.2f} GiB"
            )
        dims = (n, channels, *shape)
        fields_out = None
        if copy_fields:
            fields_out = np.lib.format.open_memmap(
                out / "fields.npy", mode="w+", dtype=np.float32, shape=dims
            )
        features_out = None
        names = None
        theta_out = np.lib.format.open_memmap(
            out / "theta.npy", mode="w+", dtype=np.float64, shape=theta_shape
        )
        ids = []
        groups = []
        for i in range(n):
            if fds.ndim == 3:
                raw = np.asarray(fds[...], dtype=np.float32)[None]
            elif fds.ndim == 4:
                raw = np.asarray(fds[i], dtype=np.float32)[None]
            else:
                raw = np.asarray(fds[i], dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError(f"non-finite field at row {i}")
            if fields_out is not None:
                fields_out[i] = raw
            vals, names_now = field_summary(
                raw[summary_channel], box_size=box_size_mpc_h, power_bins=power_bins
            )
            if features_out is None:
                features_out = np.lib.format.open_memmap(
                    out / "features.npy", mode="w+", dtype=np.float32, shape=(n, len(vals))
                )
                names = names_now
            elif names_now != names:
                raise RuntimeError("feature schema changed across rows")
            features_out[i] = vals.astype(np.float32)
            raw_theta = np.asarray(tds[()] if tds.ndim == 1 else tds[i], dtype=np.float64).reshape(
                -1
            )
            if not np.isfinite(raw_theta).all():
                raise ValueError(f"non-finite theta at row {i}")
            theta_out[i] = raw_theta
            ids.append(_decode(idsds[()] if idsds.shape == () else idsds[i]))
            groups.append(_decode(gds[()] if gds.shape == () else gds[i]))
        theta_out.flush()
        features_out.flush()
        if fields_out is not None:
            fields_out.flush()
    if len(set(ids)) != len(ids):
        raise ValueError(
            "simulation IDs must be unique row IDs; represent related crops/views by shared group IDs"
        )
    np.save(out / "simulation_id.npy", np.asarray(ids, dtype=str))
    np.save(out / "group_id.npy", np.asarray(groups, dtype=str))
    np.save(out / "feature_names.npy", np.asarray(names, dtype=str))
    manifest = {
        "schema_version": "1.0",
        "dataset_id": out.name,
        "source_path": str(src.resolve()),
        "source_sha256": sha256_file(src),
        "source_keys": {
            "fields": fields_key,
            "theta": theta_key,
            "simulation_id": simulation_id_key,
            "group_id": group_id_key,
        },
        "n_rows": n,
        "input_field_shape": list(fds.shape),
        "output_field_shape": list(dims) if copy_fields else None,
        "field_kind": field_kind,
        "summary_channel": summary_channel,
        "parameter_names": parameter_names,
        "box_size_mpc_h": box_size_mpc_h,
        "redshift": redshift,
        "power_bins": power_bins,
        "estimated_working_memory_gb": working_estimate,
        "copied_fields": copy_fields,
        "feature_schema": {"names": names, "width": len(names)},
        "geometry": "periodic box; caller-declared",
        "physics_level": "prototype_requires_reference_validation",
        "scientific_approval": False,
        "warnings": [
            "No source units or physical semantics were inferred from HDF5 names.",
            "CIC and P(k) normalization are not applied here; input fields are assumed already gridded.",
            "The field_summary estimator is a prototype and must be checked against a reference implementation.",
            "No survey window, redshift-space distortions, covariance, or likelihood is included.",
        ],
        "output_arrays": {
            "features": "features.npy",
            "theta": "theta.npy",
            "simulation_id": "simulation_id.npy",
            "group_id": "group_id.npy",
            "feature_names": "feature_names.npy",
            **({"fields": "fields.npy"} if copy_fields else {}),
        },
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)
