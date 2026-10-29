"""PyTorch datasets and worker-safe streaming for large simulation arrays."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch, Dataset = None, object


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs using the PyTorch worker-specific seed."""
    import random

    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to seed DataLoader workers")
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


if torch is not None:

    class ArraySimulationDataset(Dataset):
        """Map-style dataset for NumPy arrays or dicts of fixed-shape modalities.

        It stores array references and converts individual rows to tensors lazily. This avoids a second
        full copy when wrapping large loaded arrays, though the underlying arrays must still fit RAM.
        """

        def __init__(self, inputs: Any, targets: Any, *, target_dtype=None):
            self.inputs = inputs
            self.targets = np.asarray(targets)
            if (
                self.targets.ndim < 1
                or len(self.targets) == 0
                or not np.isfinite(self.targets).all()
            ):
                raise ValueError("targets must be nonempty and finite")
            self.target_dtype = target_dtype or torch.float32
            if isinstance(inputs, dict):
                if not inputs:
                    raise ValueError("input modality dictionary cannot be empty")
                lengths = {k: len(v) for k, v in inputs.items()}
                if set(lengths.values()) != {len(self.targets)}:
                    raise ValueError("all modalities must align with targets")
                for key, value in inputs.items():
                    if not np.isfinite(np.asarray(value)).all():
                        raise ValueError(f"modality {key} contains non-finite values")
            else:
                x = np.asarray(inputs)
                if len(x) != len(self.targets) or not np.isfinite(x).all():
                    raise ValueError("inputs must align with targets and be finite")

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            y = torch.as_tensor(self.targets[index], dtype=self.target_dtype)
            if isinstance(self.inputs, dict):
                x = {
                    k: torch.as_tensor(np.asarray(v[index]), dtype=torch.float32)
                    for k, v in self.inputs.items()
                }
            else:
                x = torch.as_tensor(np.asarray(self.inputs[index]), dtype=torch.float32)
            return x, y

    class HDF5SimulationDataset(Dataset):
        """Worker-safe lazy HDF5 dataset.

        HDF5 files are opened lazily per process (PID) rather than sharing an open handle after fork.
        Requires ``pip install nufield-starter[data,ml]``. Use chunked/compressed datasets and a
        DataLoader with a conservative worker count when storage is network-mounted.
        """

        def __init__(
            self,
            path: str | Path,
            *,
            input_key: str,
            target_key: str,
            group_key: str | None = None,
            input_transform=None,
            target_transform=None,
        ):
            try:
                import h5py
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Install nufield-starter[data,ml] for HDF5 streaming"
                ) from exc
            self.path = str(path)
            self.input_key, self.target_key, self.group_key = input_key, target_key, group_key
            self.input_transform, self.target_transform = input_transform, target_transform
            self._handle = None
            self._pid = None
            with h5py.File(self.path, "r") as handle:
                for key in (input_key, target_key):
                    if key not in handle:
                        raise KeyError(f"HDF5 dataset key not found: {key}")
                self._length = int(handle[input_key].shape[0])
                self._input_shape = tuple(handle[input_key].shape[1:])
                self._target_shape = tuple(handle[target_key].shape[1:])
                if handle[target_key].shape[0] != self._length:
                    raise ValueError("input and target row counts differ")
                if group_key is not None and (
                    group_key not in handle or handle[group_key].shape[0] != self._length
                ):
                    raise ValueError("group key missing or group row count mismatch")

        def _file(self):
            import h5py

            pid = os.getpid()
            if self._handle is None or self._pid != pid:
                if self._handle is not None:
                    try:
                        self._handle.close()
                    except Exception:
                        pass
                self._handle = h5py.File(self.path, "r")
                self._pid = pid
            return self._handle

        def __len__(self):
            return self._length

        def __getitem__(self, index):
            if not 0 <= int(index) < self._length:
                raise IndexError(index)
            h = self._file()
            x = np.asarray(h[self.input_key][index])
            y = np.asarray(h[self.target_key][index])
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"non-finite HDF5 row at index {index}")
            if self.input_transform is not None:
                x = self.input_transform(x)
            if self.target_transform is not None:
                y = self.target_transform(y)
            sample = (
                torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
            )
            if self.group_key is not None:
                group = h[self.group_key][index]
                if isinstance(group, bytes):
                    group = group.decode("utf-8")
                return (sample[0], sample[1], str(group))
            return sample

        def close(self):
            if self._handle is not None:
                self._handle.close()
                self._handle = None
                self._pid = None

        def __getstate__(self):
            state = self.__dict__.copy()
            state["_handle"] = None
            state["_pid"] = None
            return state

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass
else:

    class ArraySimulationDataset:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError(
                "Install nufield-starter[ml] to use ArraySimulationDataset"
            )

    class HDF5SimulationDataset:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError(
                "Install nufield-starter[data,ml] to use HDF5SimulationDataset"
            )


def estimate_array_memory_bytes(
    shapes: dict[str, tuple[int, ...]],
    *,
    dtype_bytes: dict[str, int] | None = None,
    copies: int = 2,
    workers: int = 0,
) -> dict[str, Any]:
    """Estimate lower-bound data memory for configured arrays and worker-local copies."""
    if copies < 1 or workers < 0:
        raise ValueError("copies must be >=1 and workers nonnegative")
    dtype_bytes = dtype_bytes or {}
    parts = {}
    total = 0
    for name, shape in shapes.items():
        if not shape or any(int(d) < 1 for d in shape):
            raise ValueError(f"invalid shape for {name}")
        n = int(np.prod(shape)) * int(dtype_bytes.get(name, 4))
        parts[name] = n
        total += n
    peak = total * copies + total * workers
    return {
        "per_array_bytes": parts,
        "dataset_bytes": int(total),
        "copies": copies,
        "workers": workers,
        "lower_bound_peak_bytes": int(peak),
        "lower_bound_peak_gib": float(peak / 1024**3),
        "caveat": "Does not include model activations, optimizer state, HDF5 caches, pinned memory or temporary arrays.",
    }
