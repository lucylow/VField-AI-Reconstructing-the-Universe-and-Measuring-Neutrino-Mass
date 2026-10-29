"""Content-addressed cache for expensive Boltzmann-solver spectra.

Caching is useful during finite-difference derivative studies and parameter-grid construction. The key
includes parameters, requested k grid, redshift, backend class/version and selected public configuration
attributes. Every archive carries an integrity fingerprint and adjacent provenance JSON. A cache hit
only avoids recomputation; it does not validate the cached theory against observations or another solver.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum

CACHE_SCHEMA_VERSION = "1.0"


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(np.asarray(value))
        digest.update(str(arr.dtype).encode())
        digest.update(json.dumps(arr.shape).encode())
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _stable_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def backend_configuration(backend: object) -> dict[str, object]:
    """Capture public, simple-value backend fields likely to affect numerical outputs."""
    result: dict[str, object] = {}
    for name, value in sorted(vars(backend).items() if hasattr(backend, "__dict__") else []):
        if name.startswith("_"):
            continue
        if isinstance(value, (str, bool, int)):
            result[name] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[name] = value
        elif value is None:
            result[name] = None
    return result


@dataclass(frozen=True, slots=True)
class CachedSpectrumResult:
    spectrum: MatterPowerSpectrum
    cache_hit: bool
    cache_key: str
    archive_path: str
    manifest_path: str
    manifest: dict[str, object]


def make_theory_cache_key(
    backend: object,
    params: PhysicalCosmology,
    redshift: float,
    k_mpc_inv: np.ndarray,
    *,
    override_metadata: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Return SHA-256 cache key and canonical key payload."""
    k = np.asarray(k_mpc_inv, float)
    if not math.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and non-negative")
    if (
        k.ndim != 1
        or k.size < 3
        or np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError("k_mpc_inv must be positive, finite, increasing and contain >=3 samples")
    payload: dict[str, object] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "backend_version": str(getattr(backend, "version", "unknown")),
        "backend_configuration": backend_configuration(backend),
        "physical_cosmology": params.to_dict(),
        "redshift": float(redshift),
        "k_mpc_inv_sha256": hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest(),
        "k_count": int(k.size),
        "k_min_mpc_inv": float(k[0]),
        "k_max_mpc_inv": float(k[-1]),
        "override_metadata": override_metadata or {},
    }
    return hashlib.sha256(_stable_json(payload)).hexdigest(), payload


class TheorySpectrumCache:
    """Filesystem-backed deterministic cache of ``MatterPowerSpectrum`` outputs."""

    def __init__(self, root: str | Path, *, verify_integrity: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verify_integrity = bool(verify_integrity)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.npz", self.root / f"{key}.json"

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @staticmethod
    def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def get(
        self,
        key: str,
        *,
        expected_payload: dict[str, object] | None = None,
    ) -> CachedSpectrumResult | None:
        """Return a verified cached spectrum, or None on a miss/corrupt entry."""
        archive, sidecar = self._paths(key)
        if not archive.exists() or not sidecar.exists():
            return None
        try:
            manifest = json.loads(sidecar.read_text())
            if (
                manifest.get("cache_key") != key
                or manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            ):
                return None
            if expected_payload is not None and manifest.get("key_payload") != expected_payload:
                return None
            with np.load(archive, allow_pickle=False) as data:
                k = np.asarray(data["k_mpc_inv"], float)
                p = np.asarray(data["pk_mpc3"], float)
                z = float(np.asarray(data["redshift"]).item())
                linear = bool(np.asarray(data["linear"]).item())
            if self.verify_integrity and _array_digest(
                k, p, np.asarray([z]), np.asarray([linear])
            ) != manifest.get("array_sha256"):
                return None
            metadata = tuple(
                sorted((str(k), str(v)) for k, v in manifest.get("spectrum_metadata", {}).items())
            )
            spectrum = MatterPowerSpectrum(
                k,
                p,
                z,
                linear=linear,
                source=str(manifest.get("spectrum_source", "cached")),
                metadata=metadata,
            )
            return CachedSpectrumResult(spectrum, True, key, str(archive), str(sidecar), manifest)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, EOFError):
            return None

    def get_or_compute(
        self,
        backend,
        params: PhysicalCosmology,
        redshift: float,
        k_mpc_inv: np.ndarray,
        *,
        override_metadata: dict[str, object] | None = None,
        force: bool = False,
    ) -> CachedSpectrumResult:
        """Use the cache if valid; otherwise call the backend, validate, and atomically persist output."""
        k = np.asarray(k_mpc_inv, float)
        key, payload = make_theory_cache_key(
            backend, params, redshift, k, override_metadata=override_metadata
        )
        archive, sidecar = self._paths(key)
        if not force:
            cached = self.get(key, expected_payload=payload)
            if cached is not None:
                return cached
        spectrum = backend.matter_power(params, float(redshift), k)
        if not np.isclose(spectrum.redshift, redshift, atol=1e-10, rtol=0):
            raise ValueError("backend returned spectrum at a redshift different from the cache key")
        values = spectrum.evaluate(k)
        arrays = (
            spectrum.k_mpc_inv,
            spectrum.pk_mpc3,
            np.asarray([spectrum.redshift]),
            np.asarray([spectrum.linear]),
        )
        data_hash = _array_digest(*arrays)
        manifest: dict[str, object] = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "key_payload": payload,
            "array_sha256": data_hash,
            "spectrum_source": spectrum.source,
            "spectrum_metadata": dict(spectrum.metadata),
            "linear": bool(spectrum.linear),
            "redshift": float(spectrum.redshift),
            "created_by": "nufield.cosmology.physics.theory_cache",
            "scientific_note": "A cache hit is a performance feature, not external physical validation.",
        }
        self._write_npz_atomic(
            archive,
            k_mpc_inv=spectrum.k_mpc_inv,
            pk_mpc3=values,
            redshift=np.asarray(spectrum.redshift),
            linear=np.asarray(spectrum.linear),
        )
        self._write_json_atomic(sidecar, manifest)
        # Re-read through integrity verification to exercise exactly the same path as future cache hits.
        loaded = self.get(key, expected_payload=payload)
        if loaded is None:
            raise OSError("new cache entry failed integrity verification")
        return CachedSpectrumResult(
            loaded.spectrum, False, key, str(archive), str(sidecar), loaded.manifest
        )

    def remove(self, key: str) -> bool:
        """Remove one cache entry. Returns True if at least one file was removed."""
        archive, sidecar = self._paths(key)
        found = False
        for path in (archive, sidecar):
            try:
                path.unlink()
                found = True
            except FileNotFoundError:
                pass
        return found

    def inventory(self) -> dict[str, object]:
        """Count cache objects without loading arrays."""
        archives = list(self.root.glob("*.npz"))
        manifests = list(self.root.glob("*.json"))
        return {
            "root": str(self.root),
            "archives": len(archives),
            "manifests": len(manifests),
            "bytes_archives": int(sum(p.stat().st_size for p in archives)),
            "bytes_manifests": int(sum(p.stat().st_size for p in manifests)),
            "orphan_archives": sorted(
                p.stem for p in archives if not p.with_suffix(".json").exists()
            ),
        }


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CachedSpectrumResult",
    "backend_configuration",
    "make_theory_cache_key",
    "TheorySpectrumCache",
]
