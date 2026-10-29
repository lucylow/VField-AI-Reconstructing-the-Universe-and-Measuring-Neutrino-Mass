"""Physics-aware contracts for massive-neutrino simulation datasets.

The contract stores the independent simulation identity, cosmological parameter labels, field component,
redshift, Fourier/space units, and declared neutrino treatment. It is intentionally independent of one
specific data vendor or HDF5 layout so dataset loaders can be validated before neural training.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NeutrinoSimulationRecord:
    dataset: str
    release: str
    parent_simulation_id: str
    realization_id: str
    sum_mnu_ev: float
    redshift: float
    field_component: str
    density_contrast: str
    box_size_mpc: float
    grid_size: int
    k_units: str
    length_units: str
    power_units: str
    simulation_method: str
    initial_condition_method: str
    cosmology_parameters: dict[str, float]
    neutrino_treatment: str
    neutrino_mass_hierarchy: str
    source_uri: str

    def __post_init__(self) -> None:
        required = (
            self.dataset,
            self.release,
            self.parent_simulation_id,
            self.realization_id,
            self.field_component,
            self.density_contrast,
            self.k_units,
            self.length_units,
            self.power_units,
            self.simulation_method,
            self.initial_condition_method,
            self.neutrino_treatment,
            self.neutrino_mass_hierarchy,
            self.source_uri,
        )
        if any(not str(x).strip() for x in required):
            raise ValueError("all dataset provenance and physics convention strings are required")
        if not math.isfinite(self.sum_mnu_ev) or self.sum_mnu_ev < 0:
            raise ValueError("sum_mnu_ev must be finite and non-negative")
        if not math.isfinite(self.redshift) or self.redshift < 0:
            raise ValueError("redshift must be finite and non-negative")
        if not math.isfinite(self.box_size_mpc) or self.box_size_mpc <= 0 or self.grid_size < 4:
            raise ValueError("box_size_mpc must be positive and grid_size >=4")
        if any(not math.isfinite(float(v)) for v in self.cosmology_parameters.values()):
            raise ValueError("cosmology parameter labels must be finite")
        if self.sum_mnu_ev > 0 and self.neutrino_treatment.lower() in {
            "none",
            "massless",
            "omitted",
            "unknown",
        }:
            raise ValueError("nonzero neutrino mass cannot be labeled massless/omitted/unknown")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def unique_sample_id(self) -> str:
        return f"{self.dataset}:{self.release}:{self.parent_simulation_id}:{self.realization_id}:z={self.redshift:g}"


@dataclass(frozen=True, slots=True)
class MassCoverageAudit:
    n_records: int
    n_parent_simulations: int
    n_unique_mass_values: int
    mass_min_ev: float
    mass_max_ev: float
    unique_mass_values_ev: tuple[float, ...]
    bin_counts: tuple[int, ...]
    bin_edges_ev: tuple[float, ...]
    zero_count_bins: tuple[int, ...]
    parameter_correlation_with_mass: dict[str, float]
    split_overlap_parents: tuple[str, ...]
    passed_basic_checks: bool
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "unique_mass_values_ev": list(self.unique_mass_values_ev),
            "bin_counts": list(self.bin_counts),
            "bin_edges_ev": list(self.bin_edges_ev),
            "zero_count_bins": list(self.zero_count_bins),
            "split_overlap_parents": list(self.split_overlap_parents),
            "caveats": list(self.caveats),
        }


def audit_mass_coverage(
    records: Sequence[NeutrinoSimulationRecord],
    *,
    mass_bin_edges_ev: Sequence[float] | None = None,
    split_assignments: Mapping[str, str] | None = None,
    required_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> MassCoverageAudit:
    """Audit mass support, parent independence and optional train/validation/test group leakage.

    ``split_assignments`` maps parent_simulation_id to a split name. All snapshots/crops from a parent
    simulation must stay in one split. Repeated mass points can be scientifically valid but are counted
    as repeated realizations, not independent cosmological parameter settings.
    """
    rows = tuple(records)
    if not rows:
        raise ValueError("at least one simulation record is required")
    masses = np.asarray([r.sum_mnu_ev for r in rows], float)
    parents = {r.parent_simulation_id for r in rows}
    edges = np.asarray(
        mass_bin_edges_ev
        if mass_bin_edges_ev is not None
        else np.linspace(float(masses.min()), float(masses.max()) + 1e-12, 6),
        float,
    )
    if (
        edges.ndim != 1
        or len(edges) < 2
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("mass_bin_edges_ev must be finite strictly increasing edges")
    counts, _ = np.histogram(masses, bins=edges)
    # Correlations are descriptive only; correlations are undefined if a label is constant.
    correlations: dict[str, float] = {}
    names = sorted(set.intersection(*(set(r.cosmology_parameters) for r in rows))) if rows else []
    for name in names:
        values = np.asarray([r.cosmology_parameters[name] for r in rows], float)
        if np.std(values) > 0 and np.std(masses) > 0:
            correlations[name] = float(np.corrcoef(values, masses)[0, 1])
    overlaps: set[str] = set()
    if split_assignments is not None:
        by_split: dict[str, set[str]] = {s: set() for s in required_splits}
        for parent, split in split_assignments.items():
            if split not in by_split:
                raise ValueError(f"unknown split {split!r}; expected one of {required_splits}")
            by_split[split].add(parent)
        for i, a in enumerate(required_splits):
            for b in required_splits[i + 1 :]:
                overlaps |= by_split[a] & by_split[b]
    passed = bool(len(parents) >= 3 and len(np.unique(masses)) >= 2 and len(overlaps) == 0)
    caveats = (
        "Mass coverage does not guarantee broad coverage of other cosmological parameters or systematics.",
        "A dataset with many snapshots/crops can still have few independent parent simulations.",
        "Correlation with mass reveals design coupling, not causal physical influence; inspect the full design matrix.",
        "A passed contract establishes metadata consistency, not that the simulator correctly models massive neutrinos.",
    )
    if split_assignments is None:
        caveats += (
            "No split assignment was supplied; train/validation/test leakage was not checked.",
        )
    return MassCoverageAudit(
        len(rows),
        len(parents),
        len(np.unique(masses)),
        float(masses.min()),
        float(masses.max()),
        tuple(float(x) for x in np.unique(masses)),
        tuple(int(x) for x in counts),
        tuple(float(x) for x in edges),
        tuple(int(i) for i in np.flatnonzero(counts == 0)),
        correlations,
        tuple(sorted(overlaps)),
        passed,
        caveats,
    )


def validate_field_payload(
    field: np.ndarray,
    record: NeutrinoSimulationRecord,
    *,
    expected_shape: tuple[int, int, int] | None = None,
    expected_dtype_kind: tuple[str, ...] = ("f", "i", "u"),
) -> dict[str, object]:
    """Validate a loaded gridded field against its declared volume/grid/unit metadata."""
    x = np.asarray(field)
    shape = expected_shape or (record.grid_size,) * 3
    if x.shape != shape or x.ndim != 3:
        raise ValueError(f"field shape {x.shape} does not match declared cubic grid {shape}")
    if x.dtype.kind not in expected_dtype_kind:
        raise ValueError(
            f"field dtype kind {x.dtype.kind!r} not among allowed kinds {expected_dtype_kind}"
        )
    if np.any(~np.isfinite(x)):
        raise ValueError("field contains NaN or infinite values")
    if record.length_units not in {"Mpc", "Mpc (comoving)"}:
        raise ValueError("physics pipeline currently requires physical comoving Mpc box units")
    if record.field_component.lower() not in {
        "total",
        "cb",
        "cdm",
        "baryon",
        "nu",
        "galaxy",
        "unknown",
    }:
        raise ValueError("field_component is not recognized; update contract before processing")
    return {
        "passed": True,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "record_id": record.unique_sample_id,
        "warning": "Finite shape-compatible field data are not a proof of physical correctness.",
    }


def validate_split_parent_independence(
    records: Sequence[NeutrinoSimulationRecord], split_assignments: Mapping[str, str]
) -> dict[str, object]:
    """Return a strict parent-group split audit, raising if any parent spans multiple splits."""
    assignments: dict[str, set[str]] = {}
    for record in records:
        if record.parent_simulation_id not in split_assignments:
            raise ValueError(f"missing split for parent {record.parent_simulation_id}")
        assignments.setdefault(record.parent_simulation_id, set()).add(
            split_assignments[record.parent_simulation_id]
        )
    leakage = {parent: sorted(labels) for parent, labels in assignments.items() if len(labels) != 1}
    if leakage:
        raise ValueError(f"parent simulations span multiple splits: {leakage}")
    grouped: dict[str, int] = {}
    for parent, labels in assignments.items():
        label = next(iter(labels))
        grouped[label] = grouped.get(label, 0) + 1
    return {
        "passed": True,
        "n_parent_simulations": len(assignments),
        "parent_counts_by_split": grouped,
        "warning": "Parent independence does not guarantee independence of cosmological parameter draws.",
    }


__all__ = [
    "NeutrinoSimulationRecord",
    "MassCoverageAudit",
    "audit_mass_coverage",
    "validate_field_payload",
    "validate_split_parent_independence",
]
