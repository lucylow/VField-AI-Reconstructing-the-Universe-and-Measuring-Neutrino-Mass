"""Light-cone shell bookkeeping for snapshot-based cosmology simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.distances import radial_comoving_distance_mpc


@dataclass(frozen=True, slots=True)
class Snapshot:
    redshift: float
    scale_factor: float
    path: str
    simulation_id: str

    def __post_init__(self) -> None:
        if self.redshift < 0 or not np.isfinite(self.redshift):
            raise ValueError("snapshot redshift must be finite and non-negative")
        if not np.isclose(self.scale_factor, 1 / (1 + self.redshift), rtol=1e-5, atol=1e-7):
            raise ValueError("scale_factor is inconsistent with redshift")
        if not self.path or not self.simulation_id:
            raise ValueError("snapshot path and simulation_id must be nonempty")


@dataclass(frozen=True, slots=True)
class LightconeShell:
    z_min: float
    z_max: float
    chi_min_mpc: float
    chi_max_mpc: float
    snapshot_id: str


def validate_snapshot_sequence(snapshots: list[Snapshot] | tuple[Snapshot, ...]) -> list[Snapshot]:
    """Return snapshots sorted by redshift and require unique IDs/redshifts."""
    if not snapshots:
        raise ValueError("at least one snapshot is required")
    ordered = sorted(snapshots, key=lambda s: s.redshift)
    if len({s.simulation_id for s in ordered}) != len(ordered):
        raise ValueError("simulation_id must be unique for each snapshot entry")
    zs = [s.redshift for s in ordered]
    if len(set(zs)) != len(zs):
        raise ValueError("duplicate snapshot redshifts are not permitted")
    return ordered


def assign_lightcone_shells(
    background: BackgroundCosmology,
    snapshots: list[Snapshot] | tuple[Snapshot, ...],
    *,
    z_min: float,
    z_max: float,
) -> list[LightconeShell]:
    """Split a redshift range at midpoints between supplied snapshots in comoving distance.

    Each shell uses the closest snapshot in comoving distance to its center. This is bookkeeping only;
    it does not evolve particle positions between snapshots, apply periodic replicas, or construct a
    survey light cone with angular masks.
    """
    if z_min < 0 or z_max <= z_min:
        raise ValueError("require 0<=z_min<z_max")
    snaps = validate_snapshot_sequence(snapshots)
    zvals = np.asarray([s.redshift for s in snaps])
    zlo, zhi = max(z_min, float(zvals.min())), min(z_max, float(zvals.max()))
    if zhi <= zlo:
        raise ValueError("requested redshift range does not intersect snapshot coverage")
    chi_nodes = np.array([radial_comoving_distance_mpc(background, float(z)) for z in zvals])
    chi_lo = radial_comoving_distance_mpc(background, zlo)
    chi_hi = radial_comoving_distance_mpc(background, zhi)
    mid = 0.5 * (chi_nodes[:-1] + chi_nodes[1:])
    boundaries = np.concatenate(([chi_lo], mid[(mid > chi_lo) & (mid < chi_hi)], [chi_hi]))
    shells: list[LightconeShell] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        center = 0.5 * (left + right)
        idx = int(np.argmin(np.abs(chi_nodes - center)))
        # Convert shell limits back to redshift by interpolating the monotone snapshot distance table.
        # This is only exact at the nodes; for more accuracy use the background distance inverse solver.
        from nufield.cosmology.physics.distances import redshift_at_comoving_distance

        zleft = redshift_at_comoving_distance(background, float(left), z_bracket=(zlo, zhi))
        zright = redshift_at_comoving_distance(background, float(right), z_bracket=(zlo, zhi))
        shells.append(
            LightconeShell(
                float(zleft), float(zright), float(left), float(right), snaps[idx].simulation_id
            )
        )
    return shells


__all__ = ["Snapshot", "LightconeShell", "validate_snapshot_sequence", "assign_lightcone_shells"]
