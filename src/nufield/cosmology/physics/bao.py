"""BAO distance variables and Alcock--Paczynski dilation parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.constants import C_KM_S
from nufield.cosmology.physics.distances import (
    hubble_distance_mpc,
    transverse_comoving_distance_mpc,
)


@dataclass(frozen=True, slots=True)
class BAODistances:
    z: float
    D_M_over_rdrag: float
    D_H_over_rdrag: float
    D_V_over_rdrag: float
    D_M_Mpc: float
    D_H_Mpc: float
    D_V_Mpc: float
    r_drag_Mpc: float


def bao_distances(background: BackgroundCosmology, z: float, r_drag_mpc: float) -> BAODistances:
    """Compute common BAO distance combinations; r_drag should come from a validated solver."""
    if z <= 0 or r_drag_mpc <= 0 or not math.isfinite(z + r_drag_mpc):
        raise ValueError("z and r_drag_mpc must be finite and positive")
    dm = transverse_comoving_distance_mpc(background, z)
    dh = hubble_distance_mpc(background, z)
    dv = (z * dh * dm**2) ** (1.0 / 3.0)
    return BAODistances(
        z, dm / r_drag_mpc, dh / r_drag_mpc, dv / r_drag_mpc, dm, dh, dv, r_drag_mpc
    )


def bao_dilation_parameters(
    *,
    D_M_mpc: float,
    H_km_s_mpc: float,
    r_drag_mpc: float,
    D_M_fid_mpc: float,
    H_fid_km_s_mpc: float,
    r_drag_fid_mpc: float,
) -> tuple[float, float]:
    r"""Return (alpha_perp, alpha_parallel) relative to a fiducial cosmology.

    alpha_perp = (D_M/r_d)/(D_M,fid/r_d,fid); alpha_parallel = (D_H/r_d)/(D_H,fid/r_d,fid).
    """
    vals = [D_M_mpc, H_km_s_mpc, r_drag_mpc, D_M_fid_mpc, H_fid_km_s_mpc, r_drag_fid_mpc]
    if any(not math.isfinite(v) or v <= 0 for v in vals):
        raise ValueError(
            "all distance, Hubble, and sound-horizon values must be finite and positive"
        )
    alpha_perp = (D_M_mpc / r_drag_mpc) / (D_M_fid_mpc / r_drag_fid_mpc)
    alpha_parallel = (C_KM_S / H_km_s_mpc / r_drag_mpc) / (C_KM_S / H_fid_km_s_mpc / r_drag_fid_mpc)
    return float(alpha_perp), float(alpha_parallel)


def isotropic_bao_alpha(
    D_V_mpc: float, r_drag_mpc: float, D_V_fid_mpc: float, r_drag_fid_mpc: float
) -> float:
    vals = [D_V_mpc, r_drag_mpc, D_V_fid_mpc, r_drag_fid_mpc]
    if any(v <= 0 or not math.isfinite(v) for v in vals):
        raise ValueError("all distance and sound-horizon values must be finite and positive")
    return float((D_V_mpc / r_drag_mpc) / (D_V_fid_mpc / r_drag_fid_mpc))


__all__ = ["BAODistances", "bao_distances", "bao_dilation_parameters", "isotropic_bao_alpha"]
