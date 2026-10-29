"""Redshift-drift observable for homogeneous cosmological backgrounds.

The velocity drift is the Sandage–Loeb signal, reported in cm s⁻¹ yr⁻¹:
``c [H0 - H(z)/(1+z)]`` with H converted from km s⁻¹ Mpc⁻¹ to yr⁻¹.
This background-only observable does not include peculiar accelerations or survey effects.
"""

from __future__ import annotations

import math

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.constants import C_KM_S

MPC_KM = 3.085677581491367e19
SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0


def redshift_drift_velocity_cm_s_per_year(background: BackgroundCosmology, z: float) -> float:
    """Return cosmological spectroscopic velocity drift in cm s⁻¹ yr⁻¹.

    Positive values indicate increasing observed redshift in the Sandage–Loeb convention.
    At z=0 the drift is exactly zero. Only the homogeneous expansion contribution is included.
    """
    if not math.isfinite(z) or z < 0:
        raise ValueError("z must be finite and non-negative")
    h0 = background.params.H0_km_s_Mpc
    hz = float(background.H(z))
    expansion_rate_per_year = (h0 - hz / (1.0 + z)) * SECONDS_PER_YEAR / MPC_KM
    return float(C_KM_S * 100.0 * expansion_rate_per_year)


__all__ = ["redshift_drift_velocity_cm_s_per_year"]
