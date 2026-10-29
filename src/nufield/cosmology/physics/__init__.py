"""Physics-focused cosmology modules for explicit, testable calculations.

The primary public surface uses ``PhysicalCosmology`` rather than the legacy toy parameter schema.
Optional solver backends are imported by their adapter classes on construction, not at package import.
"""

from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.redshift_drift import redshift_drift_velocity_cm_s_per_year
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum, top_hat_window
from nufield.cosmology.physics.validity import PhysicsLevel, PhysicsScope, default_scope_for_level

__all__ = [
    "PhysicalCosmology",
    "BackgroundCosmology",
    "redshift_drift_velocity_cm_s_per_year",
    "MatterPowerSpectrum",
    "top_hat_window",
    "PhysicsLevel",
    "PhysicsScope",
    "default_scope_for_level",
]
