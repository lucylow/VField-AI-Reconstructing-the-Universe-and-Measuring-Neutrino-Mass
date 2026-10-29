"""Cosmology backend interfaces.

Only an optional CLASS adapter is provided. The default simulator remains explicitly toy-level.
"""

from nufield.cosmology.backend import BackendMetadata, LinearPowerBackend

__all__ = ["LinearPowerBackend", "BackendMetadata"]
