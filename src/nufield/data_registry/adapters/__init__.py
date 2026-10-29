"""Adapters describe source-specific requirements; they do not pretend files share one schema."""

from nufield.data_registry.adapters.desi import DESICatalogSpec, validate_desi_spec
from nufield.data_registry.adapters.quijote import QuijoteProductSpec, validate_quijote_spec

__all__ = ["QuijoteProductSpec", "validate_quijote_spec", "DESICatalogSpec", "validate_desi_spec"]
