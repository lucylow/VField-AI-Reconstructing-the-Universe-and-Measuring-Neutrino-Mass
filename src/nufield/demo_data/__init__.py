"""Synthetic demo-data toolkit for NUFIELD frontend development.

The package exists to make UI demonstrations repeatable without presenting
fabricated values as research results. It deliberately avoids invoking the
project's scientific simulation and inference modules.
"""

from .bundle import build_demo_bundle, write_demo_bundle
from .profiles import DEMO_SCENARIOS, get_scenario

__all__ = ["DEMO_SCENARIOS", "build_demo_bundle", "get_scenario", "write_demo_bundle"]
