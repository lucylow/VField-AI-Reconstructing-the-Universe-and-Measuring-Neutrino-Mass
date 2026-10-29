"""Run νField's internal physics checks for the example parameter file.

Usage from the project root:
  PYTHONPATH=src python examples/physics/validate_reference_lcdm.py

Passing means equation-level internal checks passed; it is not a cosmological parameter fit and does
not validate the model against real survey data or an independent Boltzmann code.
"""
from __future__ import annotations

from pathlib import Path
from nufield.cosmology.physics.cli import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    raise SystemExit(main(["validate", "--config", str(root/"configs/physics/reference_lcdm.yaml"),
                           "--output", str(root/"outputs/physics/reference_validation.json")]))
