#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[dev]'
pytest
python -m nufield.cli demo --output-dir outputs/smoke --n-sims 80 --grid-size 8 --seed 21
python -m nufield.cli inspect outputs/smoke/dataset.npz
