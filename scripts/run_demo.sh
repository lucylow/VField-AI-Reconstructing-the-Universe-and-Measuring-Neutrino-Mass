#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-outputs/demo}"
NSIMS="${NSIMS:-160}"
GRID="${GRID:-12}"
SEED="${SEED:-7}"
python -m nufield.cli demo --output-dir "$OUT" --n-sims "$NSIMS" --grid-size "$GRID" --seed "$SEED"
