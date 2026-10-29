"""Programmatic end-to-end start for νField.

Run from the repository root after `pip install -e '.[dev]'`:
    python examples/first_run.py
"""
from pathlib import Path
from nufield.experiments.runner import run_demo

if __name__ == "__main__":
    result = run_demo(Path("outputs/example_run"), n_sims=80, grid_size=8, seed=17)
    print(result)
