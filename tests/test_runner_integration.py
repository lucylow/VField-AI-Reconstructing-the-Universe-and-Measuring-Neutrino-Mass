import json

from nufield.config import SimulationConfig
from nufield.experiments.runner import run_benchmark
from nufield.simulation.dataset import create_dataset


def test_benchmark_writes_report(tmp_path):
    dataset = tmp_path / "dataset.npz"
    create_dataset(SimulationConfig(n_sims=36, grid_size=8, seed=3, n_k_bins=6), dataset)
    report = run_benchmark(dataset, tmp_path / "result", seed=4)
    loaded = json.loads((tmp_path / "result" / "benchmark_report.json").read_text())
    assert report["split_disjoint"] is True
    assert loaded["n_test"] > 0
    assert "scientific_interpretation" in loaded
