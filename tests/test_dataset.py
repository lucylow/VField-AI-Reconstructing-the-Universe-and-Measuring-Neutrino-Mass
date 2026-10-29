from nufield.config import SimulationConfig, SplitConfig
from nufield.simulation.dataset import create_dataset, load_dataset, make_split


def test_dataset_roundtrip_and_split(tmp_path):
    path = tmp_path / "d.npz"
    create_dataset(SimulationConfig(n_sims=30, grid_size=8, seed=9, n_k_bins=6), path)
    data = load_dataset(path)
    split = make_split(data["simulation_id"], SplitConfig(), 88)
    assert len(data["features"]) == 30
    assert len(split["train"]) + len(split["validation"]) + len(split["test"]) == 30
    assert not set(split["train"]) & set(split["test"])
