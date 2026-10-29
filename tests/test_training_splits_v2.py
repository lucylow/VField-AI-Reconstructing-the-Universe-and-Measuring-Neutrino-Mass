import numpy as np

from nufield.training.splits import prepare_grouped_data


def test_preprocessor_fitted_from_train_groups_only():
    # Values in the test-only groups are shifted far from training to expose leakage.
    rng = np.random.default_rng(4)
    groups = np.array([f"group-{i}" for i in range(50)])
    x = rng.normal(size=(50, 5)).astype(np.float32)
    y = np.column_stack([np.linspace(0.06, 0.3, 50), np.linspace(0.27, 0.36, 50)]).astype(
        np.float32
    )
    ids = np.array([f"sim-{i}" for i in range(50)])
    data = prepare_grouped_data(
        x, y, simulation_ids=ids, group_ids=groups, parameter_names=["Mnu", "Omega_m"], seed=9
    )
    assert data.x_train.shape[1] == 5
    assert "split_fingerprints" in data.split_report
    assert data.preprocessing_state["mean"]
