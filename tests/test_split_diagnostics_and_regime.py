import numpy as np

from nufield.data_registry.split_diagnostics import diagnose_split_support
from nufield.data_registry.splits import group_train_val_test_split
from nufield.models.data_regime import characterize_regime, model_capacity_advice


def test_split_diagnostics_records_support_and_group_counts():
    rng = np.random.default_rng(10)
    n = 90
    groups = np.array([f"g{i}" for i in range(n)])
    theta = np.column_stack([rng.uniform(0.1, 0.4, n), rng.uniform(0.2, 0.4, n)])
    split = group_train_val_test_split(groups, seed=9)
    r = diagnose_split_support(theta, ["sum_mnu_ev", "omega_m"], groups, split)
    assert r["group_disjoint"] and set(r["split_summary"]) == {"train", "validation", "test"}


def test_model_capacity_advice_blocks_unverified_fno_and_flags_toy_physics():
    x = np.random.default_rng(1).normal(size=(80, 2))
    y = np.column_stack([np.tile([0.1, 0.2, 0.3, 0.4], 20), np.arange(80) / 1000 + 0.3])
    ids = np.array([f"g{i}" for i in range(len(y))])
    regime = characterize_regime(
        theta=y,
        parameter_names=["sum_mnu_ev", "omega_m"],
        group_ids=ids,
        observation_shape=(1, 24, 24, 24),
        physics_level="toy",
        periodic_regular_grid=False,
    )
    r = model_capacity_advice(regime, budget="workstation")
    fno = next(m for m in r["models"] if m["model_id"] == "fno3d_npe")
    assert not fno["eligible_for_benchmark"]
    assert r["warnings"]
