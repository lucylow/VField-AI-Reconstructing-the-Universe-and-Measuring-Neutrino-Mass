import numpy as np
import pytest

from nufield.data_registry.splits import (
    SplitError,
    assert_group_disjoint,
    stratified_group_train_val_test_split,
)
from nufield.data_registry.support import parameter_support_report


def test_stratified_group_split_has_mnu_bins_in_each_partition():
    n = 120
    groups = np.array([f"g{i}" for i in range(n)])
    mnu = np.linspace(0.06, 0.30, n)
    splits = stratified_group_train_val_test_split(groups, mnu, seed=1, n_bins=6)
    assert_group_disjoint(groups, splits)
    for idx in splits.values():
        assert np.ptp(mnu[idx]) > 0.1


def test_stratified_split_rejects_nonconstant_target_in_group():
    groups = np.array(["a", "a", "b", "c", "d", "e"])
    values = np.array([0.1, 0.2, 0.1, 0.2, 0.3, 0.4])
    with pytest.raises(SplitError):
        stratified_group_train_val_test_split(groups, values, require_group_constant=True)


def test_support_report_identifies_prior_support_mismatch():
    theta = np.column_stack([np.linspace(0.1, 0.2, 100), np.linspace(0.2, 0.4, 100)])
    result = parameter_support_report(
        theta, ["Mnu", "Omega_m"], prior_low=[0.06, 0.2], prior_high=[0.3, 0.4]
    )
    assert any(issue["code"] == "PRIOR_UNCOVERED_Mnu" for issue in result["issues"])
