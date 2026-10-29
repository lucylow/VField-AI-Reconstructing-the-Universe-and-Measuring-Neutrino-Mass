import numpy as np
import pytest

from nufield.data_registry.npz_profile import profile_npz
from nufield.training.npe_workflow import assert_unique_parent_groups_for_sbi_training


def test_npz_profile_hash_and_shapes(tmp_path):
    x = tmp_path / "arr.npz"
    np.savez(x, features=np.arange(15).reshape(5, 3), theta=np.ones((5, 1)))
    r = profile_npz(x, inspect_values=True)
    assert len(r["archive_sha256"]) == 64
    assert r["arrays"][0]["shape"] == [5, 3]


def test_sbi_row_level_validation_refuses_repeated_groups():
    groups = np.array(["a", "a", "b", "c", "d"])
    with pytest.raises(ValueError, match="row-based"):
        assert_unique_parent_groups_for_sbi_training(groups, np.array([0, 1, 2, 3]))
    assert_unique_parent_groups_for_sbi_training(groups, np.array([0, 2, 3, 4]))
