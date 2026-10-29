import pytest

from nufield.validation.leakage import assert_disjoint_splits, split_hash


def test_disjoint_splits_pass_and_hash_stable():
    assert_disjoint_splits(["a"], ["b"], ["c"])
    assert split_hash(["b", "a"], ["c"], ["d"]) == split_hash(["a", "b"], ["c"], ["d"])


def test_overlap_is_reported():
    with pytest.raises(ValueError):
        assert_disjoint_splits(["a"], ["a"], ["b"])
