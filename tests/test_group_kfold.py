import numpy as np

from nufield.training.cross_validation import cross_validation_manifest, group_kfold_indices


def test_group_kfold_keeps_parent_groups_together_and_covers_all_rows():
    groups = np.repeat([f"g{i}" for i in range(15)], 2)
    target = np.repeat(np.linspace(0.05, 0.4, 15), 2)
    folds = group_kfold_indices(groups, n_splits=5, seed=13, stratify_values=target)
    assert len(folds) == 5
    tests = np.concatenate([fold.test_indices for fold in folds])
    assert sorted(tests.tolist()) == list(range(len(groups)))
    for fold in folds:
        assert not (set(groups[fold.train_indices]) & set(groups[fold.test_indices]))
    manifest = cross_validation_manifest(folds, dataset_id="toy", target_name="sum_mnu_ev", seed=13)
    assert manifest["n_folds"] == 5
