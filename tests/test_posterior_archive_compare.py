import numpy as np
import pytest

from nufield.benchmarking.posterior_archive_compare import compare_posterior_archives


def _archive(path, seed, scale, order=None):
    rng = np.random.default_rng(seed)
    n = 24
    p = 2
    truth = np.column_stack([rng.uniform(0.1, 0.4, n), rng.uniform(0.25, 0.35, n)])
    # For the test helper, allow a separate deterministic common truth below.
    ids = np.array([f"s{i}" for i in range(n)])
    groups = np.array([f"g{i // 2}" for i in range(n)])
    samples = (
        truth[:, None, :]
        + rng.normal(0, scale, size=(n, 128, p)) * np.array([0.3, 0.1])[None, None, :]
    )
    if order is not None:
        truth = truth[order]
        ids = ids[order]
        groups = groups[order]
        samples = samples[order]
    np.savez_compressed(path, samples=samples, truth=truth, simulation_id=ids, group_id=groups)
    return truth, ids, groups


def test_comparison_requires_exact_matched_ids_and_groups(tmp_path):
    rng = np.random.default_rng(123)
    n = 24
    p = 2
    truth = np.column_stack([rng.uniform(0.1, 0.4, n), rng.uniform(0.25, 0.35, n)])
    ids = np.array([f"s{i}" for i in range(n)])
    groups = np.array([f"g{i // 2}" for i in range(n)])
    a = truth[:, None, :] + rng.normal(0, 0.06, (n, 128, p))
    b = truth[:, None, :] + rng.normal(0, 0.03, (n, 128, p))
    pa = tmp_path / "a.npz"
    pb = tmp_path / "b.npz"
    np.savez_compressed(pa, samples=a, truth=truth, simulation_id=ids, group_id=groups)
    order = np.arange(n)[::-1]
    np.savez_compressed(
        pb, samples=b[order], truth=truth[order], simulation_id=ids[order], group_id=groups[order]
    )
    report = compare_posterior_archives(
        pa,
        pb,
        parameter_names=["sum_mnu_ev", "omega_m"],
        prior_low=[0.1, 0.25],
        prior_high=[0.4, 0.35],
        n_bootstrap=100,
    )
    assert report["n_test_rows"] == n
    assert report["n_parent_groups"] == 12
    assert len(report["normalized_crps_mean_by_parameter_model_a"]) == 2
    assert report["scientific_approval"] is False


def test_comparison_rejects_different_test_simulations(tmp_path):
    truth, ids, groups = _archive(tmp_path / "a.npz", 3, 0.1)
    # Make B's set differ by changing one ID. Truth/order cannot be silently assumed equivalent.
    path = tmp_path / "b.npz"
    _archive(path, 4, 0.1)
    with np.load(path) as z:
        arr = {k: z[k] for k in z.files}
    arr["simulation_id"] = arr["simulation_id"].astype(str)
    arr["simulation_id"][0] = "other"
    np.savez_compressed(path, **arr)
    with pytest.raises(ValueError, match="exact same"):
        compare_posterior_archives(
            tmp_path / "a.npz",
            path,
            parameter_names=["a", "b"],
            prior_low=[0, 0],
            prior_high=[1, 1],
            n_bootstrap=100,
        )
