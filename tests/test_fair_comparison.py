from nufield.benchmarking.fair_comparison import audit_fair_comparison


def _record(model):
    return {
        "model_id": model,
        "dataset_id": "q-mnu",
        "dataset_release": "v1",
        "split_id": "fixed-split",
        "test_fingerprint": "abc123",
        "parameter_names": ["sum_mnu_ev", "omega_m"],
        "prior_low": [0.06, 0.25],
        "prior_high": [0.4, 0.4],
        "physics_level": "prototype",
    }


def test_fair_comparison_passes_equal_conditions():
    r = audit_fair_comparison([_record("mlp"), _record("cnn")])
    assert r.fair_comparison_possible


def test_fair_comparison_blocks_test_split_mismatch():
    b = _record("cnn")
    b["test_fingerprint"] = "different"
    r = audit_fair_comparison([_record("mlp"), b])
    assert not r.fair_comparison_possible
    assert any(f.field == "test_fingerprint" for f in r.findings)
