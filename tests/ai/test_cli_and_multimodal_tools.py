import json

import numpy as np
import pytest

from nufield.ai.cli import main
from nufield.ai.multimodal_network import validate_modality_batch
from nufield.ai.symmetry import cubic_symmetry_tta
from nufield.ai.teacher_student import ensemble_teacher_targets


def test_multimodal_batch_contract():
    result = validate_modality_batch(
        {"galaxy": np.zeros((3, 8)), "lensing": np.zeros((3, 4))},
        [[1, 1], [1, 0], [0, 1]],
        ("galaxy", "lensing"),
    )
    assert result["valid"] and result["batch_size"] == 3
    with pytest.raises(ValueError):
        validate_modality_batch({"galaxy": np.zeros((3, 8))}, [[0], [1], [1]], ("galaxy",))


def test_teacher_ensemble_targets_include_epistemic_spread():
    means = np.array([[[0.0], [1.0]], [[2.0], [3.0]], [[1.0], [2.0]]])
    mean, var = ensemble_teacher_targets(means, np.ones_like(means) * 0.1)
    assert np.allclose(mean.ravel(), [1, 2])
    assert np.all(var > 0.1)


def test_cli_acquisition_writes_json(tmp_path):
    out = tmp_path / "acquire.json"
    code = main(
        [
            "acquire",
            "--low",
            "0",
            "0",
            "--high",
            "1",
            "2",
            "--n-candidates",
            "40",
            "--n-select",
            "5",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert len(json.loads(out.read_text())["selected"]) == 5


def test_cli_audit_dataset(tmp_path):
    from nufield.ai.synthetic import make_toy_inverse_problem

    ds = make_toy_inverse_problem(n_samples=80, seed=2)
    path = tmp_path / "toy.npz"
    ds.save(path)
    out = tmp_path / "audit.json"
    assert main(["audit-dataset", str(path), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["valid"]


def test_cli_demo_end_to_end_cpu(tmp_path):
    pytest.importorskip("torch")
    code = main(
        [
            "demo",
            "--output-dir",
            str(tmp_path / "demo"),
            "--n-samples",
            "120",
            "--epochs",
            "2",
            "--batch-size",
            "32",
            "--seed",
            "9",
        ]
    )
    assert code == 0
    result = json.loads((tmp_path / "demo" / "ai_demo_report.json").read_text())
    assert result["not_a_cosmology_result"] is True
    assert result["test_metrics"]["n"] > 0


def test_symmetry_tta_for_scalar_model():
    torch = pytest.importorskip("torch")
    from torch import nn

    model = nn.Sequential(nn.Flatten(), nn.Linear(4 * 4 * 4, 1))
    fields = torch.randn(3, 1, 4, 4, 4)
    result = cubic_symmetry_tta(model, fields, transforms=(0, 1, 2), batch_size=2)
    assert result["n_transforms"] == 3
    assert len(result["mean_prediction"]) == 3
