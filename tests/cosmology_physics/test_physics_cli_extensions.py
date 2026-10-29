"""CLI-level tests for scientific physics diagnostics introduced in v0.4.0."""

from __future__ import annotations

import json

import numpy as np

from nufield.cosmology.physics.cli import main
from nufield.cosmology.physics.growth_equation_validation import einstein_de_sitter_reference
from nufield.cosmology.physics.neutrino_mass_observables import minimum_mass_sum_ev


def test_cli_topology_writes_json(tmp_path):
    field = np.zeros((5, 5, 5), float)
    field[2, 2, 2] = 1
    field_path = tmp_path / "field.npy"
    np.save(field_path, field)
    output = tmp_path / "topology.json"
    code = main(
        [
            "topology",
            "--field",
            str(field_path),
            "--box-size",
            "100",
            "--threshold",
            "0.5",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text())
    assert payload["topology"]["volume_fraction"] == 1 / 125
    assert payload["topology"]["euler_characteristic"] == 1


def test_cli_sigma8_loads_physical_spectrum(tmp_path):
    k = np.geomspace(1e-4, 5.0, 512)
    p = 1e4 * k**0.96 / (1 + (k / 0.18) ** 2) ** 2
    path = tmp_path / "spectrum.npz"
    np.savez(path, k_mpc_inv=k, pk_mpc3=p, redshift=np.asarray(0.0), linear=np.asarray(True))
    output = tmp_path / "sigma8.json"
    code = main(
        [
            "sigma8",
            "--spectrum",
            str(path),
            "--n-k-values",
            "64",
            "128",
            "256",
            "--output",
            str(output),
        ]
    )
    # A coarse tabulated spectrum can fail a deliberately strict convergence threshold, but the report must exist.
    assert code in {0, 1}
    payload = json.loads(output.read_text())
    assert payload["sigma8"] > 0
    assert payload["convergence"]["n_k_values"] == [64, 128, 256]


def test_cli_neutrino_continuity_and_prior_commands(tmp_path):
    continuity = tmp_path / "continuity.json"
    code = main(
        [
            "neutrino-continuity",
            "--mass-ev",
            "0",
            "--z-max",
            "100",
            "--n-samples",
            "128",
            "--output",
            str(continuity),
        ]
    )
    assert code == 0
    assert json.loads(continuity.read_text())["passed"]
    floor = minimum_mass_sum_ev("normal")
    prior_out = tmp_path / "prior.json"
    assert (
        main(
            [
                "mass-prior",
                "--hierarchy",
                "normal",
                "--upper-sum-ev",
                str(floor + 0.1),
                "--n-samples",
                "500",
                "--seed",
                "3",
                "--output",
                str(prior_out),
            ]
        )
        == 0
    )
    prior_payload = json.loads(prior_out.read_text())
    assert prior_payload["sample_audit"]["support_valid"]


def test_cli_growth_check_and_gate_assess(tmp_path):
    data = einstein_de_sitter_reference(129, a_min=0.02)
    history = tmp_path / "growth.npz"
    np.savez(history, **data)
    growth_out = tmp_path / "growth-report.json"
    assert main(["growth-check", "--history", str(history), "--output", str(growth_out)]) == 0
    growth = json.loads(growth_out.read_text())
    assert growth["points_used"] > 100
    evidence = {
        name: {
            "passed": True,
            "severity": "info",
            "evidence_source": "synthetic test",
            "summary": "test pass",
        }
        for name in (
            "physics_equations",
            "solver_reference",
            "dataset_contract",
            "unit_conventions",
            "covariance_validation",
            "posterior_calibration",
            "heldout_systematics",
            "provenance",
        )
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    gate_out = tmp_path / "gate.json"
    # The CLI intentionally returns 0 only for ready_for_scientific_review; the gate here is all explicit passes.
    assert (
        main(
            [
                "gate-assess",
                "--evidence",
                str(evidence_path),
                "--campaign",
                "test",
                "--output",
                str(gate_out),
            ]
        )
        == 0
    )
    assert json.loads(gate_out.read_text())["status"] == "ready_for_scientific_review"
