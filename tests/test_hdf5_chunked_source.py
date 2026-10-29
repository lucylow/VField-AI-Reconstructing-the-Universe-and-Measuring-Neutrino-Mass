import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from nufield.data_registry.chunked_source import (
    ArrayContract,
    ChunkedArraySource,
    validate_aligned_arrays,
)
from nufield.data_registry.hdf5_profile import profile_hdf5
from nufield.data_registry.hdf5_profile_cli import main as profile_main


def _fixture(path):
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset(
            "fields",
            data=np.arange(60, dtype=np.float32).reshape(10, 2, 3),
            chunks=(2, 2, 3),
            compression="gzip",
        )
        ds.attrs["meaning"] = "test-only synthetic feature data"
        h5.create_dataset(
            "theta", data=np.column_stack([np.linspace(0.1, 0.4, 10), np.linspace(0.25, 0.35, 10)])
        )
        h5.create_dataset("group_id", data=np.asarray([f"g{i}".encode() for i in range(10)]))


def test_profile_hdf5_is_metadata_first_and_reports_chunks(tmp_path):
    f = tmp_path / "x.h5"
    _fixture(f)
    result = profile_hdf5(f)
    fields = next(row for row in result["objects"] if row["path"] == "/fields")
    assert fields["shape"] == [10, 2, 3]
    assert fields["chunks"] == [2, 2, 3]
    assert fields["compression"] == "gzip"
    assert result["interpretation_warning"]


def test_profile_cli_writes_json(tmp_path, capsys):
    f = tmp_path / "x.h5"
    _fixture(f)
    out = tmp_path / "profile.json"
    assert profile_main([str(f), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["object_count_returned"] == 3
    assert json.loads(capsys.readouterr().out)["scientific_semantics_inferred"] is False


def test_chunked_reader_batches_preserve_requested_index_order(tmp_path):
    f = tmp_path / "x.h5"
    _fixture(f)
    with ChunkedArraySource(f, kind="hdf5") as source:
        report = validate_aligned_arrays(
            source,
            [
                ArrayContract("fields", 3, (2, 3)),
                ArrayContract("theta", 2, (2,)),
                ArrayContract("group_id", 1, dtype_kind="string"),
            ],
        )
        assert report["row_aligned"]
        batches = list(
            source.iter_batches(
                "fields", batch_size=3, indices=np.array([8, 1, 5, 3, 9]), dtype="float64"
            )
        )
        idx = np.concatenate([b[0] for b in batches])
        values = np.concatenate([b[1] for b in batches])
    assert idx.tolist() == [8, 1, 5, 3, 9]
    np.testing.assert_array_equal(values[:, 0, 0], np.array([48, 6, 30, 18, 54], dtype=float))


def test_chunked_contract_reports_alignment_and_shape_errors(tmp_path):
    f = tmp_path / "x.h5"
    with h5py.File(f, "w") as h5:
        h5.create_dataset("x", data=np.ones((4, 2)))
        h5.create_dataset("theta", data=np.ones((5, 1)))
    with ChunkedArraySource(f) as src:
        report = validate_aligned_arrays(
            src, [ArrayContract("x", expected_rank=3), ArrayContract("theta", expected_rank=2)]
        )
    assert not report["row_aligned"]
    assert report["errors"]
    assert report["arrays"][0]["issues"]
