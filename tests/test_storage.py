import numpy as np
import pytest

from nufield.storage.array_store import NumpyArtifactStore


def test_artifact_store_roundtrip_and_hash(tmp_path):
    store = NumpyArtifactStore(tmp_path)
    info = store.save("fixture", {"x": np.arange(4)}, {"physics_level": "toy"})
    arrays, metadata = store.load("fixture")
    assert arrays["x"].tolist() == [0, 1, 2, 3]
    assert metadata["sha256"] == info["sha256"]


def test_artifact_store_rejects_object_arrays(tmp_path):
    store = NumpyArtifactStore(tmp_path)
    with pytest.raises(ValueError):
        store.save("bad", {"x": np.asarray([object()], dtype=object)}, {})
