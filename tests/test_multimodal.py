import numpy as np
import pytest

from nufield.multimodal.fusion import fixed_width_fuse


def test_fixed_width_fusion_tracks_missing_modality():
    fused, meta = fixed_width_fuse(
        {"galaxy": np.ones(3), "lensing": None}, {"galaxy": 3, "lensing": 2}
    )
    assert fused.shape == (3 + 2 + 2,)
    assert meta["availability"] == {"galaxy": True, "lensing": False}
    assert (fused[3:5] == 0).all()


def test_fixed_width_rejects_mismatched_width():
    with pytest.raises(ValueError):
        fixed_width_fuse({"a": np.ones(2)}, {"a": 3})
