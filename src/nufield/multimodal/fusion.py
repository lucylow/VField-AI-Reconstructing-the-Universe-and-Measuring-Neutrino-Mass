"""Mask-aware late fusion API for modality feature vectors."""

from __future__ import annotations

import numpy as np


def fuse_features(
    modalities: dict[str, np.ndarray | None], masks: dict[str, bool] | None = None
) -> tuple[np.ndarray, dict]:
    """Concatenate modality features with availability flags.

    All present vectors are flattened and finite-checked. This primitive does not learn a shared
    representation or account for cross-covariance; it is only a data-contract starter.
    """
    if not modalities:
        raise ValueError("at least one modality key is required")
    names = sorted(modalities)
    if masks is not None and set(masks) != set(names):
        raise ValueError("masks keys must exactly match modality keys")
    values = []
    status = {}
    for name in names:
        raw = modalities[name]
        available = raw is not None and (masks is None or bool(masks[name]))
        if available:
            arr = np.asarray(raw, dtype=np.float32).ravel()
            if not np.isfinite(arr).all():
                raise ValueError(f"modality {name} contains non-finite values")
        else:
            arr = np.zeros(0, dtype=np.float32)
        values.append(arr)
        status[name] = {"available": bool(available), "n_features": int(arr.size)}
    # Missing modalities are represented by no values plus an explicit availability bit. Since each
    # modality's dimensionality must be fixed for a production dataset, callers should pass zero-filled
    # vectors and mask=False when fixed dimensions are needed.
    fused = np.concatenate(
        [*values, np.asarray([status[n]["available"] for n in names], dtype=np.float32)]
    )
    return fused, {
        "modality_order": names,
        "status": status,
        "fusion": "concatenation_plus_availability_flags",
    }


def fixed_width_fuse(
    modalities: dict[str, np.ndarray | None],
    widths: dict[str, int],
    masks: dict[str, bool] | None = None,
) -> tuple[np.ndarray, dict]:
    """Concatenate fixed-width vectors; missing vectors are zero-filled and clearly masked."""
    if set(modalities) != set(widths):
        raise ValueError("modalities and widths must have identical keys")
    names = sorted(modalities)
    chunks, avail = [], []
    for name in names:
        width = int(widths[name])
        if width < 1:
            raise ValueError("each modality width must be positive")
        raw = modalities[name]
        is_available = raw is not None and (masks is None or bool(masks.get(name, True)))
        if is_available:
            arr = np.asarray(raw, dtype=np.float32).ravel()
            if arr.size != width or not np.isfinite(arr).all():
                raise ValueError(f"modality {name} must be finite and have width {width}")
        else:
            arr = np.zeros(width, dtype=np.float32)
        chunks.append(arr)
        avail.append(float(is_available))
    return np.concatenate(chunks + [np.asarray(avail, np.float32)]), {
        "modality_order": names,
        "widths": {k: int(widths[k]) for k in names},
        "availability": dict(zip(names, map(bool, avail))),
    }
