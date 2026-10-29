"""Symmetry-aware test-time augmentation for periodic box inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from nufield.ai.uncertainty import decompose_ensemble
from nufield.exceptions import OptionalDependencyError

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def cubic_symmetry_tta(
    model: Any,
    fields: Any,
    *,
    transforms: tuple[int, ...] = (0, 1, 2, 3),
    batch_size: int = 32,
    device: str = "cpu",
    output_kind: str = "scalar",
) -> dict[str, Any]:
    """Average predictions over selected cube symmetries and quantify TTA sensitivity.

    The transform index uses the same 48 signed permutations as ``augmentation.cube_symmetry_transforms``.
    For scalar inference, outputs are expected to be invariant and can be directly aggregated. For
    field output, each prediction is inverse-transformed before aggregation. TTA spread measures model
    sensitivity to these transforms; it is not a calibrated predictive variance.
    """
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to use symmetry TTA")
    from nufield.ai.augmentation import cube_symmetry_transforms

    x = fields if torch.is_tensor(fields) else torch.as_tensor(fields, dtype=torch.float32)
    if (
        x.ndim != 5
        or batch_size < 1
        or len(transforms) < 2
        or output_kind not in {"scalar", "field"}
    ):
        raise ValueError("fields must be [B,C,D,H,W]; valid transforms/batch/output_kind required")
    if any(i < 0 or i >= 48 for i in transforms):
        raise ValueError("transform indices must be in [0,47]")
    trans_list = cube_symmetry_transforms()
    predictions = []
    model.eval()
    with torch.no_grad():
        for t_index in transforms:
            perm, flip = trans_list[t_index]
            dims = tuple(range(x.ndim - 3)) + tuple(x.ndim - 3 + p for p in perm)
            transformed = x.permute(dims)
            flip_dims = tuple(x.ndim - 3 + i for i, flag in enumerate(flip) if flag)
            if flip_dims:
                transformed = torch.flip(transformed, dims=flip_dims)
            chunks = []
            for start in range(0, len(x), batch_size):
                out = model(transformed[start : start + batch_size].to(device))
                if isinstance(out, dict):
                    out = out.get("mean", out.get("prediction"))
                    if out is None:
                        raise ValueError("model dictionary output needs mean/prediction")
                chunks.append(out.detach().cpu())
            pred = torch.cat(chunks, dim=0)
            if output_kind == "field":
                # Inverse signed permutation: undo flips first, then transpose inverse axes.
                if flip_dims:
                    pred = torch.flip(pred, dims=flip_dims)
                inverse = np.argsort(perm).tolist()
                pred = pred.permute(
                    tuple(range(pred.ndim - 3)) + tuple(pred.ndim - 3 + i for i in inverse)
                )
            predictions.append(pred.numpy())
    stack = np.stack(predictions, axis=0)
    result = decompose_ensemble(
        stack if stack.shape[0] >= 2 else np.concatenate([stack, stack], axis=0)
    )
    return {
        "mean_prediction": result.mean.tolist(),
        "transform_sensitivity_variance": result.epistemic_variance.tolist(),
        "n_transforms": len(transforms),
        "transform_indices": list(transforms),
        "output_kind": output_kind,
        "warning": "Transform sensitivity is not a calibrated measurement uncertainty.",
    }
