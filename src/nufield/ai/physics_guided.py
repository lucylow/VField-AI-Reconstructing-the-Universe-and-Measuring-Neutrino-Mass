"""Differentiable physics-guided objectives for neural field reconstruction.

These utilities are generic penalties, not authoritative likelihoods. They should be enabled only
when the observable and physical constraint are justified for a specific experiment. Overweighted
regularization can suppress a real signal or force a model toward an incorrect approximation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch, F = None, None


def _require_torch():
    if torch is None:
        raise OptionalDependencyError(
            "Install nufield-starter[ml] to use differentiable physics objectives"
        )


def box_constraint_penalty(prediction, lower, upper, *, reduction: str = "mean"):
    """Squared penalty for predictions outside a declared parameter box.

    A bounded output transform is usually preferable. This penalty is most useful as an independent
    diagnostic or when targets are intentionally allowed to lie near the box edges.
    """
    _require_torch()
    lo = torch.as_tensor(lower, dtype=prediction.dtype, device=prediction.device)
    hi = torch.as_tensor(upper, dtype=prediction.dtype, device=prediction.device)
    if torch.any(hi <= lo):
        raise ValueError("upper bounds must exceed lower bounds")
    violation = F.relu(lo - prediction).square() + F.relu(prediction - hi).square()
    if reduction == "none":
        return violation
    if reduction == "sum":
        return violation.sum()
    if reduction == "mean":
        return violation.mean()
    raise ValueError("reduction must be none/sum/mean")


def smoothness_penalty(field, *, periodic: bool = False, order: int = 1, reduction: str = "mean"):
    """Finite-difference smoothness penalty for predictions on a regular grid.

    Do not use this as a generic loss for nonlinear cosmological density fields: physical structure
    can legitimately be sharp. It is intended for denoising/reconstruction baselines with an explicit
    smoothness prior and should be ablated against the unregularized model.
    """
    _require_torch()
    if field.ndim not in {4, 5}:
        raise ValueError("field must be [B,D,H,W] or [B,C,D,H,W]")
    if order not in {1, 2}:
        raise ValueError("order must be 1 or 2")
    if periodic:
        diffs = []
        for axis in range(field.ndim - 3, field.ndim):
            first = torch.roll(field, shifts=-1, dims=axis) - field
            diff = first if order == 1 else torch.roll(first, shifts=-1, dims=axis) - first
            diffs.append(diff.square().mean())
        value = torch.stack(diffs).mean()
    else:
        diffs = []
        for axis in range(field.ndim - 3, field.ndim):
            first = torch.diff(field, dim=axis)
            diff = first if order == 1 else torch.diff(first, dim=axis)
            diffs.append(diff.square().mean())
        value = torch.stack(diffs).mean()
    if reduction == "none":
        return torch.stack(diffs)
    if reduction == "sum":
        return torch.stack(diffs).sum()
    if reduction == "mean":
        return value
    raise ValueError("reduction must be none/sum/mean")


def rotation_invariance_loss(
    model: Callable, fields, *, axes: tuple[int, int] = (-1, -2), target_kind: str = "scalar"
):
    """Penalize output changes under a 90-degree rotation of an isotropic periodic field.

    Use only for scalar parameters whose likelihood is expected to be invariant under the chosen
    cube symmetry. For field-valued outputs, rotate the output back before comparison.
    """
    _require_torch()
    if fields.ndim != 5:
        raise ValueError("fields must be [B,C,D,H,W]")
    if target_kind not in {"scalar", "field"}:
        raise ValueError("target_kind must be scalar or field")
    a, b = axes
    rotated_input = torch.rot90(fields, k=1, dims=(a, b))
    original = model(fields)
    transformed = model(rotated_input)
    if isinstance(original, dict) or isinstance(transformed, dict):
        raise ValueError("rotation_invariance_loss expects tensor outputs")
    if target_kind == "field":
        transformed = torch.rot90(transformed, k=-1, dims=(a, b))
    if original.shape != transformed.shape:
        raise ValueError("model outputs must preserve shape under the selected transform")
    return F.mse_loss(original, transformed)


def equivariance_consistency_loss(
    model: Callable,
    fields,
    *,
    transform: str = "roll",
    shift: tuple[int, int, int] = (1, 2, 3),
    target_kind: str = "scalar",
):
    """Consistency loss under periodic translations or cube rotations.

    ``scalar`` tasks compare invariant model outputs. ``field`` tasks inverse-transform the prediction
    before comparison. This does not enforce all cosmological symmetries; it tests only the transform
    explicitly requested here.
    """
    _require_torch()
    if fields.ndim != 5:
        raise ValueError("fields must be [B,C,D,H,W]")
    if target_kind not in {"scalar", "field"}:
        raise ValueError("target_kind must be scalar or field")
    if transform == "roll":
        transformed_fields = torch.roll(fields, shifts=shift, dims=(-3, -2, -1))
    elif transform == "rotate_xy":
        transformed_fields = torch.rot90(fields, k=1, dims=(-2, -1))
    else:
        raise ValueError("transform must be roll or rotate_xy")
    out = model(fields)
    aug = model(transformed_fields)
    if isinstance(out, dict) or isinstance(aug, dict):
        raise ValueError("model outputs must be tensors for consistency loss")
    if target_kind == "field":
        if transform == "roll":
            aug = torch.roll(aug, shifts=tuple(-int(v) for v in shift), dims=(-3, -2, -1))
        else:
            aug = torch.rot90(aug, k=-1, dims=(-2, -1))
    if out.shape != aug.shape:
        raise ValueError("outputs must have matching shape")
    return F.mse_loss(out, aug)


def composite_objective(data_loss, *, regularizers: dict[str, tuple[Any, float]]):
    """Combine scalar losses with explicit nonnegative weights and return term-level diagnostics."""
    _require_torch()
    total = data_loss
    terms = {"data_loss": data_loss.detach()}
    for name, (loss, weight) in regularizers.items():
        if weight < 0:
            raise ValueError(f"regularizer {name} has a negative weight")
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError(f"regularizer {name} must be a scalar tensor")
        total = total + float(weight) * loss
        terms[name] = loss.detach()
    terms["total"] = total.detach()
    return total, terms
