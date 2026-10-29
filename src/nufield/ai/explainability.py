"""Model explanation utilities for feature models and spatial fields.

Attribution indicates how a particular fitted model depends on inputs; it does not establish causal
physical mechanisms. Run explanations on held-out observations and compare stability across seeds,
model families, and small physically motivated input perturbations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nufield.exceptions import OptionalDependencyError

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _torch_required():
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] for gradient attribution")


def _select_scalar(output: Any, target_index: int = 0):
    if isinstance(output, dict):
        if "mean" in output:
            output = output["mean"]
        elif "prediction" in output:
            output = output["prediction"]
        else:
            raise ValueError("model output dictionary must contain mean or prediction")
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 1:
        if target_index != 0:
            raise ValueError("target_index out of range for scalar output")
        return output
    if output.ndim < 2 or not 0 <= target_index < output.shape[-1]:
        raise ValueError("target_index out of range")
    return output[..., target_index]


def input_gradient_saliency(
    model: Any, inputs: Any, *, target_index: int = 0, absolute: bool = True
) -> Any:
    """Gradient of the selected model output with respect to input features/field voxels."""
    _torch_required()
    was_training = model.training
    model.eval()
    x = inputs.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = _select_scalar(model(x), target_index)
    output.sum().backward()
    grad = x.grad.detach()
    if was_training:
        model.train()
    return grad.abs() if absolute else grad


def integrated_gradients(
    model: Any,
    inputs: Any,
    *,
    baseline: Any | None = None,
    target_index: int = 0,
    steps: int = 64,
    internal_batch_size: int = 16,
) -> Any:
    """Integrated gradients with a trapezoidal Riemann approximation.

    The baseline is a scientific choice: zero density contrast, a smooth field, and a mean summary
    vector answer different questions. Always save the chosen baseline with the explanation output.
    """
    _torch_required()
    if steps < 2 or internal_batch_size < 1:
        raise ValueError("steps must be >=2 and internal_batch_size positive")
    x = inputs.detach()
    b = (
        torch.zeros_like(x)
        if baseline is None
        else torch.as_tensor(baseline, dtype=x.dtype, device=x.device)
    )
    if b.shape != x.shape:
        try:
            b = torch.broadcast_to(b, x.shape)
        except RuntimeError as exc:
            raise ValueError("baseline must match or broadcast to inputs") from exc
    delta = x - b
    alphas = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)
    total_grad = torch.zeros_like(x)
    was_training = model.training
    model.eval()
    for start in range(0, steps, internal_batch_size):
        alpha = alphas[start : start + internal_batch_size]
        shape = (len(alpha),) + (1,) * x.ndim
        scaled = b.unsqueeze(0) + alpha.reshape(shape) * delta.unsqueeze(0)
        flat = scaled.reshape(-1, *x.shape[1:]).detach().requires_grad_(True)
        output = _select_scalar(model(flat), target_index)
        grad = torch.autograd.grad(output.sum(), flat)[0].reshape_as(scaled)
        weights = torch.ones(len(alpha), device=x.device, dtype=x.dtype)
        for i, a in enumerate(alpha):
            if torch.isclose(a, torch.tensor(0.0, device=x.device, dtype=x.dtype)) or torch.isclose(
                a, torch.tensor(1.0, device=x.device, dtype=x.dtype)
            ):
                weights[i] = 0.5
        total_grad += (grad * weights.reshape(shape)).sum(dim=0)
    if was_training:
        model.train()
    return delta * total_grad / (steps - 1)


def permutation_importance(
    features: Any,
    targets: Any,
    predict: Callable[[np.ndarray], np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_repeats: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute held-out feature permutation importance as metric degradation.

    The metric must be oriented so lower is better. This method can be misleading for correlated
    power-spectrum bins and should be complemented with grouped/physics-motivated ablations.
    """
    x, y = np.asarray(features, float), np.asarray(targets, float)
    if (
        x.ndim != 2
        or len(x) != len(y)
        or len(x) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("features and targets must be finite with aligned sample counts")
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    baseline_pred = np.asarray(predict(x))
    baseline = float(metric(y, baseline_pred))
    rng = np.random.default_rng(seed)
    result = []
    for col in range(x.shape[1]):
        scores = []
        for _ in range(n_repeats):
            permuted = x.copy()
            permuted[:, col] = permuted[rng.permutation(len(x)), col]
            scores.append(float(metric(y, np.asarray(predict(permuted)))))
        deltas = np.asarray(scores) - baseline
        result.append(
            {
                "feature_index": col,
                "importance_mean": float(deltas.mean()),
                "importance_std": float(deltas.std(ddof=1) if len(deltas) > 1 else 0),
                "permuted_scores": scores,
            }
        )
    return {
        "baseline_score": baseline,
        "features": sorted(result, key=lambda item: item["importance_mean"], reverse=True),
        "interpretation": "Importance is predictive reliance under this split, not a causal effect.",
    }


def modality_ablation_report(
    observed_by_setting: dict[str, Any],
    *,
    metric: Callable[[Any], float],
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Compare precomputed outputs under modality ablations using one frozen evaluation metric."""
    if "all_modalities" not in observed_by_setting:
        raise ValueError("an all_modalities reference output is required")
    scores = {key: float(metric(value)) for key, value in observed_by_setting.items()}
    base = scores["all_modalities"]
    delta = {key: value - base for key, value in scores.items() if key != "all_modalities"}
    if not lower_is_better:
        delta = {key: -value for key, value in delta.items()}
    return {
        "scores": scores,
        "degradation_vs_all_modalities": delta,
        "lower_is_better": lower_is_better,
        "warning": "Compare identical evaluation rows and do not tune on the test split.",
    }
