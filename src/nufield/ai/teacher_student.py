"""Gaussian predictive distillation for compressing ensemble inference models."""

from __future__ import annotations

from typing import Any

from nufield.ai.losses import gaussian_nll
from nufield.exceptions import OptionalDependencyError

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def gaussian_kl_teacher_to_student(
    teacher_mean, teacher_variance, student_mean, student_log_variance
):
    """KL(N_teacher || N_student), summed over independent target coordinates."""
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to use distillation losses")
    student_variance = torch.exp(student_log_variance)
    if not (
        teacher_mean.shape == teacher_variance.shape == student_mean.shape == student_variance.shape
    ):
        raise ValueError("teacher/student mean and variance tensors must share shape")
    if torch.any(teacher_variance <= 0) or not torch.isfinite(teacher_variance).all():
        raise ValueError("teacher variance must be finite and positive")
    term = (
        student_log_variance
        - torch.log(teacher_variance)
        + (teacher_variance + (teacher_mean - student_mean).square()) / student_variance
        - 1
    )
    return 0.5 * term.sum(dim=-1).mean()


def distillation_loss(
    student_output: dict[str, Any],
    batch_target: dict[str, Any],
    *,
    supervised_weight: float = 1.0,
    distillation_weight: float = 1.0,
    variance_floor: float = 1e-8,
):
    """Combine true-label Gaussian NLL with ensemble-teacher distribution matching.

    The batch target must contain ``target``, ``teacher_mean`` and ``teacher_variance`` generated
    from training examples only. Teacher outputs on validation/test data must not enter student fit.
    """
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to use distillation losses")
    if (
        supervised_weight < 0
        or distillation_weight < 0
        or supervised_weight + distillation_weight <= 0
    ):
        raise ValueError("loss weights must be nonnegative and not both zero")
    required = {"target", "teacher_mean", "teacher_variance"}
    if not required.issubset(batch_target):
        raise ValueError(f"batch target missing keys: {sorted(required - set(batch_target))}")
    sup = gaussian_nll(student_output, batch_target["target"])
    teacher_var = batch_target["teacher_variance"].clamp_min(variance_floor)
    kl = gaussian_kl_teacher_to_student(
        batch_target["teacher_mean"],
        teacher_var,
        student_output["mean"],
        student_output["log_variance"],
    )
    return supervised_weight * sup + distillation_weight * kl


def ensemble_teacher_targets(member_means: Any, member_variances: Any | None = None):
    """Create Gaussian moment-matched targets from teacher members.

    Inputs are NumPy arrays [M,N,P]. Total teacher variance includes model spread and average
    predicted variances when provided, so a student is not forced to copy only the ensemble mean.
    """
    import numpy as np

    means = np.asarray(member_means, float)
    if means.ndim != 3 or means.shape[0] < 2 or not np.isfinite(means).all():
        raise ValueError("member_means must be finite [M>=2,N,P]")
    mean = means.mean(axis=0)
    epistemic = means.var(axis=0, ddof=1)
    if member_variances is None:
        aleatoric = np.zeros_like(epistemic)
    else:
        variances = np.asarray(member_variances, float)
        if (
            variances.shape != means.shape
            or not np.isfinite(variances).all()
            or np.any(variances < 0)
        ):
            raise ValueError("member_variances must match means and be nonnegative")
        aleatoric = variances.mean(axis=0)
    return mean, epistemic + aleatoric
