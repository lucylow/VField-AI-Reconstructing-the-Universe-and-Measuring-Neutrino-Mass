"""Paired-view representation learning helpers for cosmological fields and modalities."""

from __future__ import annotations

from typing import Any

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch, nn, F = None, None, None


def multi_positive_info_nce(
    view_a, view_b, group_ids, *, temperature: float = 0.1, normalize: bool = True
):
    """Supervised multi-positive InfoNCE across two paired representations.

    Embeddings with the same parent group are positives; examples from the same parent simulation
    are therefore not incorrectly treated as negatives. Inputs are [B,D] and group_ids has length B.
    Returns a scalar loss. This objective learns representations, not cosmological posteriors.
    """
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to use contrastive loss")
    if view_a.ndim != 2 or view_b.shape != view_a.shape:
        raise ValueError("view_a and view_b must share [B,D] shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = torch.as_tensor(group_ids, device=view_a.device).reshape(-1)
    batch = view_a.shape[0]
    if len(ids) != batch or batch < 2:
        raise ValueError("group_ids length must match batch and batch must be >=2")
    if normalize:
        a, b = F.normalize(view_a, dim=-1), F.normalize(view_b, dim=-1)
    else:
        a, b = view_a, view_b
    z = torch.cat([a, b], dim=0)
    zids = torch.cat([ids, ids], dim=0)
    logits = z @ z.T / temperature
    eye = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    positive = zids[:, None].eq(zids[None, :]) & ~eye
    valid = ~eye
    if torch.any(positive.sum(dim=1) < 1):
        raise ValueError("each anchor must have at least one positive paired view")
    # log probability mass assigned to all positives; denominator excludes the anchor itself.
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    log_denom = torch.logsumexp(logits, dim=1)
    log_positive = torch.logsumexp(
        logits.masked_fill(~positive, torch.finfo(logits.dtype).min), dim=1
    )
    return -(log_positive - log_denom).mean()


if nn is not None:

    class ProjectionHead(nn.Module):
        """Two-layer projection head typically discarded after representation pretraining."""

        def __init__(self, input_dim: int, projection_dim: int = 64, hidden_dim: int = 128):
            super().__init__()
            if min(input_dim, projection_dim, hidden_dim) < 1:
                raise ValueError("projection dimensions must be positive")
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, projection_dim)
            )

        def forward(self, x):
            return self.network(x)
else:

    class ProjectionHead:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError("Install nufield-starter[ml] to use ProjectionHead")


def validate_paired_group_ids(group_ids: Any, *, minimum_groups: int = 2) -> dict[str, int]:
    """Summarize grouped contrastive batches and flag degenerate group structure."""
    import numpy as np

    ids = np.asarray(group_ids).reshape(-1).astype(str)
    if len(ids) < 2:
        raise ValueError("at least two examples are required")
    unique, counts = np.unique(ids, return_counts=True)
    if len(unique) < minimum_groups:
        raise ValueError("too few distinct parent groups in contrastive batch")
    return {
        "batch_rows": int(len(ids)),
        "unique_groups": int(len(unique)),
        "max_rows_per_group": int(counts.max()),
        "singleton_group_fraction": float(np.mean(counts == 1)),
    }
