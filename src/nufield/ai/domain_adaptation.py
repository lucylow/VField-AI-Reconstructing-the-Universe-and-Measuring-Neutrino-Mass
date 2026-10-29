"""Optional domain-adversarial components for simulation-to-survey representation studies.

Domain adaptation is a stress-test tool, not a guarantee of removing simulator mismatch. A domain
classifier can also erase cosmological signal if the domain label is correlated with parameters.
Use only with scientifically justified domains and report parameter recovery before/after adaptation.
"""

from __future__ import annotations

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
    from torch.autograd import Function
except ImportError:  # pragma: no cover
    torch, nn, Function = None, None, None


def _need_torch():
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] for domain adaptation")


if nn is not None:

    class _GradientReverse(Function):
        @staticmethod
        def forward(ctx, inputs, strength):
            ctx.strength = float(strength)
            return inputs.view_as(inputs)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.strength * grad_output, None

    def gradient_reverse(x, strength: float = 1.0):
        """Identity forward pass and sign-reversed scaled gradient."""
        if strength < 0:
            raise ValueError("strength must be nonnegative")
        return _GradientReverse.apply(x, strength)

    class DomainAdversarialHead(nn.Module):
        """Train a domain discriminator through a gradient-reversal layer."""

        def __init__(self, embedding_dim: int, n_domains: int = 2, hidden_dim: int = 64):
            super().__init__()
            if min(embedding_dim, n_domains, hidden_dim) < 1:
                raise ValueError("dimensions must be positive")
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_domains)
            )

        def forward(self, embedding, strength: float = 1.0):
            return self.classifier(gradient_reverse(embedding, strength))

    class DomainAdversarialModel(nn.Module):
        """Shared encoder with a task head and a domain head."""

        def __init__(
            self,
            encoder: nn.Module,
            embedding_dim: int,
            target_dim: int,
            n_domains: int = 2,
            hidden_dim: int = 64,
        ):
            super().__init__()
            if min(embedding_dim, target_dim) < 1:
                raise ValueError("dimensions must be positive")
            self.encoder = encoder
            self.task_head = nn.Linear(embedding_dim, target_dim)
            self.domain_head = DomainAdversarialHead(embedding_dim, n_domains, hidden_dim)

        def forward(self, x, *, reversal_strength: float = 1.0):
            embedding = self.encoder(x)
            return {
                "prediction": self.task_head(embedding),
                "domain_logits": self.domain_head(embedding, reversal_strength),
                "embedding": embedding,
            }
else:

    def gradient_reverse(*args, **kwargs):
        _need_torch()

    class DomainAdversarialHead:
        def __init__(self, *args, **kwargs):
            _need_torch()

    class DomainAdversarialModel:
        def __init__(self, *args, **kwargs):
            _need_torch()


def reversal_strength_schedule(
    step: int, total_steps: int, *, maximum: float = 1.0, gamma: float = 10.0
) -> float:
    """Smooth DANN strength ramp from zero to maximum using a logistic schedule."""
    if step < 0 or total_steps < 1 or gamma <= 0 or maximum < 0:
        raise ValueError("invalid schedule parameters")
    p = min(step / max(total_steps - 1, 1), 1.0)
    return float(maximum * (2 / (1 + __import__("math").exp(-gamma * p)) - 1))


def domain_balance_report(domain_labels, parameter_targets) -> dict:
    """Measure target differences across domain groups before training a domain-adversarial model."""
    import numpy as np

    domains = np.asarray(domain_labels).reshape(-1)
    theta = np.asarray(parameter_targets, float)
    if theta.ndim == 1:
        theta = theta[:, None]
    if theta.ndim != 2 or len(theta) != len(domains) or not np.isfinite(theta).all():
        raise ValueError("parameter_targets must be finite [N,P] and align with domain_labels")
    groups = np.unique(domains)
    if len(groups) < 2:
        raise ValueError("at least two domains are required")
    means = {str(g): theta[domains == g].mean(axis=0).tolist() for g in groups}
    return {
        "n_domains": int(len(groups)),
        "counts": {str(g): int(np.sum(domains == g)) for g in groups},
        "target_means_by_domain": means,
        "warning": "Target/domain confounding can make domain invariance remove scientifically relevant signal.",
    }
