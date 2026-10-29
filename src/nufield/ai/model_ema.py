"""Exponential moving average for stabilizing neural model evaluation."""

from __future__ import annotations

import copy
from typing import Any

from nufield.exceptions import OptionalDependencyError

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class ModelEMA:
    """Maintain shadow parameters as an exponential moving average of a PyTorch model.

    This helper is independent of the core trainer so experiment owners can compare raw and EMA
    checkpoints without silently changing the training recipe. Always choose EMA settings on
    validation groups and report the comparison.
    """

    def __init__(self, model: Any, *, decay: float = 0.999, warmup_steps: int = 0):
        if torch is None:
            raise OptionalDependencyError("Install nufield-starter[ml] to use ModelEMA")
        if not 0 <= decay < 1 or warmup_steps < 0:
            raise ValueError("decay must be in [0,1) and warmup_steps nonnegative")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.steps = 0
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    def update(self, model: Any) -> None:
        if torch is None:
            raise OptionalDependencyError("PyTorch is required")
        self.steps += 1
        effective_decay = (
            min(self.decay, (1 + self.steps) / (10 + self.steps))
            if self.steps <= self.warmup_steps
            else self.decay
        )
        source = model.state_dict()
        with torch.no_grad():
            for name, shadow_value in self.shadow.state_dict().items():
                value = source[name].detach().to(device=shadow_value.device)
                if torch.is_floating_point(shadow_value):
                    shadow_value.mul_(effective_decay).add_(value, alpha=1 - effective_decay)
                else:
                    shadow_value.copy_(value)

    def copy_to(self, model: Any) -> Any:
        model.load_state_dict(self.shadow.state_dict())
        return model

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "steps": self.steps,
            "shadow_state_dict": self.shadow.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["decay"]) != self.decay or int(state["warmup_steps"]) != self.warmup_steps:
            raise ValueError("EMA hyperparameters differ from checkpoint")
        self.steps = int(state["steps"])
        self.shadow.load_state_dict(state["shadow_state_dict"])
