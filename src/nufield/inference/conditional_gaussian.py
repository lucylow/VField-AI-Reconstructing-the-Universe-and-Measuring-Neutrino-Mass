"""Optional MLP baseline for conditional mean/variance estimation.

This is not a full flexible posterior estimator. It predicts a diagonal Gaussian approximation and
must be checked for calibration and multimodality before scientific use.
"""

from __future__ import annotations

import numpy as np

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch import nn
except ImportError:
    torch, nn = None, None

if nn is not None:

    class ConditionalGaussianMLP(nn.Module):
        def __init__(self, input_dim: int, parameter_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2 * parameter_dim),
            )
            self.parameter_dim = parameter_dim

        def forward(self, x):
            output = self.body(x)
            mean, log_var = output[..., : self.parameter_dim], output[..., self.parameter_dim :]
            return mean, torch.clamp(log_var, min=-12, max=8)

        @staticmethod
        def gaussian_nll(mean, log_var, y):
            return 0.5 * torch.mean(log_var + (y - mean) ** 2 * torch.exp(-log_var))
else:

    class ConditionalGaussianMLP:
        def __init__(self, *args, **kwargs):
            raise OptionalDependencyError(
                "Install nufield-starter[ml] to use ConditionalGaussianMLP"
            )


def train_conditional_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Fit an optional conditional diagonal Gaussian model on arrays (N,F) -> (N,P)."""
    if torch is None:
        raise OptionalDependencyError("Install nufield-starter[ml] to train this model")
    xa, ya = np.asarray(x, np.float32), np.asarray(y, np.float32)
    if xa.ndim != 2 or ya.ndim != 2 or len(xa) != len(ya) or len(xa) < 5:
        raise ValueError("x/y must be aligned 2-D arrays with at least five rows")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("invalid training hyperparameters")
    torch.manual_seed(seed)
    model = ConditionalGaussianMLP(xa.shape[1], ya.shape[1]).to(device)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(xa), torch.from_numpy(ya))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        losses = []
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, logvar = model(bx)
            loss = model.gaussian_nll(mean, logvar, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_nll": float(np.mean(losses))})
    return {
        "model": model,
        "history": history,
        "physics_level": "inherited from dataset",
        "posterior_family": "diagonal Gaussian; potentially misspecified",
    }
