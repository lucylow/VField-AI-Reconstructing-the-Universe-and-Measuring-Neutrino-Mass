"""Tiny optional training loop for a paired-field dataset.

Expected arrays: x and y with shape (N, X, Y, Z); this function does not create physically valid pairs.
"""

from __future__ import annotations

import numpy as np

from nufield.exceptions import OptionalDependencyError


def train_unet3d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise OptionalDependencyError(
            "Install nufield-starter[ml] to train the optional U-Net"
        ) from exc
    from nufield.reconstruction.network import UNet3D

    xa, ya = np.asarray(x, np.float32), np.asarray(y, np.float32)
    if xa.ndim != 4 or ya.ndim != 4 or xa.shape != ya.shape or len(xa) < 2:
        raise ValueError("x and y must be same-shaped arrays of shape (N,X,Y,Z), N>=2")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("invalid training hyperparameters")
    torch.manual_seed(seed)
    model = UNet3D().to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xa[:, None]), torch.from_numpy(ya[:, None])),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = torch.nn.functional.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
    return {
        "model": model,
        "history": history,
        "physics_level": "not specified by trainer; inspect data provenance",
    }
