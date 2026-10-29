"""Differentiable field losses for reconstruction experiments.

These losses are only meaningful when fields share a documented density convention, grid, and
periodic box size. They are not a cosmological likelihood and must not be used to imply a neutrino
mass detection. The P(k) loss below omits survey windows, shot-noise terms, and covariance weighting.
"""

from __future__ import annotations

from nufield.exceptions import OptionalDependencyError

try:
    import torch
    from torch.nn import functional as F
except ImportError:
    torch, F = None, None


def _need_torch():
    if torch is None:
        raise OptionalDependencyError(
            "Install νField with the [ml] extra to use differentiable losses"
        )


def _as_5d(field):
    _need_torch()
    if field.ndim == 4:
        field = field[:, None]
    if field.ndim != 5 or field.shape[1] != 1:
        raise ValueError("fields must be [B,D,H,W] or [B,1,D,H,W]")
    if min(field.shape[-3:]) < 4:
        raise ValueError("each grid dimension must be at least 4")
    if not torch.is_floating_point(field):
        raise ValueError("field tensor must be floating point")
    return field


def isotropic_power_spectrum(
    field,
    *,
    box_size_mpc_h: float,
    n_bins: int = 20,
    k_min: float | None = None,
    k_max: float | None = None,
    eps: float = 1e-12,
):
    """Batch isotropic 3-D P(k) with differentiable bin means. Returns k-centers and P(k).

    Convention: P(k) = V / N_voxels**2 * |FFT(delta)|**2, using the unnormalized forward FFT.
    The zero mode is excluded. Box size is in Mpc/h, so resulting k is h/Mpc and P is (Mpc/h)^3.
    """
    x = _as_5d(field)
    if box_size_mpc_h <= 0 or n_bins < 3 or eps <= 0:
        raise ValueError("invalid box size, bins, or eps")
    b, _, d, h, w = x.shape
    # Remove sample mean; density contrast convention must have been fixed by the caller.
    delta = x[:, 0] - x[:, 0].mean(dim=(-3, -2, -1), keepdim=True)
    ft = torch.fft.fftn(delta, dim=(-3, -2, -1), norm="backward")
    volume = float(box_size_mpc_h) ** 3
    p3d = ft.abs().square() * (volume / float((d * h * w) ** 2))
    freq = [
        torch.fft.fftfreq(n, d=float(box_size_mpc_h) / n, device=x.device) * (2.0 * torch.pi)
        for n in (d, h, w)
    ]
    kz, ky, kx = torch.meshgrid(freq[0], freq[1], freq[2], indexing="ij")
    kmag = torch.sqrt(kz.square() + ky.square() + kx.square())
    nonzero = kmag > 0
    kmin = float(k_min) if k_min is not None else float(2.0 * torch.pi / box_size_mpc_h)
    kmax = float(k_max) if k_max is not None else float(kmag.max().detach().cpu())
    if not (0 < kmin < kmax):
        raise ValueError("must satisfy 0 < k_min < k_max")
    edges = torch.logspace(
        torch.log10(torch.tensor(kmin, device=x.device)),
        torch.log10(torch.tensor(kmax, device=x.device)),
        n_bins + 1,
        device=x.device,
    )
    idx = torch.bucketize(kmag.contiguous(), edges) - 1
    idx = idx.clamp(0, n_bins - 1)
    k_centers = torch.sqrt(edges[:-1] * edges[1:])
    outputs = []
    for i in range(n_bins):
        mask = (idx == i) & nonzero & (kmag >= kmin) & (kmag <= kmax)
        count = int(mask.sum().item())
        if count:
            outputs.append(p3d[:, mask].sum(dim=1) / count)
        else:
            # A differentiable zero placeholder, marked by a separate support mask below.
            outputs.append(p3d.sum(dim=(-3, -2, -1)) * 0.0)
    spectra = torch.stack(outputs, dim=1)
    occupied = torch.stack(
        [((idx == i) & nonzero & (kmag >= kmin) & (kmag <= kmax)).any() for i in range(n_bins)]
    ).to(x.device)
    return k_centers, spectra.clamp_min(eps), occupied


def log_power_spectrum_loss(
    predicted,
    target,
    *,
    box_size_mpc_h: float,
    n_bins: int = 20,
    eps: float = 1e-12,
    reduction: str = "mean",
):
    """Huber loss in log P(k) over occupied isotropic bins; no survey window/covariance correction."""
    p = _as_5d(predicted)
    t = _as_5d(target)
    if p.shape != t.shape:
        raise ValueError("predicted and target field shapes must match")
    _, pp, occp = isotropic_power_spectrum(p, box_size_mpc_h=box_size_mpc_h, n_bins=n_bins, eps=eps)
    _, pt, occt = isotropic_power_spectrum(t, box_size_mpc_h=box_size_mpc_h, n_bins=n_bins, eps=eps)
    mask = occp & occt
    if not bool(mask.any()):
        raise ValueError("no occupied power-spectrum bins; check box size/grid/bounds")
    value = F.smooth_l1_loss(torch.log(pp[:, mask]), torch.log(pt[:, mask]), reduction="none").mean(
        dim=1
    )
    if reduction == "mean":
        return value.mean()
    if reduction == "none":
        return value
    if reduction == "sum":
        return value.sum()
    raise ValueError("reduction must be mean, sum, or none")


def mass_conservation_loss(predicted, target, *, reduction: str = "mean"):
    """Penalize disagreement in per-volume field mean; use only if physical convention warrants it."""
    p = _as_5d(predicted)
    t = _as_5d(target)
    if p.shape != t.shape:
        raise ValueError("predicted and target field shapes must match")
    loss = (p.mean(dim=(-3, -2, -1)) - t.mean(dim=(-3, -2, -1))).square().mean(dim=1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be mean, sum, or none")


def gradient_matching_loss(predicted, target, *, reduction: str = "mean"):
    """Match periodic finite-difference gradients; emphasizes structure without asserting mass physics."""
    p = _as_5d(predicted)
    t = _as_5d(target)
    if p.shape != t.shape:
        raise ValueError("predicted and target field shapes must match")
    losses = []
    for dim in (-3, -2, -1):
        gp = torch.roll(p, shifts=-1, dims=dim) - p
        gt = torch.roll(t, shifts=-1, dims=dim) - t
        losses.append((gp - gt).square().mean(dim=(-4, -3, -2, -1)))
    per_sample = torch.stack(losses, dim=1).mean(dim=1)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError("reduction must be mean, sum, or none")


def composite_reconstruction_loss(
    predicted,
    target,
    *,
    box_size_mpc_h: float,
    lambda_field: float = 1.0,
    lambda_pk: float = 0.1,
    lambda_grad: float = 0.05,
    lambda_mean: float = 0.0,
    n_bins: int = 20,
) -> dict[str, object]:
    """Return total and component losses for transparent ablations, not just a scalar objective."""
    for name, val in (
        ("lambda_field", lambda_field),
        ("lambda_pk", lambda_pk),
        ("lambda_grad", lambda_grad),
        ("lambda_mean", lambda_mean),
    ):
        if val < 0:
            raise ValueError(f"{name} must be nonnegative")
    p = _as_5d(predicted)
    t = _as_5d(target)
    if p.shape != t.shape:
        raise ValueError("predicted and target field shapes must match")
    field_loss = F.mse_loss(p, t)
    pk_loss = log_power_spectrum_loss(p, t, box_size_mpc_h=box_size_mpc_h, n_bins=n_bins)
    grad_loss = gradient_matching_loss(p, t)
    mean_loss = mass_conservation_loss(p, t)
    total = (
        lambda_field * field_loss
        + lambda_pk * pk_loss
        + lambda_grad * grad_loss
        + lambda_mean * mean_loss
    )
    return {
        "total": total,
        "field_mse": field_loss,
        "log_pk_huber": pk_loss,
        "gradient_mse": grad_loss,
        "mean_conservation": mean_loss,
        "weights": {
            "field": lambda_field,
            "pk": lambda_pk,
            "gradient": lambda_grad,
            "mean": lambda_mean,
        },
    }
