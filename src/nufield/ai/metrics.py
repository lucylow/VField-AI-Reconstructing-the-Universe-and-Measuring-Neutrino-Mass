"""AI evaluation metrics for point predictions, predictive distributions and posterior samples."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtr


def _matrix(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or len(arr) < 1 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite [N,P] array")
    return arr


def regression_metrics(
    target: Any, prediction: Any, *, variance: Any | None = None
) -> dict[str, Any]:
    """Compute target-wise point metrics and optional Gaussian probabilistic scores."""
    y, p = _matrix(target, "target"), _matrix(prediction, "prediction")
    if y.shape != p.shape:
        raise ValueError("target and prediction shape mismatch")
    residual = p - y
    mse = np.mean(residual**2, axis=0)
    mae = np.mean(np.abs(residual), axis=0)
    denom = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(denom > 0, 1 - np.sum(residual**2, axis=0) / np.maximum(denom, 1e-15), np.nan)
    report: dict[str, Any] = {
        "n": int(len(y)),
        "target_dim": int(y.shape[1]),
        "mse_mean": float(mse.mean()),
        "mae_mean": float(mae.mean()),
        "rmse_mean": float(np.sqrt(mse).mean()),
        "mse_by_target": mse.tolist(),
        "mae_by_target": mae.tolist(),
        "rmse_by_target": np.sqrt(mse).tolist(),
        "r2_by_target": r2.tolist(),
        "bias_by_target": residual.mean(axis=0).tolist(),
    }
    if variance is not None:
        v = _matrix(variance, "variance")
        if v.shape != y.shape or np.any(v <= 0):
            raise ValueError("variance must match target shape and be positive")
        report["gaussian_nll_mean"] = float(
            np.mean(0.5 * (np.log(2 * np.pi * v) + residual**2 / v))
        )
        standardized = residual / np.sqrt(v)
        report["standardized_residual_mean"] = standardized.mean(axis=0).tolist()
        report["standardized_residual_std"] = standardized.std(axis=0, ddof=1).tolist()
        report["coverage_68_by_target"] = (np.abs(standardized) <= 1).mean(axis=0).tolist()
        report["coverage_95_by_target"] = (
            (np.abs(standardized) <= 1.959963984540054).mean(axis=0).tolist()
        )
    return report


def gaussian_crps(mean: Any, variance: Any, target: Any) -> np.ndarray:
    """Closed-form CRPS for a univariate Gaussian predictive distribution, per row/target."""
    m, v, y = map(lambda z: np.asarray(z, float), (mean, variance, target))
    if (
        m.shape != v.shape
        or m.shape != y.shape
        or not np.isfinite(m).all()
        or not np.isfinite(v).all()
        or not np.isfinite(y).all()
        or np.any(v <= 0)
    ):
        raise ValueError(
            "mean, positive variance and target arrays must have identical finite shapes"
        )
    sigma = np.sqrt(v)
    z = (y - m) / sigma
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return sigma * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def calibration_curve(mean: Any, variance: Any, target: Any, *, n_bins: int = 10) -> dict[str, Any]:
    """Compare nominal Gaussian interval coverage with empirical coverage across confidence levels."""
    if n_bins < 3:
        raise ValueError("n_bins must be >=3")
    mu, var, y = _matrix(mean, "mean"), _matrix(variance, "variance"), _matrix(target, "target")
    if mu.shape != var.shape or mu.shape != y.shape or np.any(var <= 0):
        raise ValueError("mean, variance and target shape mismatch or nonpositive variance")
    z = np.abs((y - mu) / np.sqrt(var))
    nominal = np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins)
    # Invert central normal coverage to a z threshold. A small bisection avoids adding scipy.stats.
    from scipy.stats import norm

    thresholds = norm.ppf((1 + nominal) / 2)
    empirical = np.asarray([(z <= t).mean() for t in thresholds])
    return {
        "nominal": nominal.tolist(),
        "empirical": empirical.tolist(),
        "absolute_calibration_error_mean": float(np.mean(np.abs(empirical - nominal))),
        "n": int(len(y)),
        "target_dim": int(y.shape[1]),
    }


def simulation_based_calibration_ranks(posterior_draws: Any, truths: Any) -> np.ndarray:
    """Calculate rank counts for iid posterior draws and known simulated parameters.

    Inputs: posterior_draws [N_simulations,N_draws,P], truths [N_simulations,P]. Ties are handled
    conservatively by random tie breaking with a fixed seed passed by the caller via ``rng``.
    This function defaults to deterministic mid-ranks for exact reproducibility.
    """
    draws, theta = np.asarray(posterior_draws, float), _matrix(truths, "truths")
    if draws.ndim != 3 or draws.shape[0] != len(theta) or draws.shape[2] != theta.shape[1]:
        raise ValueError("draws must be [N_sim,N_draw,P] and truths [N_sim,P]")
    if draws.shape[1] < 1 or not np.isfinite(draws).all():
        raise ValueError("posterior draws must be finite and nonempty")
    lower = (draws < theta[:, None, :]).sum(axis=1)
    ties = (draws == theta[:, None, :]).sum(axis=1)
    return lower + ties / 2


def sbc_rank_histograms(
    posterior_draws: Any, truths: Any, *, bins: int | None = None
) -> dict[str, Any]:
    """Compute parameter-wise normalized SBC rank histograms."""
    draws = np.asarray(posterior_draws, float)
    ranks = simulation_based_calibration_ranks(draws, truths)
    n_draws = draws.shape[1]
    n_bins = n_draws + 1 if bins is None else int(bins)
    if n_bins < 2:
        raise ValueError("bins must be >=2")
    histograms = []
    for j in range(ranks.shape[1]):
        counts, edges = np.histogram(ranks[:, j], bins=n_bins, range=(0, n_draws + 1))
        histograms.append((counts / max(counts.sum(), 1)).tolist())
    return {
        "histograms": histograms,
        "bins": n_bins,
        "n_simulations": int(len(ranks)),
        "draws_per_simulation": int(n_draws),
        "warning": "A flat rank histogram is a diagnostic, not proof of all forms of calibration.",
    }


def energy_score(samples: Any, truth: Any, *, max_pairs: int = 2000, seed: int = 42) -> np.ndarray:
    """Multivariate posterior energy score using sample approximations.

    samples has shape [N,S,P] and truth [N,P]. Pairwise sample distances are subsampled when the
    draw count is large to bound memory and runtime.
    """
    x, y = np.asarray(samples, float), _matrix(truth, "truth")
    if x.ndim != 3 or x.shape[0] != len(y) or x.shape[2] != y.shape[1] or x.shape[1] < 2:
        raise ValueError("samples must be [N,S,P] with S>=2 and truth [N,P]")
    if not np.isfinite(x).all():
        raise ValueError("samples must be finite")
    first = np.linalg.norm(x - y[:, None, :], axis=-1).mean(axis=1)
    rng = np.random.default_rng(seed)
    second = np.empty(len(x), float)
    n = x.shape[1]
    if n * n <= max_pairs:
        for i in range(len(x)):
            second[i] = np.linalg.norm(x[i, :, None, :] - x[i, None, :, :], axis=-1).mean() / 2
    else:
        for i in range(len(x)):
            a = rng.integers(0, n, size=max_pairs)
            b = rng.integers(0, n, size=max_pairs)
            second[i] = np.linalg.norm(x[i, a] - x[i, b], axis=-1).mean() / 2
    return first - second


def stratified_error_report(
    target: Any, prediction: Any, *, stratifier: Any, n_bins: int = 5
) -> dict[str, Any]:
    """Report MAE and bias in equal-count bins of a declared scalar diagnostic.

    Choose the stratifier before opening the locked test results (e.g., signal-to-noise or redshift).
    """
    y, p = _matrix(target, "target"), _matrix(prediction, "prediction")
    z = np.asarray(stratifier, float).reshape(-1)
    if y.shape != p.shape or len(z) != len(y) or not np.isfinite(z).all() or n_bins < 2:
        raise ValueError("invalid targets, predictions, stratifier, or n_bins")
    edges = np.unique(np.quantile(z, np.linspace(0, 1, n_bins + 1)))
    out = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (z >= left) & (z <= right if i == len(edges) - 2 else z < right)
        if not mask.any():
            continue
        residual = p[mask] - y[mask]
        out.append(
            {
                "low": float(left),
                "high": float(right),
                "n": int(mask.sum()),
                "mae": float(np.abs(residual).mean()),
                "bias": float(residual.mean()),
            }
        )
    return {"bins": out, "stratifier_range": [float(z.min()), float(z.max())]}
