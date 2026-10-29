"""Physics-oriented benchmark metrics for reconstructed initial density fields.

Metrics compare a reconstructed field with known simulation truth. They distinguish amplitude recovery,
mode correlation and real-space error; no single scalar score is sufficient. The estimator assumes a
periodic cubic box, and survey masks/noise require a separate observation operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.spectral_estimators import estimate_isotropic_power


@dataclass(frozen=True, slots=True)
class ReconstructionBandMetrics:
    k_mpc_inv: np.ndarray
    mode_counts: np.ndarray
    truth_power_mpc3: np.ndarray
    reconstruction_power_mpc3: np.ndarray
    cross_power_mpc3: np.ndarray
    amplitude_ratio: np.ndarray
    signed_cross_transfer: np.ndarray
    cross_correlation_coefficient: np.ndarray
    valid_bins: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "k_mpc_inv": self.k_mpc_inv.tolist(),
            "mode_counts": self.mode_counts.tolist(),
            "truth_power_mpc3": self.truth_power_mpc3.tolist(),
            "reconstruction_power_mpc3": self.reconstruction_power_mpc3.tolist(),
            "cross_power_mpc3": self.cross_power_mpc3.tolist(),
            "amplitude_ratio": self.amplitude_ratio.tolist(),
            "signed_cross_transfer": self.signed_cross_transfer.tolist(),
            "cross_correlation_coefficient": self.cross_correlation_coefficient.tolist(),
            "valid_bins": self.valid_bins.tolist(),
        }


@dataclass(frozen=True, slots=True)
class FieldReconstructionReport:
    box_size_mpc: float
    grid_size: int
    mean_truth: float
    mean_reconstruction: float
    std_truth: float
    std_reconstruction: float
    real_space_correlation: float
    normalized_rmse: float
    mae_in_truth_sigma: float
    residual_q16_sigma: float
    residual_median_sigma: float
    residual_q84_sigma: float
    bands: ReconstructionBandMetrics
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "box_size_mpc": self.box_size_mpc,
            "grid_size": self.grid_size,
            "mean_truth": self.mean_truth,
            "mean_reconstruction": self.mean_reconstruction,
            "std_truth": self.std_truth,
            "std_reconstruction": self.std_reconstruction,
            "real_space_correlation": self.real_space_correlation,
            "normalized_rmse": self.normalized_rmse,
            "mae_in_truth_sigma": self.mae_in_truth_sigma,
            "residual_q16_sigma": self.residual_q16_sigma,
            "residual_median_sigma": self.residual_median_sigma,
            "residual_q84_sigma": self.residual_q84_sigma,
            "bands": self.bands.to_dict(),
            "caveats": list(self.caveats),
        }


def evaluate_field_reconstruction(
    truth_field: np.ndarray,
    reconstructed_field: np.ndarray,
    box_size_mpc: float,
    *,
    k_edges_mpc_inv: np.ndarray | None = None,
    n_k_bins: int = 24,
    subtract_mean: bool = True,
) -> FieldReconstructionReport:
    """Compute scalar and scale-dependent metrics for a pair of periodic 3-D fields."""
    truth, recon = np.asarray(truth_field, float), np.asarray(reconstructed_field, float)
    if (
        truth.ndim != 3
        or truth.shape != recon.shape
        or len(set(truth.shape)) != 1
        or truth.shape[0] < 4
    ):
        raise ValueError(
            "truth and reconstruction must be matching cubic 3-D arrays with >=4 cells per axis"
        )
    if np.any(~np.isfinite(truth)) or np.any(~np.isfinite(recon)):
        raise ValueError("fields must be finite")
    if box_size_mpc <= 0 or not np.isfinite(box_size_mpc):
        raise ValueError("box_size_mpc must be positive and finite")
    sigma = float(np.std(truth))
    if sigma <= 0:
        raise ValueError("truth field must have nonzero variance")
    tr_mean, re_mean = float(np.mean(truth)), float(np.mean(recon))
    tc, rc = truth - tr_mean, recon - re_mean
    std_recon = float(np.std(recon))
    corr = float(np.mean(tc * rc) / (sigma * std_recon)) if std_recon > 0 else 0.0
    residual = recon - truth
    norm_rmse = float(np.sqrt(np.mean(residual**2)) / sigma)
    mae = float(np.mean(np.abs(residual)) / sigma)
    residual_sigma = (residual - np.mean(residual)) / sigma
    edges = k_edges_mpc_inv
    p_truth = estimate_isotropic_power(
        truth, box_size_mpc, bin_edges_mpc_inv=edges, n_bins=n_k_bins, subtract_mean=subtract_mean
    )
    p_recon = estimate_isotropic_power(
        recon,
        box_size_mpc,
        bin_edges_mpc_inv=p_truth.bin_edges_mpc_inv,
        n_bins=n_k_bins,
        subtract_mean=subtract_mean,
    )
    p_cross = estimate_isotropic_power(
        truth,
        box_size_mpc,
        field_b=recon,
        bin_edges_mpc_inv=p_truth.bin_edges_mpc_inv,
        n_bins=n_k_bins,
        subtract_mean=subtract_mean,
    )
    if not (
        np.array_equal(p_truth.mode_counts, p_recon.mode_counts)
        and np.array_equal(p_truth.mode_counts, p_cross.mode_counts)
    ):
        raise ArithmeticError("power estimators did not return consistent mode counts")
    valid = (p_truth.mode_counts > 0) & (p_truth.power_mpc3 > 0) & (p_recon.power_mpc3 > 0)
    amp_ratio = np.divide(
        p_recon.power_mpc3,
        p_truth.power_mpc3,
        out=np.full_like(p_truth.power_mpc3, np.nan),
        where=valid,
    )
    transfer = np.divide(
        p_cross.power_mpc3,
        p_truth.power_mpc3,
        out=np.full_like(p_truth.power_mpc3, np.nan),
        where=valid,
    )
    corr_k = np.divide(
        p_cross.power_mpc3,
        np.sqrt(p_truth.power_mpc3 * p_recon.power_mpc3),
        out=np.full_like(p_truth.power_mpc3, np.nan),
        where=valid,
    )
    bands = ReconstructionBandMetrics(
        p_truth.k_centers_mpc_inv,
        p_truth.mode_counts,
        p_truth.power_mpc3,
        p_recon.power_mpc3,
        p_cross.power_mpc3,
        amp_ratio,
        transfer,
        corr_k,
        valid,
    )
    return FieldReconstructionReport(
        float(box_size_mpc),
        int(truth.shape[0]),
        tr_mean,
        re_mean,
        sigma,
        std_recon,
        corr,
        norm_rmse,
        mae,
        float(np.quantile(residual_sigma, 0.16)),
        float(np.quantile(residual_sigma, 0.50)),
        float(np.quantile(residual_sigma, 0.84)),
        bands,
        (
            "Metrics assume periodic cubic fields with matching voxelization and coordinate units.",
            "A high field correlation can coexist with suppressed small-scale amplitude; inspect per-k diagnostics.",
            "Truth/reconstruction pairs from the same initial simulation are not independent test simulations.",
            "A low residual against simulated truth does not establish accuracy on real observed survey fields.",
        ),
    )


__all__ = [
    "ReconstructionBandMetrics",
    "FieldReconstructionReport",
    "evaluate_field_reconstruction",
]
