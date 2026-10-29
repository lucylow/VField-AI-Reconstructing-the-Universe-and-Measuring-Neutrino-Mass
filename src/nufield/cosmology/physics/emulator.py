"""Small radial-basis spectral emulator scaffold for boltzmann-code acceleration experiments.

This class deliberately does not report posterior uncertainty. It returns an interpolated central
prediction and diagnostics for support/leave-one-out error. An emulator intended for inference needs
an independently validated error model, especially for derivatives with respect to neutrino mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RBFInterpolator

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class EmulatorDiagnostics:
    n_train: int
    n_parameters: int
    n_k: int
    training_log_power_rms_residual: float
    nearest_neighbor_radius_95: float
    has_uncertainty_model: bool = False
    warnings: tuple[str, ...] = (
        "RBF central estimate only; no posterior predictive uncertainty is provided.",
        "Training residual is not an independent validation error.",
    )


class SpectralRBFEmulator:
    """RBF interpolation of log P(k) over a multidimensional parameter grid."""

    def __init__(
        self,
        *,
        smoothing: float = 1e-8,
        kernel: str = "thin_plate_spline",
        support_radius_factor: float = 2.0,
    ):
        if smoothing < 0 or support_radius_factor <= 0:
            raise ValueError("smoothing must be non-negative and support_radius_factor positive")
        self.smoothing = float(smoothing)
        self.kernel = kernel
        self.support_radius_factor = float(support_radius_factor)
        self._fitted = False

    def fit(
        self,
        theta: np.ndarray,
        log_pk: np.ndarray,
        k_mpc_inv: np.ndarray,
        parameter_names: tuple[str, ...] | list[str],
        *,
        group_ids: np.ndarray | None = None,
    ) -> SpectralRBFEmulator:
        x, y, k = np.asarray(theta, float), np.asarray(log_pk, float), np.asarray(k_mpc_inv, float)
        names = tuple(parameter_names)
        if x.ndim != 2 or len(x) < max(6, x.shape[1] + 2) or x.shape[1] != len(names):
            raise ValueError("theta must be (n_sim,n_parameter) with enough simulations and names")
        if y.ndim != 2 or y.shape != (x.shape[0], k.size):
            raise ValueError("log_pk must have shape (n_sim,n_k)")
        if k.ndim != 1 or k.size < 3 or np.any(k <= 0) or np.any(np.diff(k) <= 0):
            raise ValueError("k grid must be positive, increasing and have >=3 points")
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("training arrays must be finite")
        if group_ids is not None:
            gids = np.asarray(group_ids)
            if gids.shape != (x.shape[0],):
                raise ValueError("group_ids must contain one group per simulation")
            # Duplicated groups are legal in a training ensemble, but they must not be split across
            # train/validation/test in the outer workflow; this class only receives its fit partition.
        self.parameter_names = names
        self.theta_mean = x.mean(axis=0)
        self.theta_scale = x.std(axis=0, ddof=0)
        self.theta_scale[self.theta_scale == 0] = 1.0
        xs = (x - self.theta_mean) / self.theta_scale
        self.theta_train = xs
        self.theta_train_raw = x.copy()
        self.k_mpc_inv = k.copy()
        self.log_pk_train = y.copy()
        self._rbf = RBFInterpolator(xs, y, kernel=self.kernel, smoothing=self.smoothing, degree=1)
        pred = np.asarray(self._rbf(xs))
        rms = float(np.sqrt(np.mean((pred - y) ** 2)))
        distances = []
        for i in range(xs.shape[0]):
            other = np.delete(xs, i, axis=0)
            distances.append(float(np.min(np.linalg.norm(other - xs[i], axis=1))))
        self.nn_radius_95 = float(np.quantile(distances, 0.95))
        if self.nn_radius_95 <= 0:
            self.nn_radius_95 = float(np.max(np.linalg.norm(xs - xs.mean(axis=0), axis=1)) or 1.0)
        self.diagnostics = EmulatorDiagnostics(
            x.shape[0], x.shape[1], k.size, rms, self.nn_radius_95, False
        )
        self._fitted = True
        return self

    def predict_log_pk(self, theta: np.ndarray, *, allow_extrapolation: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("emulator must be fit before prediction")
        x = np.asarray(theta, float)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != len(self.parameter_names) or np.any(~np.isfinite(x)):
            raise ValueError(
                "theta must be finite with shape (n_parameter,) or (n_sample,n_parameter)"
            )
        xs = (x - self.theta_mean) / self.theta_scale
        nn_dist = np.min(
            np.linalg.norm(xs[:, None, :] - self.theta_train[None, :, :], axis=2), axis=1
        )
        if not allow_extrapolation and np.any(
            nn_dist > self.support_radius_factor * self.nn_radius_95
        ):
            raise ValueError("query lies outside training support; extrapolation disabled")
        result = np.asarray(self._rbf(xs), float)
        return result[0] if single else result

    def predict_spectrum(
        self,
        theta: np.ndarray,
        redshift: float,
        *,
        allow_extrapolation: bool = False,
        source: str = "RBF-emulator-central-estimate",
    ) -> MatterPowerSpectrum:
        logp = self.predict_log_pk(theta, allow_extrapolation=allow_extrapolation)
        if np.asarray(logp).ndim != 1:
            raise ValueError("predict_spectrum accepts one parameter vector")
        return MatterPowerSpectrum(
            self.k_mpc_inv,
            np.exp(logp),
            redshift,
            linear=True,
            source=source,
            metadata=(
                ("emulator", "RBF"),
                ("uncertainty_model", "none"),
                ("parameter_names", ",".join(self.parameter_names)),
            ),
        )

    def leave_one_out_diagnostic(self) -> dict[str, float | str]:
        """Estimate LOO RMSE by refitting one small RBF per held-out simulation."""
        if not self._fitted:
            raise RuntimeError("emulator must be fit first")
        x, y = self.theta_train, self.log_pk_train
        pred = np.zeros_like(y)
        for i in range(x.shape[0]):
            mask = np.arange(x.shape[0]) != i
            # Thin-plate spline degree=1 needs enough independent points for its polynomial tail.
            model = RBFInterpolator(
                x[mask], y[mask], kernel=self.kernel, smoothing=self.smoothing, degree=1
            )
            pred[i] = model(x[i : i + 1])[0]
        residual = pred - y
        return {
            "metric": "leave_one_out_log_power_rmse",
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "median_abs": float(np.median(np.abs(residual))),
            "note": "LOO may still overstate generalization when simulations share parent realizations.",
        }


__all__ = ["EmulatorDiagnostics", "SpectralRBFEmulator"]
