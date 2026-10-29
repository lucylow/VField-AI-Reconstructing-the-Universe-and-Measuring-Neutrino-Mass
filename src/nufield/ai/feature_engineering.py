"""Train-only transformations for spectral summary features used by AI models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from nufield.ai.normalization import Standardizer


@dataclass(frozen=True)
class SpectrumContract:
    k: tuple[float, ...]
    k_units: str
    power_units: str
    statistic: str
    box_size: float | None = None
    window_corrected: bool = False
    shot_noise_subtracted: bool = False

    def validate(self) -> list[str]:
        errors = []
        k = np.asarray(self.k, float)
        if k.ndim != 1 or len(k) < 3 or not np.isfinite(k).all() or np.any(k <= 0):
            errors.append("k must be a finite positive vector with at least three bins")
        elif np.any(np.diff(k) <= 0):
            errors.append("k must be strictly increasing")
        if not self.k_units.strip() or not self.power_units.strip():
            errors.append("k and power units must be explicitly declared")
        if not self.statistic.strip():
            errors.append("statistic definition must be declared")
        if self.box_size is not None and self.box_size <= 0:
            errors.append("box_size must be positive when provided")
        return errors


@dataclass
class SpectralFeaturePipeline:
    """Transform positive/negative spectrum bins to stable log summaries plus validity indicators.

    The pipeline's floor and standardizer are fitted only on training spectra. It does not subtract
    shot noise, correct survey windows, or decide which k range is scientifically valid; those must
    be explicit upstream choices recorded in SpectrumContract.
    """

    floor_fraction: float = 1e-6
    include_validity_mask: bool = True
    smooth_sigma_bins: float = 0.0
    scaler_: Standardizer | None = None
    floor_: np.ndarray | None = None
    n_bins_: int = 0
    contract_: SpectrumContract | None = None

    def fit(self, spectra: Any, contract: SpectrumContract) -> SpectralFeaturePipeline:
        x = np.asarray(spectra, float)
        errors = contract.validate()
        if errors:
            raise ValueError("invalid spectrum contract: " + "; ".join(errors))
        if x.ndim != 2 or len(x) < 2 or x.shape[1] != len(contract.k) or not np.isfinite(x).all():
            raise ValueError("spectra must be finite [N,K] and align with k contract")
        if self.floor_fraction <= 0 or self.smooth_sigma_bins < 0:
            raise ValueError("floor_fraction must be positive and smooth_sigma_bins nonnegative")
        positive = np.maximum(x, 0)
        scale = np.quantile(positive, 0.5, axis=0)
        global_scale = max(float(np.median(scale[scale > 0])) if np.any(scale > 0) else 1.0, 1e-30)
        self.floor_ = np.maximum(
            scale * self.floor_fraction, global_scale * self.floor_fraction * 1e-3
        )
        self.n_bins_ = x.shape[1]
        self.contract_ = contract
        transformed = self._raw_transform(x)
        self.scaler_ = Standardizer().fit(transformed)
        return self

    def _raw_transform(self, spectra: np.ndarray) -> np.ndarray:
        if self.floor_ is None:
            raise RuntimeError("fit must be called before transform")
        x = np.asarray(spectra, float)
        if x.ndim != 2 or x.shape[1] != self.n_bins_ or not np.isfinite(x).all():
            raise ValueError("spectra must be finite [N,K] with fitted bin count")
        valid = x > 0
        stable = np.log(np.maximum(x, self.floor_[None, :]))
        if self.smooth_sigma_bins:
            stable = gaussian_filter1d(stable, sigma=self.smooth_sigma_bins, axis=1, mode="nearest")
        features = [stable]
        if self.include_validity_mask:
            features.append(valid.astype(float))
        return np.concatenate(features, axis=1)

    def transform(self, spectra: Any) -> np.ndarray:
        if self.scaler_ is None:
            raise RuntimeError("fit must be called before transform")
        return self.scaler_.transform(self._raw_transform(np.asarray(spectra, float)))

    def to_dict(self) -> dict[str, Any]:
        if self.scaler_ is None or self.floor_ is None or self.contract_ is None:
            raise RuntimeError("fit must be called before exporting state")
        return {
            "floor_fraction": self.floor_fraction,
            "include_validity_mask": self.include_validity_mask,
            "smooth_sigma_bins": self.smooth_sigma_bins,
            "floor": self.floor_.tolist(),
            "n_bins": self.n_bins_,
            "contract": {
                "k": list(self.contract_.k),
                "k_units": self.contract_.k_units,
                "power_units": self.contract_.power_units,
                "statistic": self.contract_.statistic,
                "box_size": self.contract_.box_size,
                "window_corrected": self.contract_.window_corrected,
                "shot_noise_subtracted": self.contract_.shot_noise_subtracted,
            },
            "scaler": self.scaler_.to_dict(),
            "warning": "Fit all preprocessing on training simulations only.",
        }


def spectral_slopes(k: Any, power: Any, *, min_power: float = 1e-30) -> np.ndarray:
    """Compute d log(P) / d log(k) finite-difference slopes for positive bins."""
    kval, p = np.asarray(k, float), np.asarray(power, float)
    if kval.ndim != 1 or p.ndim not in {1, 2} or p.shape[-1] != len(kval):
        raise ValueError("k must be [K] and power [K] or [N,K]")
    if not np.isfinite(kval).all() or np.any(kval <= 0) or np.any(np.diff(kval) <= 0):
        raise ValueError("k must be finite, positive and increasing")
    if not np.isfinite(p).all() or min_power <= 0:
        raise ValueError("power must be finite and min_power positive")
    logk = np.log(kval)
    return np.gradient(np.log(np.maximum(p, min_power)), logk, axis=-1, edge_order=2)


def smooth_broadband_component(power: Any, *, sigma_bins: float = 4.0) -> np.ndarray:
    """Smooth a positive spectrum to obtain a diagnostic broadband trend (not BAO template fitting)."""
    p = np.asarray(power, float)
    if p.ndim not in {1, 2} or not np.isfinite(p).all() or np.any(p <= 0) or sigma_bins <= 0:
        raise ValueError("power must be finite positive [K]/[N,K] and sigma_bins positive")
    return np.exp(gaussian_filter1d(np.log(p), sigma=sigma_bins, axis=-1, mode="nearest"))
