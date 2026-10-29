"""Out-of-distribution diagnostics fitted and calibrated on separate data partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class MahalanobisOOD:
    """Shrinkage-regularized Mahalanobis detector for embedding or summary features.

    Use the representation and covariance fitted on training data. Calibrate the decision threshold
    with an independent in-distribution validation set, not the final test set.
    """

    ridge: float = 1e-4
    location_: np.ndarray | None = None
    precision_: np.ndarray | None = None
    threshold_: float | None = None
    calibration_quantile_: float | None = None
    n_fit_: int = 0

    def fit(self, features: Any) -> MahalanobisOOD:
        x = np.asarray(features, float)
        if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
            raise ValueError("features must be a finite [N,F] matrix with N>=3")
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        self.location_ = x.mean(axis=0)
        centered = x - self.location_
        cov = np.cov(centered, rowvar=False, ddof=1)
        cov = np.atleast_2d(cov)
        scale = max(float(np.trace(cov) / cov.shape[0]), 1.0)
        cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov)) + np.eye(cov.shape[0]) * self.ridge * scale
        self.precision_ = np.linalg.pinv(cov, hermitian=True)
        self.n_fit_ = int(len(x))
        self.threshold_ = None
        return self

    def score(self, features: Any) -> np.ndarray:
        if self.location_ is None or self.precision_ is None:
            raise RuntimeError("fit must be called first")
        x = np.asarray(features, float)
        if x.ndim != 2 or x.shape[1] != len(self.location_) or not np.isfinite(x).all():
            raise ValueError("features must be finite [N,F] with matching width")
        delta = x - self.location_
        squared = np.einsum("ni,ij,nj->n", delta, self.precision_, delta, optimize=True)
        return np.sqrt(np.maximum(squared, 0.0))

    def calibrate(
        self, in_distribution_validation: Any, *, quantile: float = 0.99
    ) -> MahalanobisOOD:
        if not 0.5 < quantile < 1:
            raise ValueError("quantile must be in (0.5,1)")
        scores = self.score(in_distribution_validation)
        if len(scores) < 10:
            raise ValueError(
                "at least 10 held-out in-distribution calibration rows are recommended"
            )
        self.threshold_ = float(np.quantile(scores, quantile, method="higher"))
        self.calibration_quantile_ = float(quantile)
        return self

    def predict_ood(self, features: Any) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("calibrate on an independent in-distribution validation set first")
        return self.score(features) > self.threshold_

    def to_dict(self) -> dict[str, Any]:
        if self.location_ is None or self.precision_ is None:
            raise RuntimeError("fit must be called first")
        return {
            "ridge": self.ridge,
            "location": self.location_.tolist(),
            "precision": self.precision_.tolist(),
            "threshold": self.threshold_,
            "calibration_quantile": self.calibration_quantile_,
            "n_fit": self.n_fit_,
            "warning": "Mahalanobis distance is a diagnostic, not a guaranteed OOD probability.",
        }


@dataclass(frozen=True)
class NearestReferenceOOD:
    reference_features: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    threshold: float | None = None
    calibration_quantile: float | None = None

    @classmethod
    def fit(cls, train_features: Any) -> NearestReferenceOOD:
        x = np.asarray(train_features, float)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("train_features must be finite [N,F] with N>=2")
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls((x - center) / scale, center, scale)

    def score(self, features: Any) -> np.ndarray:
        x = np.asarray(features, float)
        if x.ndim != 2 or x.shape[1] != len(self.center) or not np.isfinite(x).all():
            raise ValueError("features must be finite [N,F] with matching width")
        return cdist((x - self.center) / self.scale, self.reference_features).min(axis=1)

    def calibrate(
        self, calibration_features: Any, *, quantile: float = 0.99
    ) -> NearestReferenceOOD:
        if not 0.5 < quantile < 1:
            raise ValueError("quantile must be in (0.5,1)")
        scores = self.score(calibration_features)
        if len(scores) < 10:
            raise ValueError("at least 10 independent calibration examples are recommended")
        return NearestReferenceOOD(
            self.reference_features,
            self.center,
            self.scale,
            float(np.quantile(scores, quantile, method="higher")),
            quantile,
        )

    def predict_ood(self, features: Any) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("calibrate before predicting OOD")
        return self.score(features) > self.threshold


def ood_separation_report(
    in_scores: Any, shifted_scores: Any, *, threshold: float
) -> dict[str, float]:
    """Report threshold behavior on two independent score vectors; does not fit the threshold."""
    a, b = np.asarray(in_scores, float), np.asarray(shifted_scores, float)
    if a.ndim != 1 or b.ndim != 1 or len(a) == 0 or len(b) == 0:
        raise ValueError("score inputs must be nonempty vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    return {
        "in_distribution_false_alarm_rate": float(np.mean(a > threshold)),
        "shifted_detection_rate": float(np.mean(b > threshold)),
        "threshold": float(threshold),
        "n_in": int(len(a)),
        "n_shifted": int(len(b)),
    }
