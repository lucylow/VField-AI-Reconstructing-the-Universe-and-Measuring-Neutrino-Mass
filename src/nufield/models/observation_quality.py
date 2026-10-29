"""Pre-training audits of cosmological observations and neural input tensors."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationFinding:
    severity: str
    code: str
    message: str
    action: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ObservationQualityReport:
    n_samples: int
    tensor_shape: tuple[int, ...]
    dtype: str
    findings: tuple[ObservationFinding, ...]

    def to_dict(self):
        return {
            "n_samples": self.n_samples,
            "tensor_shape": list(self.tensor_shape),
            "dtype": self.dtype,
            "findings": [f.to_dict() for f in self.findings],
            "can_train": not any(f.severity == "error" for f in self.findings),
            "scientific_approval": False,
        }


def audit_observations(
    observations: np.ndarray,
    *,
    expected_kind: str,
    expected_shape: tuple[int, ...] | None = None,
    allow_negative: bool = True,
    allowed_nonfinite_fraction: float = 0.0,
    constant_tolerance: float = 1e-12,
) -> ObservationQualityReport:
    x = np.asarray(observations)
    findings = []
    if x.ndim < 2:
        findings.append(
            ObservationFinding(
                "error",
                "NDIM",
                "Expected sample plus feature dimensions",
                "Provide [N,F] or [N,...] arrays.",
            )
        )
    if x.ndim and len(x) == 0:
        findings.append(
            ObservationFinding(
                "error",
                "EMPTY",
                "No observations",
                "Inspect the loader and data product selection.",
            )
        )
    if not np.issubdtype(x.dtype, np.number):
        findings.append(
            ObservationFinding(
                "error",
                "NONNUMERIC",
                "Observation tensor is not numeric",
                "Select physical values only; keep IDs/metadata separate.",
            )
        )
    elif x.ndim:
        fraction = float(np.mean(~np.isfinite(x)))
        if fraction > allowed_nonfinite_fraction:
            findings.append(
                ObservationFinding(
                    "error",
                    "NONFINITE",
                    f"Non-finite value fraction {fraction:.6g} exceeds tolerance {allowed_nonfinite_fraction:.6g}",
                    "Investigate corruption/missingness; never silently impute scientific data.",
                )
            )
        finite = x[np.isfinite(x)]
        if finite.size:
            if not allow_negative and float(finite.min()) < 0:
                findings.append(
                    ObservationFinding(
                        "warning",
                        "NEGATIVE",
                        "Negative values occur in a field expected to be nonnegative",
                        "Confirm observable convention; overdensity fields can validly be negative.",
                    )
                )
            if float(np.std(finite)) <= constant_tolerance:
                findings.append(
                    ObservationFinding(
                        "error",
                        "CONSTANT",
                        "All finite values are effectively constant",
                        "Check whether the wrong array key or an empty projection was selected.",
                    )
                )
            if np.quantile(np.abs(finite), 0.999) > 1e8:
                findings.append(
                    ObservationFinding(
                        "warning",
                        "EXTREME_SCALE",
                        "99.9th-percentile absolute input is very large",
                        "Check units, log transforms, and numerical dynamic range.",
                    )
                )
    if expected_shape is not None and tuple(x.shape[1:]) != tuple(expected_shape):
        findings.append(
            ObservationFinding(
                "error",
                "SHAPE_MISMATCH",
                f"Observed shape {x.shape[1:]} != expected {expected_shape}",
                "Do not resize fields silently; record an explicit resampling transform and convergence test.",
            )
        )
    legal = {"spectrum": 2, "field3d": 4, "field3d_channel": 5, "image2d": 4, "catalog": 3}
    if expected_kind not in legal:
        findings.append(
            ObservationFinding(
                "error",
                "KIND_UNKNOWN",
                f"Unknown expected_kind {expected_kind!r}",
                f"Choose from {sorted(legal)}.",
            )
        )
    elif x.ndim and x.ndim != legal[expected_kind]:
        findings.append(
            ObservationFinding(
                "warning",
                "KIND_RANK",
                f"Tensor rank {x.ndim} may not match {expected_kind} contract",
                f"Expected rank {legal[expected_kind]} including sample axis.",
            )
        )
    return ObservationQualityReport(
        int(x.shape[0]) if x.ndim else 0,
        tuple(x.shape[1:]) if x.ndim else (),
        str(x.dtype),
        tuple(findings),
    )
