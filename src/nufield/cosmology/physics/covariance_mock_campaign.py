"""Repeatable finite-mock covariance campaigns for cosmological observables.

This module studies estimator variability as the number of mocks changes. It does not claim that
resampling a small set of mocks creates independent information; the synthetic experiment is useful
for testing software and sample-covariance behavior only. Real campaigns should preserve parent-box and
suite identifiers so correlated mock families are not misrepresented as independent samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.theory_covariance_likelihood import (
    covariance_regularization_scan,
    hartlap_precision_factor,
    sample_covariance_from_mocks,
)


@dataclass(frozen=True, slots=True)
class CovarianceMockTrial:
    n_mocks: int
    rank: int
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    hartlap_factor: float | None
    relative_frobenius_error: float
    shrinkage_alpha: float
    shrunk_condition_number: float

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "n_mocks": self.n_mocks,
            "rank": self.rank,
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "condition_number": self.condition_number,
            "hartlap_factor": self.hartlap_factor,
            "relative_frobenius_error": self.relative_frobenius_error,
            "shrinkage_alpha": self.shrinkage_alpha,
            "shrunk_condition_number": self.shrunk_condition_number,
        }


def run_covariance_mock_campaign(
    mock_data: np.ndarray,
    truth_covariance: np.ndarray,
    *,
    mock_counts: tuple[int, ...],
    shrinkage_alpha: float = 0.1,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap mock rows and quantify covariance error/conditioning for predeclared sample sizes.

    The caller supplies a reference covariance to quantify error; this is normally available only in a
    controlled synthetic experiment. For a real suite with unknown truth covariance, report only
    stability and cross-validation diagnostics instead of labeling relative covariance error.
    """
    x, truth = np.asarray(mock_data, float), np.asarray(truth_covariance, float)
    if x.ndim != 2 or x.shape[0] < 2 or np.any(~np.isfinite(x)):
        raise ValueError("mock_data must be finite shape (n_mocks,n_data)")
    if truth.shape != (x.shape[1], x.shape[1]) or np.any(~np.isfinite(truth)):
        raise ValueError("truth_covariance shape must match data dimension and be finite")
    if not np.allclose(truth, truth.T, atol=1e-10) or np.linalg.eigvalsh(truth).min() <= 0:
        raise ValueError("truth_covariance must be symmetric positive definite")
    counts = tuple(int(n) for n in mock_counts)
    if (
        not counts
        or any(n < 2 or n > x.shape[0] for n in counts)
        or len(set(counts)) != len(counts)
    ):
        raise ValueError("mock_counts must be unique values between 2 and available mock count")
    if shrinkage_alpha < 0 or shrinkage_alpha > 1 or not np.isfinite(shrinkage_alpha) or seed < 0:
        raise ValueError("invalid shrinkage alpha or seed")
    rng = np.random.default_rng(seed)
    rows: list[CovarianceMockTrial] = []
    ndata = x.shape[1]
    for n in sorted(counts):
        # One replicate per requested n; for robust conclusions run multiple seeds or repeat externally.
        idx = rng.choice(x.shape[0], size=n, replace=False)
        cov = sample_covariance_from_mocks(x[idx])
        eig = np.linalg.eigvalsh(cov)
        rank = int(np.linalg.matrix_rank(cov))
        condition = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
        h = None
        if n > ndata + 2:
            h = hartlap_precision_factor(n, ndata)
        rel = float(np.linalg.norm(cov - truth, ord="fro") / np.linalg.norm(truth, ord="fro"))
        try:
            reg = covariance_regularization_scan(cov, (shrinkage_alpha,))
            shrunk_cond = float(reg[0]["condition_number"])
        except ValueError:
            shrunk_cond = float("inf")
        rows.append(
            CovarianceMockTrial(
                n,
                rank,
                float(eig.min()),
                float(eig.max()),
                condition,
                h,
                rel,
                float(shrinkage_alpha),
                shrunk_cond,
            )
        )
    return {
        "n_data": ndata,
        "available_mock_rows": int(x.shape[0]),
        "seed": int(seed),
        "trials": [r.to_dict() for r in rows],
        "interpretation": "Synthetic/reference-covariance experiment only. Bootstrap subsets do not increase the number of independent parent simulations.",
        "truth_covariance_source": "caller-supplied controlled reference",
    }


__all__ = ["CovarianceMockTrial", "run_covariance_mock_campaign"]
