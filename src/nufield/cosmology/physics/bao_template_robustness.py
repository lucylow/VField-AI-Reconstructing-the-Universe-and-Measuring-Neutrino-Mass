"""Diagnose BAO-wiggle sensitivity to a chosen smooth-spectrum filter.

The smooth/no-wiggle split is not unique. This module measures how descriptive diagnostics change under
several pre-declared Savitzky--Golay smoothing settings and makes that dependence visible. It does not
produce a likelihood-ready BAO template, reconstruction correction, or survey-window-convolved model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nufield.cosmology.physics.bao_diagnostics import extract_bao_wiggles
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class BAOSmoothingTrial:
    window_length: int
    polyorder: int
    wiggle_rms: float
    median_abs_wiggle: float
    zero_crossings: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BAOTemplateRobustnessReport:
    redshift: float
    k_range_mpc_inv: tuple[float, float]
    trials: tuple[BAOSmoothingTrial, ...]
    rms_range: tuple[float, float]
    fractional_rms_span: float
    zero_crossing_count_range: tuple[int, int]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "redshift": self.redshift,
            "k_range_mpc_inv": list(self.k_range_mpc_inv),
            "trials": [
                {
                    "window_length": t.window_length,
                    "polyorder": t.polyorder,
                    "wiggle_rms": t.wiggle_rms,
                    "median_abs_wiggle": t.median_abs_wiggle,
                    "zero_crossings_mpc_inv": list(t.zero_crossings),
                }
                for t in self.trials
            ],
            "rms_range": list(self.rms_range),
            "fractional_rms_span": self.fractional_rms_span,
            "zero_crossing_count_range": list(self.zero_crossing_count_range),
            "caveats": list(self.caveats),
        }


def _zero_crossings(k: np.ndarray, values: np.ndarray, mask: np.ndarray) -> tuple[float, ...]:
    idx = np.flatnonzero(mask)
    out: list[float] = []
    if len(idx) < 2:
        return ()
    for i0, i1 in zip(idx[:-1], idx[1:]):
        # The requested k interval is contiguous in the native sorted grid; skip holes if any.
        if i1 != i0 + 1:
            continue
        y0, y1 = values[i0], values[i1]
        if y0 == 0:
            out.append(float(k[i0]))
        elif y0 * y1 < 0:
            frac = abs(y0) / (abs(y0) + abs(y1))
            out.append(float(np.exp(np.log(k[i0]) * (1 - frac) + np.log(k[i1]) * frac)))
    return tuple(out)


def assess_bao_smoothing_robustness(
    spectrum: MatterPowerSpectrum,
    *,
    k_min_mpc_inv: float,
    k_max_mpc_inv: float,
    window_lengths: tuple[int, ...] = (21, 31, 41, 51),
    polynomial_orders: tuple[int, ...] = (2, 3),
) -> BAOTemplateRobustnessReport:
    """Calculate wiggle amplitude and sign-crossing diagnostics over pre-declared smoothers."""
    if not (0 < k_min_mpc_inv < k_max_mpc_inv):
        raise ValueError("require 0 < k_min < k_max")
    windows = tuple(sorted(set(int(w) for w in window_lengths)))
    orders = tuple(sorted(set(int(p) for p in polynomial_orders)))
    if len(windows) < 2 or not orders or any(w < 5 or w % 2 == 0 for w in windows):
        raise ValueError("use at least two odd smoothing windows >=5 and one polynomial order")
    if any(p < 1 or p >= max(windows) for p in orders):
        raise ValueError("polynomial orders must be >=1 and smaller than every candidate window")
    trials: list[BAOSmoothingTrial] = []
    for window in windows:
        for order in orders:
            if order >= window:
                continue
            diag = extract_bao_wiggles(
                spectrum,
                k_min_mpc_inv=k_min_mpc_inv,
                k_max_mpc_inv=k_max_mpc_inv,
                window_length=window,
                polyorder=order,
            )
            mask = (diag.k_mpc_inv >= k_min_mpc_inv) & (diag.k_mpc_inv <= k_max_mpc_inv)
            w = diag.fractional_wiggle[mask]
            trials.append(
                BAOSmoothingTrial(
                    window,
                    order,
                    diag.rms_fractional_wiggle,
                    float(np.median(np.abs(w))),
                    _zero_crossings(diag.k_mpc_inv, diag.fractional_wiggle, mask),
                )
            )
    if len(trials) < 2:
        raise ValueError("not enough valid smoothing trials")
    rms = np.asarray([t.wiggle_rms for t in trials])
    rmin, rmax = float(rms.min()), float(rms.max())
    span = float((rmax - rmin) / max(float(np.median(rms)), np.finfo(float).tiny))
    nzero = [len(t.zero_crossings) for t in trials]
    return BAOTemplateRobustnessReport(
        spectrum.redshift,
        (float(k_min_mpc_inv), float(k_max_mpc_inv)),
        tuple(trials),
        (rmin, rmax),
        span,
        (min(nzero), max(nzero)),
        (
            "The smooth spectrum is a filter-defined diagnostic, not a physical no-wiggle transfer model.",
            "A small variation across these filters does not establish BAO parameter unbiasedness.",
            "Likelihood use requires a published template, survey window, covariance and reconstruction validation.",
        ),
    )


__all__ = ["BAOSmoothingTrial", "BAOTemplateRobustnessReport", "assess_bao_smoothing_robustness"]
