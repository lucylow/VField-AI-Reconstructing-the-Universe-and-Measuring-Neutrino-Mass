"""Linear redshift-space clustering with Alcock--Paczynski remapping diagnostics.

This is a transparent Kaiser + Gaussian Finger-of-God model for controlled experiments. It is not a
full EFT-of-LSS likelihood and omits nonlinear mode coupling, wide-angle effects, window mixing, fiber
assignment, and survey-specific nuisance terms. The model is useful as a unit-tested interface between
linear theory spectra, RSD multipoles and AP geometry before adding a more advanced likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss, legval

from nufield.cosmology.physics.spectrum import MatterPowerSpectrum


@dataclass(frozen=True, slots=True)
class APKaiserParameters:
    bias: float
    growth_rate: float
    alpha_perp: float = 1.0
    alpha_parallel: float = 1.0
    sigma_v_mpc: float = 0.0
    shot_noise_mpc3: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.bias,
                self.growth_rate,
                self.alpha_perp,
                self.alpha_parallel,
                self.sigma_v_mpc,
                self.shot_noise_mpc3,
            ),
            float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("RSD/AP parameters must be finite")
        if (
            self.bias <= 0
            or self.growth_rate < 0
            or self.alpha_perp <= 0
            or self.alpha_parallel <= 0
        ):
            raise ValueError("bias and AP dilations must be positive; growth rate non-negative")
        if self.sigma_v_mpc < 0 or self.shot_noise_mpc3 < 0:
            raise ValueError("velocity dispersion and shot noise must be non-negative")


@dataclass(frozen=True, slots=True)
class RSDMultipoles:
    k_mpc_inv: np.ndarray
    p0_mpc3: np.ndarray
    p2_mpc3: np.ndarray
    p4_mpc3: np.ndarray
    parameters: APKaiserParameters
    redshift: float
    quadrature_order: int
    model: str = "Kaiser + Gaussian FoG + AP remapping"
    caveats: tuple[str, ...] = (
        "Phenomenological large-scale model; not a full nonlinear RSD likelihood.",
        "Survey-window convolution and covariance must be applied separately.",
        "Bias, growth rate, AP dilations, velocity dispersion and shot noise are not fitted by this function.",
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "k_mpc_inv": self.k_mpc_inv.tolist(),
            "p0_mpc3": self.p0_mpc3.tolist(),
            "p2_mpc3": self.p2_mpc3.tolist(),
            "p4_mpc3": self.p4_mpc3.tolist(),
            "parameters": self.parameters.__dict__
            if hasattr(self.parameters, "__dict__")
            else {
                "bias": self.parameters.bias,
                "growth_rate": self.parameters.growth_rate,
                "alpha_perp": self.parameters.alpha_perp,
                "alpha_parallel": self.parameters.alpha_parallel,
                "sigma_v_mpc": self.parameters.sigma_v_mpc,
                "shot_noise_mpc3": self.parameters.shot_noise_mpc3,
            },
            "redshift": self.redshift,
            "quadrature_order": self.quadrature_order,
            "model": self.model,
            "caveats": list(self.caveats),
            "units": {"k": "Mpc^-1", "P": "Mpc^3"},
        }


def ap_remap_observed_modes(
    k_observed_mpc_inv: np.ndarray,
    mu_observed: np.ndarray,
    alpha_perp: float,
    alpha_parallel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map observed (k, μ) to true coordinates and return the volume Jacobian.

    The convention is α_perp = D_M / D_M,fid and α_parallel = H_fid/H, with any common r_d
    convention included in these ratios. Then P_obs(k_obs, μ_obs) = P_true(k_true, μ_true)/(α_perp² α_parallel).
    Inputs are broadcast with NumPy semantics.
    """
    k, mu = np.broadcast_arrays(
        np.asarray(k_observed_mpc_inv, float), np.asarray(mu_observed, float)
    )
    if np.any(~np.isfinite(k + mu)) or np.any(k <= 0) or np.any(np.abs(mu) > 1):
        raise ValueError("k must be positive finite and |mu|<=1")
    if (
        not math.isfinite(alpha_perp)
        or not math.isfinite(alpha_parallel)
        or min(alpha_perp, alpha_parallel) <= 0
    ):
        raise ValueError("AP dilation parameters must be positive and finite")
    ratio_sq = (alpha_perp / alpha_parallel) ** 2
    radial = np.sqrt(1.0 + mu**2 * (ratio_sq - 1.0))
    k_true = k / alpha_perp * radial
    mu_true = mu * (alpha_perp / alpha_parallel) / radial
    jac = np.full_like(k_true, 1.0 / (alpha_perp**2 * alpha_parallel))
    return k_true, mu_true, jac


def kaiser_fog_power(
    k_mpc_inv: np.ndarray,
    mu: np.ndarray,
    linear_power: MatterPowerSpectrum,
    parameters: APKaiserParameters,
) -> np.ndarray:
    """Evaluate the true-coordinate redshift-space power under Kaiser + Gaussian FoG assumptions."""
    k, m = np.broadcast_arrays(np.asarray(k_mpc_inv, float), np.asarray(mu, float))
    if np.any(k <= 0) or np.any(~np.isfinite(k + m)) or np.any(np.abs(m) > 1):
        raise ValueError("k must be positive and |mu|<=1")
    p = linear_power.evaluate(k)
    amplitude = parameters.bias + parameters.growth_rate * m * m
    damping = np.exp(-((k * m * parameters.sigma_v_mpc) ** 2))
    return amplitude**2 * p * damping + parameters.shot_noise_mpc3


def observed_redshift_space_power(
    k_observed_mpc_inv: np.ndarray,
    mu_observed: np.ndarray,
    linear_power: MatterPowerSpectrum,
    parameters: APKaiserParameters,
) -> np.ndarray:
    """Evaluate P_obs including AP remapping; linear_power must cover mapped true k values."""
    k_true, mu_true, jac = ap_remap_observed_modes(
        k_observed_mpc_inv, mu_observed, parameters.alpha_perp, parameters.alpha_parallel
    )
    return jac * kaiser_fog_power(k_true, mu_true, linear_power, parameters)


def redshift_space_multipoles(
    k_observed_mpc_inv: np.ndarray,
    linear_power: MatterPowerSpectrum,
    parameters: APKaiserParameters,
    *,
    quadrature_order: int = 64,
) -> RSDMultipoles:
    r"""Integrate P_ell(k)= (2ell+1)/2 ∫_{-1}^{1}dμ P(k,μ)L_ell(μ) for ell=0,2,4."""
    k = np.asarray(k_observed_mpc_inv, float)
    if (
        k.ndim != 1
        or len(k) < 3
        or np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or np.any(np.diff(k) <= 0)
    ):
        raise ValueError("k_observed must be a positive increasing vector with >=3 values")
    if quadrature_order < 12:
        raise ValueError("quadrature_order must be >=12")
    mu, weights = leggauss(int(quadrature_order))
    kk = k[:, None]
    mm = mu[None, :]
    power = observed_redshift_space_power(kk, mm, linear_power, parameters)
    outputs = []
    for ell in (0, 2, 4):
        leg = legval(mm.ravel(), [0] * ell + [1]).reshape(mm.shape)
        multipole = (2 * ell + 1) / 2 * np.sum(power * leg * weights[None, :], axis=1)
        outputs.append(multipole)
    return RSDMultipoles(
        k.copy(), *outputs, parameters, linear_power.redshift, int(quadrature_order)
    )


def analytic_kaiser_multipoles(
    k_mpc_inv: np.ndarray,
    linear_power: MatterPowerSpectrum,
    bias: float,
    growth_rate: float,
    shot_noise_mpc3: float = 0.0,
) -> RSDMultipoles:
    """Closed-form zero-FoG/no-AP Kaiser multipoles, a reference for quadrature tests."""
    k = np.asarray(k_mpc_inv, float)
    if bias <= 0 or growth_rate < 0 or shot_noise_mpc3 < 0:
        raise ValueError("bias positive, growth rate and shot noise non-negative required")
    p = linear_power.evaluate(k)
    p0 = (bias**2 + 2 * bias * growth_rate / 3 + growth_rate**2 / 5) * p + shot_noise_mpc3
    p2 = (4 * bias * growth_rate / 3 + 4 * growth_rate**2 / 7) * p
    p4 = 8 * growth_rate**2 / 35 * p
    return RSDMultipoles(
        k,
        p0,
        p2,
        p4,
        APKaiserParameters(bias, growth_rate, shot_noise_mpc3=shot_noise_mpc3),
        linear_power.redshift,
        0,
        model="analytic Kaiser multipoles",
        caveats=(
            "Analytic linear Kaiser reference; no AP effect or FoG damping.",
            "Not a full survey likelihood; apply the survey window and covariance.",
        ),
    )


__all__ = [
    "APKaiserParameters",
    "RSDMultipoles",
    "ap_remap_observed_modes",
    "kaiser_fog_power",
    "observed_redshift_space_power",
    "redshift_space_multipoles",
    "analytic_kaiser_multipoles",
]
