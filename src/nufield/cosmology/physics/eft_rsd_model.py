"""Large-scale effective-theory-style redshift-space galaxy power templates.

The models here expose physically interpretable nuisance terms while retaining a deliberately limited
large-scale form. They are not a replacement for a survey-specific one-loop EFT-of-LSS implementation:
loop integrals, bias renormalization, IR resummation, survey window folding and validated nuisance priors
are not included. Use the outputs only over scales where the truncation is supported by simulation tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss, legval


@dataclass(frozen=True, slots=True)
class EFTLikeRSDParameters:
    """Parameters of a controlled k²-corrected Kaiser model in physical Mpc units."""

    bias_linear: float
    growth_rate: float
    counterterm_0_mpc2: float = 0.0
    counterterm_2_mpc2: float = 0.0
    counterterm_4_mpc2: float = 0.0
    velocity_dispersion_mpc: float = 0.0
    shot_noise_mpc3: float = 0.0
    stochastic_constant_mpc3: float = 0.0
    stochastic_k2_mpc5: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.bias_linear,
            self.growth_rate,
            self.counterterm_0_mpc2,
            self.counterterm_2_mpc2,
            self.counterterm_4_mpc2,
            self.velocity_dispersion_mpc,
            self.shot_noise_mpc3,
            self.stochastic_constant_mpc3,
            self.stochastic_k2_mpc5,
        )
        if any(not math.isfinite(float(v)) for v in values):
            raise ValueError("all RSD model parameters must be finite")
        if self.bias_linear <= 0 or self.velocity_dispersion_mpc < 0 or self.shot_noise_mpc3 < 0:
            raise ValueError(
                "linear bias positive; velocity dispersion and shot noise non-negative required"
            )
        if self.stochastic_constant_mpc3 < 0 or self.stochastic_k2_mpc5 < 0:
            raise ValueError("stochastic additive components must be non-negative in this baseline")


def eft_like_redshift_space_power(
    k_mpc_inv: float | np.ndarray,
    mu: float | np.ndarray,
    linear_matter_power,
    parameters: EFTLikeRSDParameters,
    *,
    require_positive: bool = True,
) -> np.ndarray:
    r"""Return a large-scale RSD power template with k² counterterms.

    P_g^s(k,μ) = (b₁+f μ²)² P_L(k) exp[-(k μ σ_v)²]
      - 2 k² P_L(k) (c₀ + c₂ μ² + c₄ μ⁴)
      + P_shot + a₀ + a₂ k².

    The displacement parameter σ_v is phenomenological and measured in Mpc. Counterterms c_i are
    measured in Mpc², while a₂ is Mpc⁵ so a₂ k² has units Mpc³. The formula is truncated at O(k²);
    if it becomes negative, that is a diagnostic that the parameterization is being extrapolated too
    far or the nuisance coefficients are unsuitable, not a reason to clip the prediction silently.
    """
    k, m = np.broadcast_arrays(np.asarray(k_mpc_inv, float), np.asarray(mu, float))
    p = parameters
    if np.any(~np.isfinite(k + m)) or np.any(k <= 0) or np.any(np.abs(m) > 1):
        raise ValueError("k must be finite positive and mu in [-1,1]")
    pl = np.asarray(linear_matter_power(k), float)
    if pl.shape != k.shape or np.any(~np.isfinite(pl)) or np.any(pl <= 0):
        raise ValueError(
            "linear_matter_power(k) must return positive finite P(k) matching broadcast k"
        )
    mu2 = m * m
    k2 = k * k
    kaiser = (p.bias_linear + p.growth_rate * mu2) ** 2 * pl
    fog = np.exp(-((k * m * p.velocity_dispersion_mpc) ** 2))
    counter = (
        -2
        * k2
        * pl
        * (p.counterterm_0_mpc2 + p.counterterm_2_mpc2 * mu2 + p.counterterm_4_mpc2 * mu2**2)
    )
    stochastic = p.shot_noise_mpc3 + p.stochastic_constant_mpc3 + p.stochastic_k2_mpc5 * k2
    result = kaiser * fog + counter + stochastic
    if require_positive and np.any(result <= 0):
        raise ValueError(
            "EFT-like truncated model predicts non-positive power; restrict k range or revisit nuisance parameters"
        )
    return result


def eft_like_power_multipoles(
    k_mpc_inv: np.ndarray,
    linear_matter_power,
    parameters: EFTLikeRSDParameters,
    *,
    multipoles: tuple[int, ...] = (0, 2, 4),
    n_mu: int = 64,
) -> dict[int, np.ndarray]:
    """Integrate multipoles of the EFT-like model with Gauss-Legendre quadrature."""
    k = np.asarray(k_mpc_inv, float)
    if k.ndim != 1 or k.size < 1 or np.any(k <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError("k_mpc_inv must be a positive strictly increasing vector")
    if not multipoles or any(l < 0 or l % 2 for l in multipoles) or n_mu < 8:
        raise ValueError("require non-negative even multipoles and n_mu>=8")
    mu, weights = leggauss(n_mu)
    out = {int(ell): np.zeros(k.size) for ell in multipoles}
    for i, kval in enumerate(k):
        p_mu = eft_like_redshift_space_power(kval, mu, linear_matter_power, parameters)
        for ell in multipoles:
            coeff = np.zeros(ell + 1)
            coeff[ell] = 1
            out[int(ell)][i] = (2 * ell + 1) / 2 * np.sum(weights * p_mu * legval(mu, coeff))
    return out


def scale_dependent_linear_bias(
    k_mpc_inv: np.ndarray,
    b1: float,
    b_laplacian_mpc2: float,
) -> np.ndarray:
    r"""Return b(k)=b₁+b_{∇²δ}k² as the leading derivative-bias expansion."""
    k = np.asarray(k_mpc_inv, float)
    if (
        np.any(~np.isfinite(k))
        or np.any(k <= 0)
        or b1 <= 0
        or not math.isfinite(b1 + b_laplacian_mpc2)
    ):
        raise ValueError("k must be positive, b1 positive, and coefficients finite")
    return b1 + b_laplacian_mpc2 * k * k


def effective_theory_scale_diagnostic(
    k_mpc_inv: np.ndarray, *, k_nonlinear_mpc_inv: float, target_fraction: float = 0.2
) -> dict[str, object]:
    """Flag bins where a k² expansion may not be perturbatively small."""
    k = np.asarray(k_mpc_inv, float)
    if k.ndim != 1 or np.any(k <= 0) or np.any(~np.isfinite(k)):
        raise ValueError("k must be a finite positive vector")
    if (
        not math.isfinite(k_nonlinear_mpc_inv)
        or k_nonlinear_mpc_inv <= 0
        or not 0 < target_fraction < 1
    ):
        raise ValueError("k_nonlinear positive and target_fraction in (0,1) required")
    expansion = (k / k_nonlinear_mpc_inv) ** 2
    ok = expansion < target_fraction
    return {
        "k_nonlinear_mpc_inv": float(k_nonlinear_mpc_inv),
        "target_fraction": float(target_fraction),
        "estimated_expansion_parameter": expansion.tolist(),
        "within_nominal_k2_domain": ok.tolist(),
        "maximum_k_within_domain_mpc_inv": float(np.max(k[ok])) if np.any(ok) else None,
        "warning": "The scale cut is heuristic; calibrate it against simulation residuals for the selected tracer and model.",
    }


__all__ = [
    "EFTLikeRSDParameters",
    "eft_like_redshift_space_power",
    "eft_like_power_multipoles",
    "scale_dependent_linear_bias",
    "effective_theory_scale_diagnostic",
]
