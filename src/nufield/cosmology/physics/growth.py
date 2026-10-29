"""Scale-independent linear growth utilities with explicit massive-neutrino caveats."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from nufield.cosmology.physics.background import BackgroundCosmology


def _solve_growth(
    background: BackgroundCosmology,
    z_eval: np.ndarray,
    *,
    a_initial: float,
    allow_massive_neutrino_approximation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    params = background.params
    if params.sum_mnu_ev > 0 and not allow_massive_neutrino_approximation:
        raise ValueError(
            "scale-independent growth omits neutrino free-streaming; pass "
            "allow_massive_neutrino_approximation=True only for exploratory checks, or use a "
            "Boltzmann backend for scale-dependent growth"
        )
    if not 0 < a_initial < 1:
        raise ValueError("a_initial must be in (0, 1)")
    x0, x1 = math.log(a_initial), 0.0

    def coeffs(x: float) -> tuple[float, float]:
        a = math.exp(x)
        z = 1 / a - 1
        # d ln H / d ln a = -(1+z) d ln H/dz, estimated with a stable centered stencil.
        dz = max(1e-5, 1e-4 * (1 + z))
        zlo = max(0.0, z - dz)
        zhi = z + dz
        if zlo == z:
            dlnh_dz = (math.log(float(background.H(zhi))) - math.log(float(background.H(z)))) / (
                zhi - z
            )
        else:
            dlnh_dz = (math.log(float(background.H(zhi))) - math.log(float(background.H(zlo)))) / (
                zhi - zlo
            )
        dlnh_dlna = -(1 + z) * dlnh_dz
        e2 = float(background.e2(z))
        # In the permitted approximation all present-day matter clusters as pressureless matter.
        # This is intentionally not a scale-dependent massive-neutrino growth solution.
        omega_m_a = background.omega_m0 * a**-3 / e2
        return 2.0 + dlnh_dlna, 1.5 * omega_m_a

    def ode(x: float, y: np.ndarray) -> tuple[float, float]:
        d, dp = float(y[0]), float(y[1])
        friction, grav = coeffs(x)
        return dp, -friction * dp + grav * d

    initial = np.array([a_initial, a_initial], dtype=float)
    sol_today = solve_ivp(
        ode, (x0, x1), initial, rtol=2e-7, atol=1e-10, dense_output=True, max_step=0.04
    )
    if not sol_today.success:
        raise RuntimeError(f"growth ODE failed: {sol_today.message}")
    norm = float(sol_today.sol(0.0)[0])
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("growth normalization is invalid")
    a_eval = 1.0 / (1.0 + z_eval)
    x_eval = np.log(a_eval)
    result = sol_today.sol(x_eval)[0] / norm
    # Growth rate f = d ln D / d ln a.
    deriv = sol_today.sol(x_eval)[1]
    rate = deriv / sol_today.sol(x_eval)[0]
    return np.asarray(result), np.asarray(rate)


def growth_factor_scale_independent(
    background: BackgroundCosmology,
    z: float | np.ndarray,
    *,
    a_initial: float = 1e-3,
    allow_massive_neutrino_approximation: bool = False,
) -> np.ndarray:
    """Return D(z), normalized to D(0)=1, in smooth-DE scale-independent approximation."""
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0) or np.any(~np.isfinite(zz)):
        raise ValueError("redshift must be finite and non-negative")
    return _solve_growth(
        background,
        zz,
        a_initial=a_initial,
        allow_massive_neutrino_approximation=allow_massive_neutrino_approximation,
    )[0]


def growth_rate_f(
    background: BackgroundCosmology,
    z: float | np.ndarray,
    *,
    a_initial: float = 1e-3,
    allow_massive_neutrino_approximation: bool = False,
) -> np.ndarray:
    """Return f=d ln D/d ln a under the same approximation as ``growth_factor_scale_independent``."""
    zz = np.asarray(z, dtype=float)
    if np.any(zz < 0) or np.any(~np.isfinite(zz)):
        raise ValueError("redshift must be finite and non-negative")
    return _solve_growth(
        background,
        zz,
        a_initial=a_initial,
        allow_massive_neutrino_approximation=allow_massive_neutrino_approximation,
    )[1]


def growth_rate_approx_omega_m(
    background: BackgroundCosmology, z: float, gamma: float = 0.55
) -> float:
    """Return heuristic f≈Ω_m(z)^gamma for rough ΛCDM sanity checks only."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    fractions = background.density_fractions(z)
    omega_m_z = fractions["baryon"] + fractions["cdm"] + fractions["nu_massive"]
    return float(omega_m_z**gamma)


def fsigma8(
    background: BackgroundCosmology,
    z: float,
    sigma8_0: float,
    *,
    allow_massive_neutrino_approximation: bool = False,
) -> float:
    """Compute f(z) σ8(z) using scale-independent growth; not a neutrino-aware precision prediction."""
    if sigma8_0 <= 0:
        raise ValueError("sigma8_0 must be positive")
    d = float(
        growth_factor_scale_independent(
            background, z, allow_massive_neutrino_approximation=allow_massive_neutrino_approximation
        )
    )
    f = float(
        growth_rate_f(
            background, z, allow_massive_neutrino_approximation=allow_massive_neutrino_approximation
        )
    )
    return f * sigma8_0 * d


__all__ = [
    "growth_factor_scale_independent",
    "growth_rate_f",
    "growth_rate_approx_omega_m",
    "fsigma8",
]
