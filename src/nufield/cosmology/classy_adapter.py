"""Optional CLASS linear-spectrum adapter.

This adapter is a narrow integration seam, not a validated cosmology implementation. CLASS Python
bindings (`classy`) are an optional dependency and may need platform-specific compilation. Before
using this for science, check CLASS parameter conventions, neutrino hierarchy, units, version, and
reference outputs. The toy field simulator does not automatically use this adapter.
"""

from __future__ import annotations

import importlib.metadata

import numpy as np

from nufield.cosmology.backend import BackendMetadata, validate_power_output
from nufield.exceptions import OptionalDependencyError
from nufield.parameters import CosmologyParameters


class ClassyLinearPowerBackend:
    """Compute linear matter power with CLASS for explicit positive `k` values in h/Mpc."""

    def __init__(self, *, output: str = "mPk", nonlinear: bool = False, z_max: float = 2.0):
        try:
            from classy import Class
        except ImportError as exc:
            raise OptionalDependencyError(
                "CLASS bindings are optional. Install a compatible `classy` build and verify its version."
            ) from exc
        self._class_ctor = Class
        self.output = output
        self.nonlinear = bool(nonlinear)
        self.z_max = float(z_max)

    def metadata(self) -> BackendMetadata:
        try:
            version = importlib.metadata.version("classy")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return BackendMetadata(
            name="CLASS/classy",
            version=version,
            physics_level="linear-boltzmann-solver",
            k_unit="h/Mpc",
            power_unit="(Mpc/h)^3",
            neutrino_model="configured CLASS species model",
            valid_scales="must be verified for chosen CLASS settings",
            caveats=(
                "Linear transfer/power only unless nonlinear option is explicitly validated.",
                "Not connected to the default toy field generator.",
                "Parameter mapping is intentionally conservative and requires expert review.",
            ),
        )

    def linear_power(
        self, params: CosmologyParameters, z: float, k_h_mpc: np.ndarray
    ) -> np.ndarray:
        kk = np.asarray(k_h_mpc, dtype=float)
        if z < 0 or z > self.z_max:
            raise ValueError(f"z must be in [0, {self.z_max}]")
        if kk.ndim != 1 or np.any(kk <= 0) or np.any(np.diff(kk) <= 0):
            raise ValueError("k_h_mpc must be a positive, strictly increasing 1-D array")
        # This is an explicit initial mapping, not an endorsed scientific parameterization. The
        # neutrino species choices below must be reviewed against the target analysis convention.
        cosmo = self._class_ctor()
        h = params.h
        omega_b = 0.049 * (h / 0.674) ** 2
        omega_cdm = max(params.omega_m * h**2 - omega_b - params.sum_mnu_ev / 93.14, 1e-5)
        settings = {
            "h": h,
            "omega_b": omega_b,
            "omega_cdm": omega_cdm,
            "A_s": 2.1e-9 * (params.sigma8 / 0.811) ** 2,
            "n_s": params.n_s,
            "tau_reio": 0.054,
            "N_ncdm": 1,
            "m_ncdm": max(params.sum_mnu_ev, 1e-5),
            "output": "mPk",
            "P_k_max_h/Mpc": float(kk.max() * 1.05),
            "z_max_pk": max(float(z), 0.1),
        }
        if self.nonlinear:
            settings["non linear"] = "halofit"
        cosmo.set(settings)
        cosmo.compute()
        # CLASS pk(k,z) commonly accepts k in 1/Mpc. Convert the requested h/Mpc input to 1/Mpc,
        # and return P in (Mpc/h)^3. Verify these conventions against the CLASS version in use.
        p = np.asarray([cosmo.pk(float(k * h), float(z)) * h**3 for k in kk], dtype=float)
        cosmo.struct_cleanup()
        cosmo.empty()
        validate_power_output(kk, p)
        return p
