"""Optional CLASS adapter implementing the same physical-unit interface as the CAMB adapter.

CLASS parameter mappings are intrinsically model-convention sensitive. This adapter records a mapping
manifest and uses an explicit effective-relativistic-species bookkeeping convention. It should first
be validated against CAMB on a matched suite of parameter points and accuracy settings; it is not
assumed interchangeable with CAMB merely because the outputs have the same units.
"""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from nufield.cosmology.physics.constants import TNU_TO_TGAMMA
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum
from nufield.exceptions import OptionalDependencyError


class ClassyPhysicsBackend:
    """Compute physical-unit linear/nonlinear P(k,z) through ``classy``.

    All wavenumbers are in 1/Mpc and powers in Mpc³ at the public API boundary. The CLASS Python
    method ``pk(k,z)`` takes k in 1/Mpc and returns Mpc³. This wrapper intentionally does not accept
    h/Mpc at the public boundary. It also does not offer a hidden default for the neutrino hierarchy:
    the exact eigenmasses are passed as ``m_ncdm`` when a massive species is requested.
    """

    # CLASS default T_ncdm has this ratio to T_gamma. Keeping this explicit makes the N_eff split
    # auditable and allows a future config field to override it in a versioned schema.
    DEFAULT_T_NCDM_OVER_T_CMB = TNU_TO_TGAMMA

    def __init__(
        self,
        *,
        nonlinear: bool = False,
        k_max_mpc_inv: float = 2.0,
        z_max: float = 3.0,
        accuracy_boost: float = 1.0,
        class_constructor: Callable[[], Any] | None = None,
    ) -> None:
        if not math.isfinite(k_max_mpc_inv) or k_max_mpc_inv <= 0:
            raise ValueError("k_max_mpc_inv must be positive and finite")
        if not math.isfinite(z_max) or z_max < 0:
            raise ValueError("z_max must be finite and non-negative")
        if not math.isfinite(accuracy_boost) or accuracy_boost < 1:
            raise ValueError("accuracy_boost must be finite and >= 1")
        if class_constructor is None:
            try:
                from classy import Class
            except ImportError as exc:
                raise OptionalDependencyError(
                    "CLASS Python bindings are optional. Install a compatible `classy` build and check its Fortran toolchain."
                ) from exc
            class_constructor = Class
        self._class_ctor = class_constructor
        self.nonlinear = bool(nonlinear)
        self.k_max_mpc_inv = float(k_max_mpc_inv)
        self.z_max = float(z_max)
        self.accuracy_boost = float(accuracy_boost)

    @property
    def version(self) -> str:
        try:
            return importlib.metadata.version("classy")
        except importlib.metadata.PackageNotFoundError:
            return "injected-or-unknown"

    @classmethod
    def effective_n_ur(cls, n_eff: float, n_massive_species: int) -> float:
        """Compute residual ultra-relativistic species for the specified ncdm temperature.

        One thermal ncdm species contributes (T_ncdm/T_standard)^4 to the conventional massless
        neutrino count. The default temperature ratio is used here, so this convention must match the
        target CLASS release and the temperature fields placed in the settings dictionary.
        """
        t_standard = (4.0 / 11.0) ** (1.0 / 3.0)
        n_per_species = (cls.DEFAULT_T_NCDM_OVER_T_CMB / t_standard) ** 4
        n_ur = float(n_eff) - int(n_massive_species) * n_per_species
        if n_ur < -1e-8:
            raise ValueError(
                "N_eff is too small for the requested number of massive CLASS species at the configured temperature"
            )
        return max(n_ur, 0.0)

    def settings_for(
        self, params: PhysicalCosmology, z: float, k_max_mpc_inv: float | None = None
    ) -> dict[str, Any]:
        """Return a deterministic CLASS settings dictionary for review and provenance."""
        if not math.isfinite(z) or z < 0 or z > self.z_max:
            raise ValueError(f"z must be in [0, {self.z_max:g}]")
        kmax = self.k_max_mpc_inv if k_max_mpc_inv is None else float(k_max_mpc_inv)
        if not math.isfinite(kmax) or kmax <= 0:
            raise ValueError("k_max_mpc_inv must be positive and finite")
        masses = params.neutrino_masses_ev
        if params.sum_mnu_ev <= 0:
            n_ncdm = 0
            masses_for_class: list[float] = []
            n_ur = float(params.N_eff)
        else:
            n_ncdm = len(masses)
            masses_for_class = [float(m) for m in masses]
            n_ur = self.effective_n_ur(params.N_eff, n_ncdm)
        settings: dict[str, Any] = {
            "h": params.h,
            "omega_b": params.omega_b_h2,
            "omega_cdm": params.omega_cdm_h2,
            "T_cmb": params.T_cmb_K,
            "N_ur": n_ur,
            "output": "mPk",
            "P_k_max_1/Mpc": max(kmax * 1.1, 0.01),
            "z_max_pk": max(z, 0.01),
            "A_s": params.A_s,
            "n_s": params.n_s,
            "alpha_s": params.alpha_s,
            "k_pivot": params.k_pivot_Mpc_inv,
            "tau_reio": params.tau_reio,
            "Omega_k": params.omega_k,
            "N_ncdm": n_ncdm,
            "Omega_Lambda": 0.0,
            "w0_fld": params.w0,
            "wa_fld": params.wa,
            "fluid_equation_of_state": "CLP",
            "tol_background_integration": 1e-8 / self.accuracy_boost,
            "tol_thermo_integration": 1e-4 / self.accuracy_boost,
            "k_per_decade_for_pk": int(round(50 * self.accuracy_boost)),
        }
        if n_ncdm:
            settings["m_ncdm"] = ",".join(f"{m:.12g}" for m in masses_for_class)
            settings["T_ncdm"] = ",".join([f"{self.DEFAULT_T_NCDM_OVER_T_CMB:.12g}"] * n_ncdm)
            settings["deg_ncdm"] = ",".join(["1"] * n_ncdm)
        if self.nonlinear:
            settings["non linear"] = "halofit"
        return settings

    def matter_power(
        self,
        params: PhysicalCosmology,
        z: float,
        k_mpc_inv: np.ndarray,
    ) -> MatterPowerSpectrum:
        """Return CLASS P(k,z) in Mpc³ for k in 1/Mpc with robust cleanup."""
        k = np.asarray(k_mpc_inv, float)
        if (
            k.ndim != 1
            or k.size < 3
            or np.any(~np.isfinite(k))
            or np.any(k <= 0)
            or np.any(np.diff(k) <= 0)
        ):
            raise ValueError(
                "k_mpc_inv must be positive, finite, increasing and contain >=3 samples"
            )
        if k[-1] > self.k_max_mpc_inv * (1 + 1e-10):
            raise ValueError("requested k exceeds configured k_max_mpc_inv")
        cosmo = self._class_ctor()
        try:
            settings = self.settings_for(params, float(z), float(k[-1]))
            cosmo.set(settings)
            cosmo.compute()
            pk = np.asarray([cosmo.pk(float(ki), float(z)) for ki in k], float)
            if np.any(~np.isfinite(pk)) or np.any(pk <= 0):
                raise ValueError("CLASS returned invalid matter power")
        finally:
            # CLASS can allocate substantial native memory; always release it on success or exception.
            for method in ("struct_cleanup", "empty"):
                fn = getattr(cosmo, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
        meta = (
            ("backend", "CLASS/classy"),
            ("version", self.version),
            ("units", "k: 1/Mpc; P: Mpc^3"),
            ("nonlinear", str(self.nonlinear)),
            ("sum_mnu_ev", repr(params.sum_mnu_ev)),
            ("neutrino_hierarchy", params.neutrino_hierarchy),
            ("n_massive_species", str(params.n_massive_species)),
            ("accuracy_boost", repr(self.accuracy_boost)),
            ("parameter_mapping", "ncdm eigenmasses; residual N_eff assigned to N_ur"),
        )
        return MatterPowerSpectrum(
            k,
            pk,
            float(z),
            linear=not self.nonlinear,
            source=f"CLASS/classy {self.version}",
            metadata=meta,
        )


__all__ = ["ClassyPhysicsBackend"]
