"""Optional CAMB adapter using the explicit physical parameter schema.

This is the preferred first physics backend in the new layer because CAMB supplies massive-neutrino
transfer functions, matter power and derived background quantities. Every scientific run should record
CAMB version, parameter mapping, nonlinear setting and output-unit convention. This adapter has not
been benchmarked against a frozen reference table in this starter release.
"""

from __future__ import annotations

import importlib.metadata

import numpy as np

from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum
from nufield.exceptions import OptionalDependencyError


class CAMBPhysicsBackend:
    """Calculate physical-unit linear or configured nonlinear matter power with CAMB."""

    def __init__(self, *, nonlinear: bool = False, accuracy_boost: float = 1.0):
        try:
            import camb
        except ImportError as exc:
            raise OptionalDependencyError(
                "CAMB is optional. Install a compatible `camb` release and see docs/PHYSICS_MODEL.md."
            ) from exc
        if accuracy_boost < 1.0 or not np.isfinite(accuracy_boost):
            raise ValueError("accuracy_boost must be finite and >= 1")
        self._camb = camb
        self.nonlinear = bool(nonlinear)
        self.accuracy_boost = float(accuracy_boost)

    @property
    def version(self) -> str:
        try:
            return importlib.metadata.version("camb")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    def _build_results(self, params: PhysicalCosmology, z_max: float, k_max_mpc_inv: float):
        camb = self._camb
        cp = camb.CAMBparams()
        cp.set_cosmology(
            H0=params.H0_km_s_Mpc,
            omk=params.omega_k,
            ombh2=params.omega_b_h2,
            omch2=params.omega_cdm_h2,
            mnu=params.sum_mnu_ev,
            nnu=params.N_eff,
            num_massive_neutrinos=params.n_massive_species,
            neutrino_hierarchy=params.neutrino_hierarchy,
            tau=params.tau_reio,
            TCMB=params.T_cmb_K,
        )
        cp.InitPower.set_params(
            As=params.A_s,
            ns=params.n_s,
            nrun=params.alpha_s,
            pivot_scalar=params.k_pivot_Mpc_inv,
        )
        cp.set_dark_energy(w=params.w0, wa=params.wa, dark_energy_model="ppf")
        cp.set_matter_power(redshifts=[float(z_max)], kmax=float(k_max_mpc_inv))
        cp.NonLinear = camb.model.NonLinear_both if self.nonlinear else camb.model.NonLinear_none
        cp.WantCls = False
        cp.Accuracy.AccuracyBoost = self.accuracy_boost
        cp.Accuracy.lAccuracyBoost = self.accuracy_boost
        cp.Accuracy.lSampleBoost = self.accuracy_boost
        return camb.get_results(cp)

    def matter_power(
        self,
        params: PhysicalCosmology,
        z: float,
        k_mpc_inv: np.ndarray,
    ) -> MatterPowerSpectrum:
        """Evaluate P_m(k,z) in Mpc^3 on k in Mpc^-1 using CAMB interpolation."""
        k = np.asarray(k_mpc_inv, dtype=float)
        if z < 0 or not np.isfinite(z):
            raise ValueError("z must be finite and non-negative")
        if (
            k.ndim != 1
            or k.size < 3
            or np.any(k <= 0)
            or np.any(~np.isfinite(k))
            or np.any(np.diff(k) <= 0)
        ):
            raise ValueError("k_mpc_inv must be a positive, strictly increasing 1-D array")
        # CAMB's internal matter-power k_max convention is Mpc^-1. Here the return interpolator is
        # requested with k_hunit=False and hubble_units=False, giving P in Mpc^3.
        results = self._build_results(params, z, float(k[-1]) * 1.1)
        interp = results.get_matter_power_interpolator(
            nonlinear=self.nonlinear,
            hubble_units=False,
            k_hunit=False,
            kmax=float(k[-1]) * 1.05,
            zmax=max(float(z), 0.01),
        )
        pk = np.asarray(interp.P(float(z), k), dtype=float)
        if np.any(~np.isfinite(pk)) or np.any(pk <= 0):
            raise ValueError(
                "CAMB returned invalid matter power; check settings and supported k,z range"
            )
        metadata = (
            ("backend", "CAMB"),
            ("version", self.version),
            ("nonlinear", str(self.nonlinear)),
            ("units", "k: 1/Mpc; P: Mpc^3"),
            ("neutrino_hierarchy", params.neutrino_hierarchy),
            ("sum_mnu_ev", repr(params.sum_mnu_ev)),
        )
        return MatterPowerSpectrum(
            k,
            pk,
            float(z),
            linear=not self.nonlinear,
            source=f"CAMB {self.version}",
            metadata=metadata,
        )

    _COMPONENT_TO_CAMB_VARIABLE = {
        "total": "delta_tot",
        "cb": "delta_nonu",
        "cdm": "delta_cdm",
        "baryon": "delta_baryon",
        "nu": "delta_nu",
    }

    def matter_power_component(
        self,
        params: PhysicalCosmology,
        z: float,
        k_mpc_inv: np.ndarray,
        *,
        component: str = "total",
    ) -> MatterPowerSpectrum:
        """Return an auto-spectrum for one CAMB density component in physical units.

        ``total`` is CAMB's CDM+baryon+massive-neutrino total-matter variable; ``cb`` is
        CAMB's ``delta_nonu`` CDM+baryon variable. Distinguishing those definitions matters for
        massive-neutrino cosmologies because neutrino perturbations are scale-dependent. The density
        transfer variables are synchronous-gauge outputs in CAMB's transfer-variable convention.

        Nonlinear component spectra are intentionally disabled except for ``total``: a generic
        nonlinear correction is not automatically a separately calibrated nonlinear model for CDM,
        baryons or massive neutrinos. This method is an auto-spectrum API, not a cross-spectrum API.
        """
        try:
            variable = self._COMPONENT_TO_CAMB_VARIABLE[component]
        except KeyError as exc:
            raise ValueError(
                f"component must be one of {sorted(self._COMPONENT_TO_CAMB_VARIABLE)}"
            ) from exc
        if self.nonlinear and component != "total":
            raise ValueError(
                "nonlinear component spectra are not enabled; use linear spectra for cb/cdm/baryon/nu "
                "or request the total-matter nonlinear spectrum"
            )
        k = np.asarray(k_mpc_inv, dtype=float)
        if z < 0 or not np.isfinite(z):
            raise ValueError("z must be finite and non-negative")
        if (
            k.ndim != 1
            or k.size < 3
            or np.any(k <= 0)
            or np.any(~np.isfinite(k))
            or np.any(np.diff(k) <= 0)
        ):
            raise ValueError(
                "k_mpc_inv must be a positive, strictly increasing 1-D array with >=3 points"
            )
        if component == "nu" and params.sum_mnu_ev <= 0:
            raise ValueError("component='nu' requires nonzero massive-neutrino mass")
        results = self._build_results(params, z, float(k[-1]) * 1.1)
        interp = results.get_matter_power_interpolator(
            nonlinear=self.nonlinear,
            var1=variable,
            var2=variable,
            hubble_units=False,
            k_hunit=False,
            kmax=float(k[-1]) * 1.05,
            zmax=max(float(z), 0.01),
        )
        pk = np.asarray(interp.P(float(z), k), dtype=float)
        if np.any(~np.isfinite(pk)) or np.any(pk <= 0):
            raise ValueError(
                f"CAMB returned non-positive or non-finite {component} auto-power; "
                "check component support and solver accuracy"
            )
        metadata = (
            ("backend", "CAMB"),
            ("version", self.version),
            ("component", component),
            ("camb_transfer_variable", variable),
            ("density_gauge", "synchronous-gauge CAMB transfer variable"),
            ("nonlinear", str(self.nonlinear)),
            ("units", "k: 1/Mpc; P: Mpc^3"),
            ("neutrino_hierarchy", params.neutrino_hierarchy),
            ("sum_mnu_ev", repr(params.sum_mnu_ev)),
            ("power_definition", "auto-spectrum; component-specific density contrast"),
        )
        return MatterPowerSpectrum(
            k,
            pk,
            float(z),
            linear=not self.nonlinear,
            source=f"CAMB {self.version} ({component})",
            metadata=metadata,
        )

    def derived_parameters(self, params: PhysicalCosmology) -> dict[str, float | str]:
        """Return CAMB derived parameters, including drag sound horizon where available."""
        results = self._build_results(params, 0.0, 0.5)
        raw = results.get_derived_params()
        aliases = {
            "r_drag_mpc": "rdrag",
            "z_drag": "zdrag",
            "z_star": "zstar",
            "theta_star_100": "thetastar",
            "age_gyr": " age",
        }
        out: dict[str, float | str] = {"backend": "CAMB", "version": self.version}
        for target, source in aliases.items():
            if source.strip() in raw:
                out[target] = float(raw[source.strip()])
        return out

    def angular_power_spectra(
        self, params: PhysicalCosmology, lmax: int = 2500
    ) -> dict[str, np.ndarray]:
        """Optional CMB C_ell access method; requires CAMB CMB transfer computation.

        This method builds a separate settings object because ``matter_power`` disables CMB spectra.
        Returned spectra follow CAMB's standard raw C_ell convention; verify unit scaling before use.
        """
        if lmax < 30:
            raise ValueError("lmax must be >= 30")
        camb = self._camb
        cp = camb.CAMBparams()
        cp.set_cosmology(
            H0=params.H0_km_s_Mpc,
            omk=params.omega_k,
            ombh2=params.omega_b_h2,
            omch2=params.omega_cdm_h2,
            mnu=params.sum_mnu_ev,
            nnu=params.N_eff,
            num_massive_neutrinos=params.n_massive_species,
            neutrino_hierarchy=params.neutrino_hierarchy,
            tau=params.tau_reio,
            TCMB=params.T_cmb_K,
        )
        cp.InitPower.set_params(
            As=params.A_s, ns=params.n_s, nrun=params.alpha_s, pivot_scalar=params.k_pivot_Mpc_inv
        )
        cp.set_dark_energy(w=params.w0, wa=params.wa, dark_energy_model="ppf")
        cp.set_for_lmax(lmax, lens_potential_accuracy=1)
        results = camb.get_results(cp)
        cls = results.get_cmb_power_spectra(cp, CMB_unit="muK")
        return {key: np.asarray(value, dtype=float) for key, value in cls.items()}


__all__ = ["CAMBPhysicsBackend"]
