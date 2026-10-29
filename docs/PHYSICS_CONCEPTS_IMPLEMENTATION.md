# νField Physics Concepts Implementation Guide

**Version:** 0.3.0 development line  
**Scope:** equations, numerical contracts, approximation boundaries and validation plans for the cosmology physics package.

> **Scientific status.** νField is research scaffolding. The internal equations and software contracts are being tested, but this document does not certify a precision cosmology pipeline or a neutrino-mass constraint. A real constraint requires a validated Boltzmann/forward model, a defensible likelihood, calibrated covariance, independent simulations and a review of nuisance/systematic assumptions.

## 1. What this layer is for

The earlier νField application layer was designed to make a reproducible inference experiment run end to end. This physics layer deliberately has a different goal: make the scientific assumptions inspectable and separable. A field-generating demo is not a cosmological N-body model; a neural posterior is not automatically Bayesian-calibrated; a map called “lensing” is not necessarily a weak-lensing observable; and a smaller interval is not necessarily an improvement if bias and coverage have not been checked.

The package therefore separates six things:

1. **Background model:** homogeneous expansion, distances, thermal neutrino energy density.
2. **Linear-theory interface:** transfer functions and matter power spectrum with explicit units and redshift metadata.
3. **Observable model:** BAO distances, linear RSD multipoles, weak-lensing Limber projections, selection functions, noise and windows.
4. **Field utilities:** Gaussian initial fields, particle-to-mesh assignment, periodic-box power estimates and field-reconstruction metrics.
5. **Inference/forecast:** likelihoods, nuisance marginalization, parameter derivatives, Fisher matrices and finite-mock diagnostics.
6. **Validation:** equation-level checks, external solver comparison, held-out posterior calibration and physics-scope gates.

These levels are not interchangeable. The `PhysicsScope` record must be preserved in experiment manifests and cannot be upgraded just because a function returns finite numbers.

## 2. Parameter and unit contract

`PhysicalCosmology` stores Hubble rate as `H0_km_s_Mpc`, density fractions as dimensionless `omega_*` (uppercase Omega convention), mass as `sum_mnu_ev`, primordial scalar amplitude as `A_s`, and wavenumbers/power spectra at the public interface in `Mpc^-1` and `Mpc^3`. This is intentionally not the same convention as tables expressed in `h/Mpc` and `(Mpc/h)^3`.

Conversions use

\[
k_{\rm phys}=h\,k_h,\qquad P_{\rm phys}=P_h/h^3,
\]

with the inverse operations implemented in `cosmological_units.py`. Note that the symbol `omega_b` in this project means \(\Omega_b\), while `omega_b_h2` means \(\Omega_bh^2\). A report should state the convention; do not rely on informal variable names.

The configuration in `configs/physics/reference_lcdm.yaml` is an illustrative reference point, not a fit. All derived outputs should preserve the input mapping and solver metadata, including neutrino hierarchy, redshift, linear/nonlinear flag, k range, numerical settings and code version.

## 3. Homogeneous expansion

The background implementation evaluates

\[
E^2(z)=\frac{H^2(z)}{H_0^2},\qquad H(z)=H_0E(z).
\]

For CPL dark energy,

\[
w(a)=w_0+w_a(1-a),\qquad
\frac{\rho_{\rm de}(z)}{\rho_{\rm de,0}}=(1+z)^{3(1+w_0+w_a)}
\exp\!\left[-\frac{3w_a z}{1+z}\right].
\]

The implementation includes photons, residual massless radiation, baryons, cold dark matter, curvature, the CPL component, and a homogeneous Fermi--Dirac integral for explicit massive-neutrino species. The closure condition is evaluated at z=0. The thermal neutrino component is handled separately from its late-time matter approximation because its equation of state changes between relativistic and non-relativistic regimes.

Comoving radial distance is

\[
D_C(z)=c\int_0^z\frac{dz'}{H(z')},
\]

while transverse comoving distance applies the curvature mapping. Angular-diameter and luminosity distances follow the usual distance duality expressions. Background consistency checks include `E(0)=1`, positivity of H, monotonicity of radial comoving distance and component-fraction closure.

**Limit:** these calculations are background only. They do not evolve metric perturbations, anisotropic stress, photon-baryon acoustic physics, massive-neutrino perturbations or nonlinear gravitational clustering.

## 4. Thermal massive neutrinos

For one thermal species, the homogeneous energy density is computed from

\[
\rho_\nu(z)=\frac{T_\nu^4}{\pi^2(\hbar c)^3}
\int_0^\infty dq\,\frac{q^2\sqrt{q^2+(m_\nu/T_\nu)^2}}{e^q+1},
\]

where \(q=p/T_\nu\), with the temperature redshifting as \(T_\nu\propto 1+z\) after standard electron-positron annihilation. Pressure is computed from the matching momentum integral. The ratio wν=Pν/ρν should approach 1/3 for a massless species and fall toward zero for a species that is non-relativistic today. These are equation-level checks.

The mass-hierarchy helper solves a reduced mapping between the total mass and three eigenmasses, using explicitly declared representative mass-squared splittings. A scientific run must pin the oscillation-splitting convention to the chosen global-fit source. “Degenerate” masses remain an approximation; it is not the same model as the normal or inverted hierarchy.

The thermal momentum sampler draws a homogeneous Fermi--Dirac momentum magnitude and isotropic direction. It does **not** supply the initial perturbative bulk velocity, density-velocity cross-correlations, neutrino hierarchy-dependent transfer functions, neutrino particle weights or a validated N-body force prescription. Never describe this sampler alone as complete neutrino initial conditions.

## 5. Transfer functions and matter power

The primordial curvature spectrum is parameterized by an amplitude and spectral tilt, with optional running. The matter spectrum interface uses

\[
\Delta^2(k)=\frac{k^3P(k)}{2\pi^2},\qquad
\sigma^2(R)=\int d\ln k\,\Delta^2(k)W^2(kR),
\]

where \(W(x)=3(\sin x-x\cos x)/x^3\). Every integral operates over available samples only; it does not silently extrapolate. The `MatterPowerSpectrum` validates finite positive P and sorted positive k, records redshift and linear/nonlinear tags, and interpolates log P against log k.

Precision linear transfer and power calculations should use a Boltzmann solver. The optional CAMB adapter returns physical-unit P(k,z) and records solver metadata. Agreement must be established against an independent fixed reference grid for a fixed parameter vector before this backend is used as a reference in automated regression. CAMB and CLASS choices must agree on all material conventions: physical densities, curvature, N_eff, massive-neutrino species mapping, dark-energy prescription, redshift, k range, nonlinear treatment and units.

The internal analytic and phenomenological spectrum generators are for software tests and exploratory sensitivity only. They are not replacements for a validated transfer function. The RBF spectral emulator is a prototype interpolator, not a production emulator with certified interpolation uncertainty.

## 6. Linear growth and perturbations

Scale-independent growth utilities are approximate smooth-component growth models. In massive-neutrino cosmologies, growth is scale dependent because neutrinos free-stream, so a single D(z) cannot represent all k. The code explicitly marks this limitation. If the scientific target is a neutrino-sensitive P(k), the transfer and growth should come from CAMB/CLASS or simulations whose neutrino dynamics and initial conditions have been validated against those codes.

The included EdS second-order kernel is a perturbation-theory baseline for a specified approximation. It should not be generalized to arbitrary dark energy, scale-dependent growth or massive-neutrino perturbations without implementing and validating those corrections. Tree-level bispectrum predictions likewise require a scale cut and covariance compatible with perturbation theory.

## 7. BAO and redshift-space distortions

The BAO module distinguishes geometric distance combinations from spectrum wiggle templates. A Gaussian damping envelope

\[
D_{\rm BAO}(k)=\exp[-k^2\Sigma_{\rm nl}^2/2]
\]

is a phenomenological damping model for the oscillatory component, not a full nonlinear BAO model. The smoothing-based wiggle extractor is a diagnostic. It is not an agreed no-wiggle template and must not be tuned on held-out results.

Linear Kaiser redshift-space multipoles are useful first baselines under specified assumptions (linear density, plane-parallel approximation, scale-independent bias/growth and no small-scale velocity damping). Actual survey analysis additionally needs the survey window, estimator response, fiber/selection effects where relevant, covariance, nuisance terms and a consistent model for the fitted scale range.

The simplified periodic-box reconstruction utility is a software/algorithm prototype. It omits some of the details required for a survey reconstruction pipeline and does not establish improved BAO constraints by itself.

## 8. Weak lensing and angular projection

The lensing utilities form a source-distribution efficiency and a convergence kernel, then project the three-dimensional matter power into angular spectra under a Limber-type approximation. Each distribution must be normalized according to its declared meaning (probability per unit redshift or number density per unit redshift); redshift units and comoving distance units must not be mixed.

The photo-z utilities calculate a Gaussian response matrix under an explicit mean-bias and scatter assumption. A real survey has a calibrated redshift response, selection function, catastrophic outlier treatment and shape/noise model. The output matrix deliberately permits probability loss beyond a finite photo-z interval rather than renormalizing all objects into selected bins.

The present Limber and tomographic implementation is a starting point for controlled forecasts. It is not a replacement for validated survey likelihoods and cannot be combined with arbitrary CMB-derived summary statistics without a joint likelihood and cross-covariance treatment.

## 9. Periodic fields and power-spectrum normalization

The periodic estimator uses the NumPy FFT convention and

\[
P(k)=\frac{V}{N_{\rm cell}^2}\left|\mathrm{FFT}[\delta]\right|^2,
\]

then averages modes in shells. It reports mode counts and Gaussian-mode standard errors. Correctness assumes a cubic periodic box with the supplied physical box size and an overdensity field. It does not automatically correct CIC assignment windows, aliasing, missing modes, survey masks or non-periodic boundaries.

The new field reconstruction diagnostics measure cross-power, propagator and stochastic residual power separately. For aligned truth and reconstruction fields, they help detect amplitude suppression or stochastic noise that a single correlation coefficient can hide. Results are only comparable when the same realization, mesh, box size, Fourier convention and k bins are used.

## 10. Windows, finite-mock covariance and nuisance models

A survey or estimator window W maps theory bandpowers to the observed basis as y=Wt; a covariance transforms as C_y=W C_t Wᵀ if this is the appropriate linear response model. The code requires the user to supply a window from a defined estimator. It does not infer the window from the data automatically.

For Gaussian independent mock covariance S estimated from N_s realizations, the Hartlap factor and Sellentin--Heavens likelihood utilities encode finite-mock effects under their assumptions. These corrections are not universal: mocks can be correlated or share phases, and a sample covariance can be singular when its dimension is too high. Prefer the finite-covariance likelihood only when the assumptions apply, and report the number and independence structure of mocks.

Linear nuisance templates are fit by generalized least squares; Gaussian-prior nuisance templates may be analytically marginalized by replacing C with C+TΛTᵀ. This is exact for linear Gaussian templates. It does not justify ignoring nonlinear nuisance parameters or correlated systematic uncertainties that are not represented in the covariance/templates.

## 11. Fisher matrices and local identifiability

The Gaussian Fisher implementation includes mean-response information, and the extended method also includes covariance derivatives:

\[
F_{ij}=\mu_{,i}^TC^{-1}\mu_{,j}+\tfrac12\mathrm{Tr}(C^{-1}C_{,i}C^{-1}C_{,j})+F^{\rm prior}_{ij}.
\]

Finite-difference step convergence is required. Fisher approximations can fail for bounded, multimodal, skewed, weakly constrained or highly non-Gaussian posteriors. The SVD of a whitened Jacobian diagnoses *local* near-degeneracies in the chosen parameter scaling and observable basis; it does not prove global identifiability.

## 12. Posterior calibration

For repeated held-out simulations with known truth, calculate nominal 68% and 95% equal-tailed interval coverage and randomized empirical PIT/rank diagnostics. Calibration is a repeated-experiment property. A well-calibrated interval on one simulation cannot be inferred from the shape of that one posterior. Calibration samples must be independent in the scientific sense: do not split catalog rows that descend from the same parent simulation into different folds.

The posterior diagnostic helper does not correct for finite calibration sample uncertainty, boundary effects from constrained priors, or hyperparameter selection on the test set. Report binomial standard errors or intervals for the coverage estimates when preparing a paper.

## 13. Validation workflow

Recommended sequence:

1. Run `nufield-physics validate` and save the JSON report.
2. Run the background and thermodynamic equation-level unit tests.
3. Use an optional solver backend at a fixed cosmology, then compare P(k,z) against a pinned reference grid in physical units.
4. Check convergence as k sampling, integration grids and solver accuracy settings change.
5. Use matched spectra at low and high neutrino mass to calculate suppression diagnostics; record whether Ω_cb, Ω_m, H0 or another quantity is held fixed.
6. Verify initial-field, CIC, power-estimator and reconstruction diagnostics against analytic synthetic fields and known Fourier modes.
7. Test photo-z probability conservation, window propagation, covariance conditioning and finite-mock assumptions.
8. Only then train the neural inference model and test posterior calibration on locked parent groups.
9. Finally add DESI-like windows and mock catalogs. Public survey mocks validate the survey pipeline but do not automatically form a neutrino-mass-spanning training set.

## 14. Dataset roles

- **Massive-neutrino simulation suite:** use an explicitly documented product where the neutrino-mass parameter varies and the cosmology/labels can be matched. Quijote documents a dedicated massive-neutrino set; verify the exact files, labels, units, initial-condition method and independence structure before production use.
- **DESI products:** use released LSS catalogs and mocks for survey-specific estimators, window/covariance validation and external robustness. Do not assume every mock has an independent neutrino mass.
- **CAMELS:** useful for baryonic and astrophysical domain-shift tests. Check the exact model suite and varying parameter list; do not treat a suite with fixed zero neutrino mass as a mass-varied training set.
- **CAMB/CLASS:** solver-generated spectra are deterministic theory calculations for a specified model, not observational training labels. Pin versions and solver settings.

## 15. Research-grade outputs

A release should contain: source commit hash; Python and dependency versions; solver version/configuration; parameter configuration and unit conventions; dataset/product IDs and SHA-256 fingerprints; train/validation/test parent IDs; random seeds; spectra and posterior artifacts; validation report; numerical settings; covariance/mask provenance; failed as well as successful experiment records; and a plain-language limitations section.

Recommended figures are (i) background E(z) and density fractions; (ii) P(k,z) comparison against a reference solver; (iii) fractional massive-neutrino spectrum ratios under matched parameter-variation conventions; (iv) finite-difference derivative convergence; (v) solver accuracy convergence; (vi) power-spectrum reconstruction transfer/correlation/stochastic residual; (vii) lensing kernels and tomographic Cℓ matrix; (viii) covariance eigenvalues and condition number; (ix) interval coverage with finite-sample error bars; and (x) accuracy-versus-compute for model comparison.

## 16. Sources and reference documentation

- CAMB documentation: <https://camb.readthedocs.io/en/stable/> — linear/nonlinear matter spectra, transfer functions, background quantities and solver configuration.
- CLASS documentation and repository: <https://class-code.net/> and <https://github.com/lesgourg/class_public> — Boltzmann code and theory outputs.
- Quijote simulations documentation: <https://quijote-simulations.readthedocs.io/> — simulation suites and dedicated massive-neutrino variants.
- DESI DR1 data documentation: <https://data.desi.lbl.gov/doc/releases/dr1/> — public LSS products and mocks.
- Sellentin & Heavens (2016), “Parameter inference with estimated covariance matrices,” MNRAS 456, L132 — multivariate-t likelihood for covariance estimated from simulations.
- Hartlap, Simon & Schneider (2007), “Why your model parameter confidences might be too optimistic,” A&A 464, 399 — inverse-covariance correction under specific assumptions.
- Eisenstein & Hu (1998), “Baryonic Features in the Matter Transfer Function,” ApJ 496, 605 — analytic transfer-function work; use only when its approximations are suitable.

## 17. Acceptance checklist

A physics change should not be merged until it has: a docstring with units and assumptions; tests for limits and invalid input; a numerical convergence test where integration or finite differences are involved; comparison against an independent method when one exists; a clear label for approximation level; and a record of whether it changes the scientific scope. New code can expand capability, but its presence alone does not authorize a stronger scientific claim.
