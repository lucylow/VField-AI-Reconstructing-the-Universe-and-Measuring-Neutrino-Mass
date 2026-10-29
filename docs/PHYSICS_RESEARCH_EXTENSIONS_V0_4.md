# νField physics research extensions — v0.4.0

## Purpose and scientific boundary

This document describes the additional physics and diagnostics modules added to the νField research scaffold. The goal is to turn hidden modeling assumptions into versioned, executable contracts and to make numerical/likelihood failure modes easier to find before attempting a real inference analysis.

νField is **not** a precision Boltzmann solver, an N-body code, a calibrated massive-neutrino simulation suite, or a validated DESI likelihood. The default field generator remains synthetic. The new modules are composable building blocks that can be connected to verified simulation and solver products. Numerical tests demonstrate internal consistency for the tested inputs; they do not prove physical correctness against observations.

## A. New scientific workflow

A defensible field-level analysis needs distinct validation layers. Do not collapse these layers into one `validate=True` flag.

1. **Definitions and units:** cosmological parameter conventions, component definitions, redshift, gauge, `k` units, `P(k)` units, Fourier normalization, and box/grid metadata.
2. **Forward-model verification:** compare background and linear perturbation outputs against a pinned external solver for the exact cosmology and convention.
3. **Observable verification:** check estimator normalization, selection/window propagation, covariance construction, and estimator response using controlled realizations.
4. **Inference validation:** use parent-independent simulations, coverage/rank diagnostics, posterior predictive checks, nuisance variations, and a locked test set.
5. **Scientific interpretation:** only after the preceding layers should parameter constraints or comparisons with real observations be discussed.

The `physics_pipeline_gates.py` interface records evidence against declared requirements. A gate report is a checklist of evidence, not a guarantee that the evidence itself is scientifically correct; reviewers must inspect sources and assumptions.

## B. Neutrino masses and component-aware spectra

### B.1 Keep mass conventions separate

The sum of the three active mass eigenvalues is

\[
\Sigma m_\nu = m_1 + m_2 + m_3.
\]

Normal and inverted ordering map a lightest eigenmass to the other two using oscillation mass-squared splittings. That mapping is not the same as selecting an arbitrary continuous value for each eigenmass independently. The `neutrino_mass_observables.py` and `neutrino_hierarchy_prior.py` modules keep those concepts separate and make the prior coordinate explicit: uniform in mass sum, uniform in lightest mass, or a log prior in lightest mass.

The prior is a modeling decision. A uniform prior in `Σmν` is not equivalent to a uniform prior in the lightest eigenmass, and neither should be presented as a parameterization-free result. Values for oscillation splittings and mixing parameters must be pinned to a cited external reference version for any research run. The defaults in the scaffold are illustrative until that reference is documented.

Laboratory observables are included as bookkeeping diagnostics, not as an attempt to combine cosmological and laboratory likelihoods. For example, the effective beta-decay mass is

\[
m_\beta^2 = \sum_i |U_{ei}|^2 m_i^2,
\]

where the weights are electron-flavor mixing probabilities. The neutrinoless double-beta effective mass is a phase-dependent complex-amplitude observable; absent the unknown phases, report a range over stated assumptions, not one definitive value.

### B.2 Distinguish `P_cb` from total-matter `P_m`

Massive neutrinos make component definitions essential. A total-matter perturbation and a cold-dark-matter-plus-baryon perturbation have different transfer functions. The new component diagnostics compare spectra only on an explicitly shared `k` grid and redshift, and report ratios rather than silently identifying one spectrum as the other.

A component ratio is meaningful only when both inputs use compatible cosmology, gauge, normalization, Fourier convention, redshift and linear/nonlinear status. An external CAMB adapter maps component names to the relevant transfer variables, but the installed CAMB version and output settings must be recorded and checked against reference data. In particular, do not assume `P_cb` can be reconstructed from `P_total` by a universal constant factor.

### B.3 Interpret mass-response diagnostics cautiously

A finite-difference derivative such as

\[
\partial P(k,z)/\partial \Sigma m_\nu
\]

is only useful if the spectra were generated with controlled, documented changes in neutrino mass while all other nuisance and cosmological parameters were held fixed or varied according to a documented design. A response grid should save its parameter values, random seeds, solver version, component name, redshift, units and interpolation procedure. Edge points need one-sided derivatives. Step stability should be measured rather than presumed.

The heuristic suppression scaling `ΔP/P ≈ -8 fν` is provided only as a rough sanity-check mnemonic in a restricted regime. It is not a universal prediction, and must not be used as a fitted neutrino-mass likelihood.

## C. Background, growth and perturbations

### C.1 Background continuity

For a homogeneous component with density `ρ` and pressure `p`, conservation gives

\[
\frac{d\rho}{d\ln a} + 3(\rho+p)=0.
\]

The neutrino continuity test evaluates this expression numerically from the thermal background routine. It can catch a convention or derivative error, but finite differences can be sensitive to step size and interpolation. It does not test the neutrino perturbation hierarchy.

### C.2 Growth-equation validation

For scale-independent linear growth in general relativity, using derivatives with respect to `ln a`, the commonly used matter-plus-smooth-component equation is

\[
D'' + \left(2 + \frac{d\ln H}{d\ln a}\right)D'
- \frac{3}{2}\Omega_m(a)D = 0.
\]

The growth validator accepts a supplied history and measures a normalized residual. An Einstein–de Sitter analytic solution `D(a) ∝ a` is used as a known limiting test. With massive neutrinos, clustering can be scale dependent; a single `D(a)` and a scale-independent equation do not represent the full response. Use these checks for the declared approximation and compare with a Boltzmann solver for the actual model.

### C.3 Tree-level perturbation theory

The `tree_bispectrum_model.py` module provides the standard Einstein–de Sitter second-order kernel as a diagnostic baseline. The tree-level bispectrum is a leading-order perturbative approximation, not a general nonlinear model. It requires closed wavevector triangles and should only be compared in a regime where perturbation theory is justified. It is not calibrated against a simulation suite by default.

## D. Power spectra and normalization

### D.1 Units are part of the data

The code uses explicit physical-unit types: `k` in `Mpc⁻¹` and `P(k)` in `Mpc³`, unless an adapter says otherwise. In commonly used `h`-scaled notation,

\[
k_{h/\mathrm{Mpc}} = k_{\mathrm{Mpc}^{-1}}/h,
\qquad
P_{(\mathrm{Mpc}/h)^3} = h^3 P_{\mathrm{Mpc}^3}.
\]

Conversions must be paired; applying a conversion to only one axis changes the physical meaning of the data. `cosmology_units_audit.py`, `cosmology_units.py`, `parameter_mapping_audit.py`, and dataset contracts are intended to catch that class of error.

### D.2 `σ8` integration

The variance smoothed with a spherical top-hat of radius `R` is

\[
\sigma_R^2 = \frac{1}{2\pi^2}\int_0^\infty dk\,k^2P(k)W^2(kR),
\]

with

\[
W(x)=3\frac{\sin x-x\cos x}{x^3}.
\]

The `sigma8_validation.py` routine integrates only over the supplied spectrum support. Numerical convergence over the available grid does **not** prove that missing low-`k` or high-`k` tails are negligible. Record the available support, interpolation assumptions, integration scheme, linear/nonlinear spectrum choice, and whether the radius is `8 Mpc/h` or another convention. For normalizing a CMB-era linear spectrum, state the redshift at which the spectrum is evaluated.

### D.3 FFT power-estimator conventions

A measured Fourier-space power spectrum depends on box volume, discrete Fourier normalization, bin averaging, mesh assignment and aliasing. The estimator module documents its normalization and supports a window correction. For a field assigned to a mesh with assignment window `W_mesh(k)`, power deconvolution involves `|W_mesh(k)|²`; correction becomes unstable close to zeros of the window. Do not apply aggressive deconvolution above a declared safe fraction of the mesh Nyquist frequency.

Before interpreting `P(k)`, test at least: (i) an exactly uniform field, (ii) white-noise fields with expected flat power under the chosen convention, (iii) a Gaussian field with a supplied target spectrum, and (iv) box-size/grid refinement. Save the volume, grid, assignment scheme, bin edges, shot-noise convention, and Fourier convention in the result manifest.

## E. Redshift-space clustering and AP distortions

In the plane-parallel approximation, the redshift-space coordinate along the line of sight is

\[
s_\parallel = x_\parallel + \frac{v_\parallel}{aH(a)}.
\]

The galaxy-survey forward model implements this mapping for comoving coordinates and peculiar velocities in matching velocity units. It is not a wide-angle treatment and does not model all selection effects.

The linear Kaiser model is

\[
P_s(k,\mu) = (b + f\mu^2)^2P_m(k),
\]

with optional Gaussian fingers-of-God damping in the approximating implementation. Its multipoles are computed by angular integration. The AP mapping remaps observed `(k, μ)` to a fiducial/theory coordinate system using transverse and radial dilation parameters. When using these expressions in an inference, ensure the data estimator and model both use the same multipole normalization, fiducial cosmology convention, bin averaging and window convolution.

AP remapping plus Kaiser RSD is a controlled validation baseline. Nonlinear scale-dependent bias, wide-angle geometry, redshift evolution, fiber collisions, survey masks and nonlinear velocity distributions require separate modeling or explicit scale cuts.

## F. Survey windows, lensing and topology

### F.1 Window functions

The observed data vector may be modeled as `d_obs = W d_true + noise`, where `W` represents the complete linearized window/estimator action. The window-validation utilities diagnose rank, impulse response and inverse amplification. Deconvolution is usually ill-conditioned; if regularization is applied, its strength and its induced bias/variance tradeoff must be reported. The actual survey window is not supplied automatically by νField.

### F.2 Weak-lensing kernels

The lensing kernel utilities check normalization, sign, redshift ordering and numerical resolution for a declared source distribution. The Limber approximation projects a 3-D spectrum into angular spectra under assumptions about angular scale and radial kernels. A resolution study is useful but cannot prove the Limber approximation is valid. Compare with a non-Limber/reference calculation when using low multipoles or broad kernels where the approximation may fail.

### F.3 Field topology

The `field_topology.py` functions report excursion-set occupancy and voxel-grid topological diagnostics for thresholds. They are grid/topology observables; exposed voxel face area is not a continuum surface-area estimator without discretization analysis. Use periodic-boundary tests and vary grid resolution, smoothing and threshold convention. Do not call the discretized Euler characteristic a calibrated cosmological statistic until it has been checked against analytic fields and known simulation examples.

## G. Covariance, likelihoods and systematic bias

### G.1 Sample covariance

For `N_s` mock vectors and `p` bins, the unbiased sample covariance is

\[
\widehat C = \frac{1}{N_s-1}\sum_{i=1}^{N_s}(d_i-\bar d)(d_i-\bar d)^T.
\]

The inverse sample covariance is noisy. The Hartlap factor

\[
\alpha = \frac{N_s-p-2}{N_s-1}
\]

is relevant only when its assumptions hold and `N_s > p+2`. It is not a general correction for any covariance model. Shrinkage can improve conditioning, but the target and shrinkage strength should be specified. The Sellentin–Heavens likelihood is a different finite-mock treatment; do not combine corrections mechanically or apply both methods without deriving the likelihood being used.

### G.2 Theory error

The theory-error module allows a correlated fractional-error model in log `k` and addition of explicitly named covariance components. This is a modeling choice, not a measured uncertainty. State its amplitude, correlation length, redshift dependence, and whether it is calibrated from simulation residuals or chosen as a sensitivity envelope. Check positive definiteness before inversion and report sensitivity to the assumed theory-error covariance.

### G.3 Parameter bias from misspecification

A common first-order approximation is

\[
\Delta\theta \simeq F^{-1}J^TC^{-1}\Delta d,
\quad F=J^TC^{-1}J,
\]

where `J` is a derivative matrix, `C` is covariance, and `Δd` is a specified systematic residual. The `parameter_bias_diagnostics.py` utilities evaluate this local linear approximation. It is not reliable for large shifts, strongly nonlinear parameter mappings, hard prior boundaries, singular Fisher matrices, or a systematic that changes the likelihood shape.

Use finite-difference response campaigns and step stability to audit `J`. The derivative step must be varied, bounds respected, and solver noise compared with the response amplitude. Degenerate or poorly constrained parameter combinations should be flagged rather than hidden by a numerical inverse.

## H. Reconstruction and data generation

The galaxy survey forward-model helpers convert a matter field into a phenomenological biased tracer, apply redshift-space displacement, generate expected voxel counts, apply a selection/completeness map, and draw Poisson counts. These routines provide a transparent place to test masks, noise and estimator plumbing. They are not substitutes for a calibrated halo occupation distribution or realistic mock catalog production.

The particle-mesh code is a grid-assignment utility, not a gravitational N-body solver. The BAO reconstruction code is a controlled simplified implementation; it should be compared with a published/reference algorithm on the same mocks before it is used for a scientific BAO result. Reconstruction benchmarks should record initial-condition truth, output field, smoothing scale, mesh, boundary conditions, random seed and metric definitions.

Initial-condition validation checks reproducibility and summary statistics of Gaussian Fourier fields. It is not enough to generate a Gaussian field with a plausible power spectrum and declare a simulated nonlinear universe. The full chain must represent gravitational evolution, neutrino effects, tracer formation, survey selection and the observation estimator at the fidelity demanded by the target claim.

## I. Solver convergence and reference comparisons

The `solver_convergence.py` and `parameter_mapping_audit.py` modules are intended to compare reproducible snapshots or spectra from a reference and candidate solver. For a valid comparison, pin:

- solver package and version/commit;
- all cosmological parameters, including neutrino masses and hierarchy assumptions;
- gauge and variable definition;
- linear vs nonlinear prescription;
- `k` and power units and the `h` convention;
- redshift and requested component (total matter, CDM+baryon, neutrino, etc.);
- precision/accuracy settings and interpolation method;
- common support and interpolation tolerance.

A same-code rerun at higher accuracy checks numerical stability; it does not independently verify the physical model. External solver agreement checks implementation differences but can still share assumptions or errors. Use an independent analytic limit where available as well.

For mass-sensitive spectra, compare `P_total(k,z)` and `P_cb(k,z)` separately. In massive-neutrino cosmologies they have distinct physical meanings. Verify component semantics from the installed solver documentation and a pinned example before interpreting either one as the matter spectrum used by a particular survey likelihood.

## J. Suggested executable research recipes

### J.1 Run all internal tests

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m compileall -q src/nufield/cosmology/physics
PYTHONPATH=src python -m nufield.cosmology.physics.cli --help
```

### J.2 Validate a growth table

Create an NPZ with arrays `scale_factor`, `growth`, `H_km_s_mpc`, and `omega_m_clustering`. The growth validator expects a monotonic scale-factor history and computes a finite-difference residual against the supplied background. Check the residual convergence when the input history is resampled. Do not test only the exact analytic EdS sequence and assume the same discretization is accurate on a realistic background.

### J.3 Compute `sigma8` from a solver result

Produce an NPZ with `k_mpc_inv`, `pk_mpc3`, `redshift` and `linear` from a *verified* external solver configuration, then run:

```bash
nufield-physics sigma8 --spectrum outputs/physics/power_camb.npz --h 0.67 \
  --output outputs/physics/sigma8.json
```

Review the `k` support and integration-convergence report. Compare the result with the solver's independently reported normalization only after matching spectrum definition, redshift and units.

### J.4 Evaluate a theory covariance envelope

Construct a data vector, a statistical covariance and an explicit theory-error model. Evaluate several amplitudes and correlation lengths rather than reporting one arbitrary theory-error configuration. Save input arrays, covariance decomposition, eigenvalue diagnostics and likelihood values together.

### J.5 Gate a result before real-data interpretation

The JSON evidence map supplied to `gate-assess` should record a boolean or structured entry for each named requirement, plus artifact path, commit, solver version, data release and reviewer notes when available. Missing evidence should block the relevant stage. Do not mark an item complete merely because a helper function exists in the repository.

## K. Versioned experiment schemas

Each published numerical experiment should save a manifest with at least:

- source repository commit, package version, Python and dependency versions;
- random seeds and all simulation parent IDs;
- cosmological parameters and parameterization/prior definitions;
- solver version, settings, accuracy options, gauge and variable semantics;
- dataset source/release/URI, exact product identifier and access-date/hash where permitted;
- field component, redshift, grid/box, coordinate and Fourier units;
- model architecture and weights hash; train/validation/test parent splits;
- likelihood, covariance, window, scale cuts, nuisance parameters and theory-error assumptions;
- diagnostics, warnings, failed checks and an artifact hash.

Do not store private credentials, access tokens, or restricted raw data in the manifest. Data access terms need to be respected independently of code licensing.

## L. What counts as a new scientific result?

The modules in this release make it easier to run controlled tests, but code volume is not itself novelty. A useful research question should have a falsifiable outcome, such as whether a specific reconstruction improves a neutrino-sensitive summary at fixed error budget, how much a realistic survey window changes calibration, or where a fast approximation becomes biased relative to a reference solver.

A convincing report should show matched baselines, held-out simulations, uncertainty calibration, robustness to nuisance assumptions, solver/reference validation, compute cost, and negative findings. Do not announce a narrower neutrino-mass constraint as a scientific improvement if it comes with undercoverage or a mismatch between the simulated and observed likelihood.

## M. External data and solver policy

- **Quijote massive-neutrino simulation products:** candidate primary controlled training data only after exact product/parameter-grid verification. Confirm what the product varies and which field/components are supplied. Do not assume the number of outputs equals the number of independent cosmologies.
- **DESI mocks and data:** use for survey/statistical robustness and late-stage observed-data analysis. Mocks are not automatically a mass-varied supervised training grid; data do not supply labelled truth per observed sky.
- **CAMELS:** useful for certain baryonic and astrophysical domain shifts. Check the exact suite and varied parameters; standard products with neutrino mass fixed do not train a nonzero-neutrino mass estimator.
- **CAMB/CLASS:** use pinned versions and settings; compare a shared grid, units, components, and redshifts. Optional adapters are plumbing until this comparison is recorded.

The source code does not bundle those external data. The user must retrieve them under the source's terms, document exact product identity, and use contracts to prevent accidental semantic mixing.

## N. Practical next milestone

The best next milestone is not another neural architecture. It is a reproducible, small controlled experiment based on a verified massive-neutrino simulation product:

1. build and audit a product adapter;
2. verify background/linear spectra against the pinned solver;
3. establish power-spectrum, BAO and covariance baselines;
4. generate independent parent-group splits;
5. run neural and classical inference on the same training/validation/test groups;
6. calibrate posterior coverage and run known systematics shifts;
7. publish the negative and positive results plus the exact artifact manifest.

Only after that experimental path is stable should a multimodal lensing extension or LLM-based algorithm discovery become a principal research claim.
