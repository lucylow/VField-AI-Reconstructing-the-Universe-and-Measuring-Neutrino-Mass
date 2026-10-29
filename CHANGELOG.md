# Changelog
## 0.5.0 — AI research feature expansion
- Added an optional AI research package with grouped splits, probabilistic networks, GP baselines, calibration, conformal prediction, OOD/drift tools, explainability, active learning, and reproducible benchmark workflows.
- Added a `nufield-ai` CLI and synthetic GP-versus-neural example.
- All examples remain synthetic; they do not establish cosmological or neutrino-mass constraints.


## νField v0.3.0 — expanded cosmology physics concepts

This release adds a dedicated, solver-neutral physics layer and equation-level tests. It expands the project from a toy inference scaffold into a modular environment for developing and testing cosmological calculations, with explicit approximation labels and validation gates.

### Physical concepts and numerical methods

- Added explicit physical cosmology parameter conventions, homogeneous CPL background expansion, radial/transverse/angular/luminosity distances, thermal Fermi–Dirac neutrino energy density and pressure, hierarchy bookkeeping, and thermal momentum diagnostics.
- Added linear matter-spectrum data contracts, physical-unit conversions, power-spectrum variance, solver interfaces and CAMB adapter metadata.
- Added BAO distance/damping diagnostics, linear Kaiser RSD multipoles, simple AP mapping, second-order EdS perturbation-theory baselines, halo mass-function/bias approximations, neutrino suppression diagnostics, and explicit fixed-Ωcb/fixed-Ωm variation conventions.
- Added weak-lensing kernels, tomographic Limber projection, redshift-distribution validation, Gaussian photo-z migration matrices, survey noise models and selection/projection utilities.
- Added periodic-box power estimation, CIC mass assignment/window treatment, reconstruction cross-power/propagator/stochastic residual metrics, Gaussian initial-field utilities, and scale-validity checks.
- Added Gaussian nuisance-template fitting/marginalization, finite-mock covariance diagnostics, a Sellentin–Heavens-style multivariate-t likelihood, Fisher forecasts with covariance derivatives, parameter-identifiability analysis, posterior interval coverage/PIT diagnostics, and solver spectrum-comparison utilities.
- Added scientific artifact manifests with SHA-256 fingerprints and a machine-readable physics validation suite; exposed `nufield-physics validate`.

### Verification and limits

- Added tests for the new redshift/photo-z, thermal phase-space, nuisance marginalization, finite-covariance likelihood, reconstruction-metric, window-propagation, solver-spectrum comparison and validation APIs.
- The full suite was run in the release workspace with `PYTHONPATH=src pytest -q`: **184 tests passed**. Optional CAMB/CLASS workflows need their solver dependencies and must be checked against a pinned independent reference grid.
- The physics package contains 5,647 lines of Python source (~113 pages at 50 code lines/page), within a repository of more than 16,000 Python source lines. These are modular implementation building blocks, not a claim that every module is production-grade.
- No Quijote, DESI or CAMELS data are bundled. No real-data inference or neutrino-mass constraint is claimed. The default simulator remains synthetic; several analytic kernels are approximations and explicitly documented as such.

## Research scaffold v0.2.0 — dataset-first model selection

This release extends the νField starter into a more complete research-engineering scaffold for choosing cosmological data products and testing inference models.

### Dataset intake and provenance

- Added exact-product scientific contracts and target-column/unit conversion checks.
- Added source-role planning that separates mass-target training, classical baselines, survey robustness, baryonic domain-shift, cross-suite validation, and observed-data application.
- Added HDF5 metadata profiling, generic chunked HDF5/Zarr reads, particle coordinate-to-periodic-grid utilities, gridded-field batch intake, field-summary generation, source hashing, and output manifests.
- Added mass-grid coverage, parameter-design diagnostics, independent parent-group splitting, split fingerprints and hard gates against treating observed data as labelled simulations.

### Model selection and inference

- Added a conditional mixture-density network (MDN) posterior for bounded summary parameters, with train-only target/feature scaling and posterior sampling in physical coordinates.
- Added nested, parent-group-safe fitting/early-stopping and separate outer validation model selection; the final test set remains locked by default.
- Added budget-bounded MDN grid search, a discrete classical mass-grid Gaussian-summary likelihood, prediction commands, posterior archive comparison and model cards.
- Added candidate-model eligibility gates, compute-budget helpers, OOD/domain-shift utilities, and validation/SBC/posterior-predictive helpers.

### Documentation and verification

- Added dataset/model decision playbook, MDN guide, streaming intake guide, classical baseline guide, and reproducible comparison guide.
- The repository test suite is intended to run offline with `PYTHONPATH=src pytest -q`; optional `sbi` workflows need the `sbi` extra and have not been certified by merely passing the base tests.

### Limitations

This release does not include a downloaded Quijote/DESI/CAMELS dataset or a production neutrino-aware forward model. The default simulator remains synthetic. Source-family recommendations are conditional; the exact product, label columns, units, cosmology, product terms and independent simulation semantics must be verified before a real-data training run. No outputs from the prototype constitute a physical neutrino-mass measurement.

## Physics research suite v0.4.0 — expanded diagnostics and validation

This release expands the cosmology physics layer with a further collection of modular models,
measurement diagnostics, and validation tools intended to make assumptions explicit and testable.

### Physics and observable additions

- Added hierarchy-aware neutrino mass-eigenvalue and laboratory-observable bookkeeping, plus total-matter versus CDM+baryon power diagnostics and neutrino-response analysis.
- Added growth-equation residuals, scale-dependent-growth comparisons, AP remapping and RSD multipoles, BAO-template robustness studies, tree-level bispectrum diagnostics, excursion-set field topology, and scale-domain audits.
- Added theory-error covariance construction, finite mock covariance and shrinkage diagnostics, posterior predictive checks, parameter-response campaigns, parameter-degeneracy analysis, and first-order systematic parameter-bias diagnostics.
- Added lensing-kernel resolution checks, neutrino phase-space sampler checks, continuity-equation diagnostics, mock-ensemble validation, survey-window stability, and covariance-mock campaigns.
- Added physics-aware dataset contracts, galaxy-count/RSD forward-model utilities, neutrino-mass mass-grid coverage checks, parameter-mapping audits, and physics snapshot artifacts.
- Expanded command-line workflows for sigma8 integration, field topology, thermal-neutrino continuity, hierarchy-aware mass priors, supplied growth-history checks, and explicit research-gate assessment.

### Verification

- Full offline test suite: **211 tests passed** in the release workspace (`PYTHONPATH=src pytest -q`).
- Python compile check passed for the physics package.
- The `nufield-physics --help` command successfully listed all 11 available commands.
- Physics-package inventory at release time: **123 Python files and approximately 14,705 lines**. A page-count analogy depends on formatting and is not a software-quality metric.

### Scientific limitations

No external cosmology dataset is bundled or used to claim a real neutrino-mass constraint. The default νField simulator remains synthetic. EdS, Kaiser, Limber, tree-level perturbation-theory, local-bias, and thermal-background calculations are limited models with the regime assumptions stated in their documentation. CAMB/CLASS outputs must be compared against pinned reference grids before use in research results. Unit tests verify internal behaviour and numerical invariants; they are not a substitute for scientific validation against trusted solvers, simulations, and observational likelihoods.
