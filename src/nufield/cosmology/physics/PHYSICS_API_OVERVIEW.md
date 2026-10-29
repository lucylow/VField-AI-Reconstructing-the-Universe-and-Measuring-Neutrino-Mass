# Physics API overview

The stable starting point is the physical parameter schema plus the solver-neutral output type:

```python
from nufield.cosmology.physics.parameters import PhysicalCosmology
from nufield.cosmology.physics.background import BackgroundCosmology
from nufield.cosmology.physics.spectrum import MatterPowerSpectrum
```

Most physical quantities use physical Mpc rather than h-scaled Mpc. The type annotations and docstrings are part of the API contract. The packages under `nufield.cosmology.physics` are organized by physical or statistical task, and should not be treated as one monolithic validated code.

## Tier A — equation-tested building blocks

- `parameters.py`, `constants.py`, `cosmological_units.py`
- `background.py`, `distances.py`, `neutrinos.py`
- `spectrum.py`, `power_estimator.py`, `particle_mesh.py`
- `redshift_distributions.py`, `survey_window_convolution.py`
- `finite_covariance_likelihood.py`, `linear_nuisance.py`, `fisher.py`, `parameter_forecast.py`
- `posterior_diagnostics.py`, `physics_validation_suite.py`

## Tier B — approximations requiring explicit scale/model review

- `growth.py`, `higher_order.py`, `bao_damping.py`, `bao_reconstruction.py`
- `lensing.py`, `lensing_tomography.py`, `angular_spectra.py`
- `halo_bias.py`, `halo_mass_function.py`, `analytic_models.py`, `emulator.py`
- `free_streaming.py`, `neutrino_phase_space.py`, `neutrino_response.py`

## Tier C — backend/experiment plumbing

- `camb_backend.py`, `boltzmann_validation.py`, `artifact_manifest.py`
- `backend_validation.py`, `validity.py`, `theory_summary.py`, `cli.py`

The tier labels are an engineering aid, not a claim that every Tier A function has been externally validated. Unit tests establish internal consistency only. A research-grade result still needs reference solver comparison, a documented dataset product, survey modeling, independent calibration and a review of nuisance/likelihood assumptions.

## Tiered research extensions (v0.4.0)

The expanded release adds the following families. They are not all equivalent in maturity, and none inherits external validation just because it lives in this package.

### Component spectra, neutrino parameters and response grids

- `neutrino_mass_observables.py`: normal/inverted hierarchy bookkeeping and laboratory-observable transforms.
- `neutrino_hierarchy_prior.py`: explicit parameterization of the mass prior; the induced prior must be reported.
- `component_power_diagnostics.py`, `neutrino_suppression_diagnostics.py`: component-aware comparisons and explicitly qualified empirical responses.
- `neutrino_dataset_contracts.py`: data-product semantics, provenance and parent-independent split requirements.
- `neutrino_phase_space_validation.py`, `neutrino_background_consistency.py`: sampler and homogeneous thermal-continuity checks.

### Observables and forward models

- `redshift_space_ap.py`, `redshift_space_covariance.py`: AP remapping, linear Kaiser/FoG approximations, multipoles and Gaussian mode covariance.
- `galaxy_survey_forward_model.py`: local bias, Poisson counts, selection map, plane-parallel RSD mapping and manifest metadata.
- `lensing_kernel_validation.py`, `survey_window_validation.py`, `field_topology.py`: kernel resolution checks, window conditioning and grid-based excursion diagnostics.
- `tree_bispectrum_model.py`, `bao_template_robustness.py`: leading-order perturbative/baryon-acoustic diagnostics with explicit approximation assumptions.

### Likelihood, calibration and research audit

- `theory_covariance_likelihood.py`, `covariance_mock_campaign.py`, `theory_error_model.py`: mock covariance, finite-mock treatments, regularization and declared theory-error covariance.
- `observable_likelihood_diagnostics.py`, `posterior_predictive_physics.py`: correlation/whitening/leave-one-bin-out and posterior predictive checks.
- `parameter_response_campaign.py`, `parameter_bias_diagnostics.py`, `parameter_degeneracy.py`: derivative-step stability, first-order misspecification bias and parameter degeneracy diagnostics.
- `solver_convergence.py`, `parameter_mapping_audit.py`, `transfer_function_diagnostics.py`, `theory_domain_audit.py`: numerical stability and convention/support audits.
- `mock_ensemble_validation.py`, `initial_conditions_validation.py`, `field_reconstruction_benchmark.py`: ensemble, initial-condition and reconstruction checks.
- `physics_pipeline_gates.py`, `physics_snapshot_campaign.py`: explicit evidence gates and immutable experiment summaries.

### Additional physics concepts

- `sigma8_validation.py`: top-hat variance integration and finite-k support diagnostics.
- `scale_dependent_growth.py`, `growth_equation_validation.py`: comparison of scale-dependent growth histories and ODE residuals. Scale-independent growth equations do not capture the complete massive-neutrino perturbation problem.
- `neutrino_suppression_diagnostics.py`: descriptive diagnostics; a heuristic suppression scaling is not a likelihood.
- `cosmology_units_audit.py`, `parameter_mapping_audit.py`: explicit coordinate/unit and parameter-convention audits.

## API selection guidance

Start with one narrow observable and baseline. For example, use `MatterPowerSpectrum` plus a pinned solver result, audit its units and component, compute sigma8 on the available support, and compare against a reference value. Do not begin by combining the full set of optional components. Build one experiment manifest and one failure-report for each validation level, and only add model complexity when an ablation shows it is necessary.

### Redshift drift

`redshift_drift_velocity_cm_s_per_year(background, z)` returns the homogeneous Sandage–Loeb
spectroscopic velocity drift in **cm s⁻¹ yr⁻¹**, using `c [H0 − H(z)/(1+z)]` with explicit
Mpc-to-km and year-to-second conversions. It returns zero at z=0 and rejects negative/non-finite
redshifts. The result excludes peculiar accelerations, instrumental effects, and survey selection;
it is an equation-level observable, not a forecast or measurement.
